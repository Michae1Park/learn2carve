"""Carve environment: rider + board on an infinite groomed slope (Stage 1; Stage 3 adds AMP on top).

Forked from Isaac Lab's ``contrib/humanoid_amp`` env. Changes:

- Rider = humanoid + snowboard articulation (envs/rider.py), scaled to the rider's mass.
- Slope, two ways (``CarveEnvCfg.mountain``), both infinite and edge-free:
  - False (Stage 1 default): flat frictionless ground, gravity tilted by the slope angle.
  - True: the ground plane itself is tilted, gravity points straight down, and the slope is dressed as a mountain
    (envs/mountain.py). This is what you see in the viewer.
  Either way the env works in the **slope frame**: x = downhill along the snow, y = across, z = snow normal. Every
  body state is read through ``body_pos`` / ``body_quat`` / ``body_lin_vel`` / ``body_ang_vel`` (slope frame) and
  written back through ``to_world``, so observations, rewards and the snow model are identical in both modes and a
  policy trained in one rides in the other. With tilted gravity the slope frame is the world frame.
- Snow = envs/snow.py forces on the board each physics step (PhysX only supplies the normal contact).
- Actions = joint targets around the snowboard stance (zero action holds the stance), not around mid-range.

Import this module only after Isaac Sim has launched (it pulls in USD).
"""

from __future__ import annotations

import math

import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from . import config
from .rider import rider_cfg
from .snow import SnowParams, air_drag, board_wrench, snow_wrench

STANCE_FILE = config.ROOT / "envs/assets/stance.yaml"  # the config.yaml binding angles


def stance_file(binding_angles: tuple[float, float] | None = None, tag: str = ""):
    """stance.yaml for the config's binding angles, stance_<front>_<back>.yaml for others (scripts/make_stance.py).
    A tag (e.g. "squat") picks a variant: stance[_<front>_<back>]_<tag>.yaml."""
    suffix = f"_{tag}" if tag else ""
    if binding_angles is None or list(binding_angles) == list(config.load()["rider"]["binding_angles_deg"]):
        return STANCE_FILE.with_name(f"stance{suffix}.yaml")
    front, back = binding_angles
    return STANCE_FILE.with_name(f"stance_{front:g}_{back:g}{suffix}.yaml")
KEY_BODIES = ["right_hand", "left_hand", "right_foot", "left_foot"]


def load_stance(binding_angles: tuple[float, float] | None = None, tag: str = "") -> dict:
    path = stance_file(binding_angles, tag)
    if not path.exists():
        hint = (f" --angles {binding_angles[0]:g} {binding_angles[1]:g}" if binding_angles else "") + (f" --tag {tag}" if tag else "")
        raise FileNotFoundError(f"{path} missing: run scripts/py scripts/make_stance.py{hint} first")
    return yaml.safe_load(path.read_text())


def tilted_gravity(angle_deg: float) -> tuple[float, float, float]:
    a = math.radians(angle_deg)
    return (9.81 * math.sin(a), 0.0, -9.81 * math.cos(a))


@configclass
class CarveSceneCfg(InteractiveSceneCfg):
    robot = None  # set in CarveEnvCfg.__post_init__ (needs the stance file)
    board_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/board")
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        collision_group=-1,
        spawn=sim_utils.GroundPlaneCfg(
            size=(2000.0, 2000.0),
            # frictionless: the snow model supplies all tangential forces. "min" wins over the bodies' materials.
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=0.0, dynamic_friction=0.0, restitution=0.0, friction_combine_mode="min"
            ),
        ),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class CarveEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4  # 120 Hz physics, 30 Hz policy

    observation_space = 81 + 6  # AMP humanoid observation + board/snow state
    action_space = 28
    state_space = 0
    action_scale = 0.5  # fraction of each joint's half-range that a ±1 action moves the target

    mountain: bool = False  # True: tilted snowy mountain + real gravity; False: flat ground + tilted gravity
    slope_deg: float | None = None  # None: config.yaml slope.angle_deg
    fall_height = 0.45  # pelvis height above the snow [m] below which the rider has fallen
    start_speed = (2.0, 5.0)  # m/s along the board at reset
    start_yaw_deg = 0.0  # ± board heading from the fall line at reset

    # rider model (defaults = the baseline 28-DoF rider at 120 Hz); see docs/HUMANOID_MODEL.md
    physics_hz: int = 120  # PhysX rate; the policy stays at 30 Hz (decimation = physics_hz / 30)
    solver_iterations: tuple[int, int] = (32, 4)  # PhysX (position, velocity) iterations
    thoracic: bool = False  # + 2-DoF thoracic joint (chest roll / pitch): 30 DoF
    neck_stiffness_scale: float = 1.0
    spine_damping_scale: float = 1.0  # abdomen + thorax damping × this (anti-jerk)
    pelvis_mass_frac: float | None = None  # pelvis share of the rider's mass (stock: 0.126); None = stock
    leg_action_scale: float | None = None  # hips + knees: fraction of half-range per unit action; None = action_scale
    knee_rotation: bool = False  # + ±20° axial twist per knee (2 DoF)
    # joint groups left to their drives (held at the stance target, still physical) instead of the policy: fewer
    # actions to explore. Any of PASSIVE_GROUPS: "ankle" (6, boots), "neck" (3), "knee:2" (2, knee twist)
    passive_joints: tuple[str, ...] = ()
    stiff_boots: bool = False  # ankles held near their stance angle, 3× stiffer (envs/rider.py BOOT_RANGE_DEG)
    compliant_drives: bool = False  # per-joint-group stiffness (envs/rider.py COMPLIANT) instead of uniform, stiff
    # board model (defaults = one rigid board); see docs/BOARD_MODEL.md
    board_flex: bool = False  # two halves + torsion/camber spring; progressive sidecut and flex in the snow model
    stance_tag: str = ""  # stance file variant, e.g. "squat" (scripts/make_stance.py --hip_height 0.5 --tag squat)
    binding_angles_deg: tuple[float, float] | None = None  # (front, back); None = config.yaml. Needs its stance file

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=PhysxCfg(gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23),
    )
    scene: CarveSceneCfg = CarveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root", asset_name="robot", env_index=0, eye=(-4.0, -4.0, 2.5), lookat=(0.0, 0.0, 0.5)
    )

    def __post_init__(self):
        self.scene.robot = rider_cfg(stance=load_stance())


# Joint groups the policy can leave passive (held by their drives at the stance target): DoF-name substring -> count
PASSIVE_GROUPS = {"ankle": 6, "neck": 3, "knee:2": 2}


def num_joints(cfg: CarveEnvCfg) -> int:
    """All rider DoF in the articulation, actuated or not (the board's spring DoFs included)."""
    return 28 + (2 if cfg.thoracic else 0) + (2 if cfg.knee_rotation else 0) + (2 if cfg.board_flex else 0)


def is_passive(cfg: CarveEnvCfg, joint: str) -> bool:
    return joint.startswith("board_flex") or any(g in joint for g in cfg.passive_joints)


def num_dof(cfg: CarveEnvCfg) -> int:
    """Actuated (policy-driven) DoF = the action size."""
    passive = sum(n for g, n in PASSIVE_GROUPS.items() if g in cfg.passive_joints
                  and (g != "knee:2" or cfg.knee_rotation))
    return num_joints(cfg) - (2 if cfg.board_flex else 0) - passive


def base_obs_size(cfg: CarveEnvCfg) -> int:
    """AMP humanoid observation (2 · all DoF + 25) + board/snow state (6, + 4 for a two-half board)."""
    return 2 * num_joints(cfg) + 25 + 6 + (4 if cfg.board_flex else 0)


def apply_rider_options(cfg: CarveEnvCfg) -> None:
    """Rebuild what depends on the rider options. Runs in __init__: --set overrides land after __post_init__."""
    if cfg.physics_hz % 30:
        raise ValueError(f"physics_hz must be a multiple of the 30 Hz policy rate, got {cfg.physics_hz}")
    cfg.decimation = cfg.physics_hz // 30
    cfg.sim.dt = 1.0 / cfg.physics_hz
    cfg.sim.render_interval = cfg.decimation
    cfg.action_space = num_dof(cfg)
    if cfg.observation_space == 81 + 6:  # the carve env's own default; subclasses set theirs from base_obs_size
        cfg.observation_space = base_obs_size(cfg)
    angles = tuple(cfg.binding_angles_deg) if cfg.binding_angles_deg is not None else None
    cfg.scene.robot = rider_cfg(
        stance=load_stance(angles, cfg.stance_tag), binding_angles=angles, solver_iterations=tuple(cfg.solver_iterations),
        thoracic=cfg.thoracic,
        neck_stiffness_scale=cfg.neck_stiffness_scale, spine_damping_scale=cfg.spine_damping_scale,
        compliant_drives=cfg.compliant_drives, board_flex=cfg.board_flex, stiff_boots=cfg.stiff_boots,
        knee_rotation=cfg.knee_rotation,
    )
    cfg.scene.board_contact.prim_path = "{ENV_REGEX_NS}/Robot/board" + (".*" if cfg.board_flex else "")


def slope_quat(angle_deg: float) -> tuple[float, float, float, float]:
    """Slope frame → world rotation (x, y, z, w): about +y, so the slope's x (downhill) points to +x and down."""
    a = math.radians(angle_deg)
    return (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))


class CarveEnv(DirectRLEnv):
    cfg: CarveEnvCfg

    def __init__(self, cfg: CarveEnvCfg, render_mode: str | None = None, **kwargs):
        apply_rider_options(cfg)
        # slope set here, not in __post_init__: --set overrides land after the cfg is constructed
        slope_deg = cfg.slope_deg if cfg.slope_deg is not None else config.load()["slope"]["angle_deg"]
        self.slope_deg = slope_deg
        if cfg.mountain:
            cfg.sim.gravity = (0.0, 0.0, -9.81)
            cfg.scene.ground.init_state.rot = slope_quat(slope_deg)
            cfg.scene.light.spawn.color = (0.62, 0.76, 1.0)  # blue sky (envs/mountain.py hides the grid, adds snow)
            cfg.scene.light.spawn.intensity = 1500.0  # sky fill; low enough that the sun shades the slopes
            q = slope_quat(slope_deg)
        else:
            cfg.sim.gravity = tilted_gravity(slope_deg)
            q = (0.0, 0.0, 0.0, 1.0)
        self.q_slope = torch.tensor(q, device=cfg.sim.device)  # slope frame → world
        self.q_slope_inv = self.q_slope * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=cfg.sim.device)
        super().__init__(cfg, render_mode, **kwargs)
        params = config.load()
        self.robot = self.scene["robot"]
        self.contact = self.scene["board_contact"]
        names = self.robot.data.body_names
        self.board_id = names.index("board")  # the whole board, or its nose half (frame at the board centre)
        self.board_ids = [self.board_id] + ([names.index("board_tail")] if self.cfg.board_flex else [])
        contact_names = self.contact.body_names
        self.contact_ids = [contact_names.index(names[i]) for i in self.board_ids]
        joints = self.robot.joint_names
        self.actuated = torch.tensor([i for i, j in enumerate(joints) if not is_passive(self.cfg, j)],
                                     device=self.device)
        assert len(self.actuated) == self.cfg.action_space, (len(self.actuated), self.cfg.action_space)
        self.bend_id = joints.index("board_flex:1") if self.cfg.board_flex else None
        self.pelvis_id = names.index("pelvis")
        self.torso_id = names.index("torso")
        self.key_ids = [names.index(n) for n in KEY_BODIES]
        self.up = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        self._scale_to_rider_mass(params["rider"]["mass"])
        self.snow = SnowParams.from_config(params["snow"], params["board"], params["rider"]["mass"])
        self.snow_info: dict[str, torch.Tensor] = {}
        # grip spring state (snow.py), one per board half
        self.slip_z = torch.zeros(self.num_envs, len(self.board_ids), device=self.device).squeeze(-1)
        if self.cfg.board_flex:
            from .rider import camber_angle

            self.camber_rest = camber_angle(params["board"]["effective_edge"])

        # stance: default joint targets, and the pelvis pose relative to the board
        stance = load_stance(tuple(self.cfg.binding_angles_deg) if self.cfg.binding_angles_deg is not None else None,
                             self.cfg.stance_tag)
        self.stance_pos = self.robot.data.default_joint_pos.torch.clone()
        self.pelvis_in_board_pos = torch.tensor(stance["pelvis_pos"], device=self.device)
        self.pelvis_in_board_quat = torch.tensor(stance["pelvis_quat"], device=self.device)
        self.board_half_thickness = params["board"]["thickness"] / 2
        self.board_half_width = params["board"]["waist_width"] / 2

        limits = self.robot.data.soft_joint_pos_limits.torch[0]
        self.joint_lower, self.joint_upper = limits[:, 0], limits[:, 1]
        scale = torch.full_like(self.joint_lower, self.cfg.action_scale)
        if self.cfg.leg_action_scale is not None:  # hips and knees drive the edge and the squat: give them more range
            legs = [i for i, j in enumerate(self.robot.joint_names) if "hip" in j or "knee" in j]
            scale[legs] = self.cfg.leg_action_scale
        self.action_scale = (scale * 0.5 * (self.joint_upper - self.joint_lower))[self.actuated]
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    def _scale_to_rider_mass(self, rider_mass: float) -> None:
        """The AMP humanoid is ~36 kg. Scale masses, inertias and joint gains so dynamics stay similar at 65 kg."""
        mass = self.robot.data.body_mass.torch.clone()
        human = [i for i in range(mass.shape[1]) if i not in self.board_ids]
        k = rider_mass / mass[0, human].sum().item()
        per_body = torch.full((mass.shape[1],), k, device=mass.device)
        per_body[self.board_ids] = 1.0
        if self.cfg.pelvis_mass_frac is not None:
            # heavy root: the pelvis takes this share of the rider's mass, the rest scale down to keep the total
            scaled = mass[0] * per_body
            others = [i for i in human if i != self.pelvis_id]
            per_body[self.pelvis_id] *= self.cfg.pelvis_mass_frac * rider_mass / scaled[self.pelvis_id].item()
            per_body[others] *= (1 - self.cfg.pelvis_mass_frac) * rider_mass / scaled[others].sum().item()
        mass *= per_body
        inertia = self.robot.data.body_inertia.torch.clone()
        inertia *= per_body.view(1, -1, 1)
        self.robot.set_masses_index(masses=mass)
        self.robot.set_inertias_index(inertias=inertia)
        gain = torch.full((len(self.robot.joint_names),), k, device=mass.device)
        gain[[i for i, j in enumerate(self.robot.joint_names) if j.startswith("board_flex")]] = 1.0  # board spring
        self.robot.write_joint_stiffness_to_sim_index(stiffness=self.robot.data.joint_stiffness.torch * gain)
        self.robot.write_joint_damping_to_sim_index(damping=self.robot.data.joint_damping.torch * gain)
        self.mass_scale = k

    def _setup_scene(self):
        if self.cfg.mountain:
            from .mountain import add_mountain

            add_mountain(self.slope_deg)

    # -- slope frame -----------------------------------------------------------------------------------------

    def to_slope(self, v: torch.Tensor) -> torch.Tensor:
        """World vectors/positions (..., 3) → slope frame. The slope passes through the world origin."""
        return quat_apply(self.q_slope_inv.expand(*v.shape[:-1], 4), v)

    def to_world(self, v: torch.Tensor) -> torch.Tensor:
        return quat_apply(self.q_slope.expand(*v.shape[:-1], 4), v)

    def quat_to_world(self, q: torch.Tensor) -> torch.Tensor:
        return quat_mul(self.q_slope.expand_as(q), q)

    def body_pos(self, ids) -> torch.Tensor:
        return self.to_slope(self.robot.data.body_pos_w.torch[:, ids])

    def body_quat(self, ids) -> torch.Tensor:
        q = self.robot.data.body_quat_w.torch[:, ids]
        return quat_mul(self.q_slope_inv.expand_as(q), q)

    def body_lin_vel(self, ids) -> torch.Tensor:
        return self.to_slope(self.robot.data.body_lin_vel_w.torch[:, ids])

    def body_ang_vel(self, ids) -> torch.Tensor:
        return self.to_slope(self.robot.data.body_ang_vel_w.torch[:, ids])

    # -- stepping ------------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        target = self.stance_pos.clone()  # the board's spring DoFs keep their rest target: the policy can't drive them
        target[:, self.actuated] += self.action_scale * self.actions
        target = target.clamp(self.joint_lower, self.joint_upper)
        self.robot.actuators.target_command.set_position_index(value=target)
        self._apply_snow()

    def _apply_snow(self):
        # all in the slope frame; the resulting forces go back to the world frame
        contact = self.contact.data.net_forces_w.torch[:, self.contact_ids]
        normal_force = (self.to_slope(contact) * self.up).sum(-1)  # (N, halves)
        if self.cfg.board_flex:
            bend = self.robot.data.joint_pos.torch[:, self.bend_id]  # -φ₀ at rest (camber), 0 pressed flat
            force, torque, self.snow_info = board_wrench(
                self.body_quat(self.board_ids), self.body_lin_vel(self.board_ids), self.body_ang_vel(self.board_ids),
                normal_force, self.up, self.physics_dt, self.snow, self.slip_z,
                camber_engaged=1.0 + bend / self.camber_rest,
            )
        else:
            force, torque, self.snow_info = snow_wrench(
                self.body_quat(self.board_id), self.body_lin_vel(self.board_id), self.body_ang_vel(self.board_id),
                normal_force[:, 0], self.up, self.physics_dt, self.snow, self.slip_z,
            )
            force, torque = force.unsqueeze(1), torque.unsqueeze(1)
        self.slip_z = self.snow_info["slip_z"]
        drag = air_drag(self.body_lin_vel(self.pelvis_id), self.snow).unsqueeze(1)
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=self.to_world(torch.cat([force, drag], dim=1)),
            torques=self.to_world(torch.cat([torque, torch.zeros_like(drag)], dim=1)),
            body_ids=self.board_ids + [self.pelvis_id],
            is_global=True,
        )

    # -- MDP -----------------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        data = self.robot.data
        obs = compute_obs(
            data.joint_pos.torch,
            data.joint_vel.torch,
            self.body_pos(self.torso_id),
            self.body_quat(self.torso_id),
            self.body_lin_vel(self.torso_id),
            self.body_ang_vel(self.torso_id),
            self.body_pos(self.key_ids),
        )
        info = self.snow_info or self._zero_snow_info()
        board = torch.stack(
            [
                info["edge_angle"],
                info["v_long"],
                info["v_lat"],
                (self.body_ang_vel(self.board_id) * self.up).sum(-1),
                info["normal_force"] / (9.81 * (self.snow.system_mass)),
                info["grip_usage"].clamp(max=5.0),
            ],
            dim=-1,
        )
        if self.cfg.board_flex:  # nose / tail edge angles (torsion), load on the nose, how flat the camber is
            board = torch.cat([board, info["edge_halves"], info["load_nose"].unsqueeze(-1),
                               info["camber_engaged"].unsqueeze(-1)], dim=-1)
        return {"policy": torch.cat([obs, board], dim=-1)}

    def _zero_snow_info(self) -> dict[str, torch.Tensor]:
        z = torch.zeros(self.num_envs, device=self.device)
        info = {k: z for k in ("edge_angle", "v_long", "v_lat", "normal_force", "grip_usage")}
        if self.cfg.board_flex:
            info |= {"edge_halves": torch.zeros(self.num_envs, 2, device=self.device), "load_nose": z + 0.5,
                     "camber_engaged": z}
        return info

    def _get_rewards(self) -> torch.Tensor:
        # Stage 3 task reward: stay upright and make progress downhill (capped so it doesn't just tuck)
        w = config.load()["train"]["task_reward"]
        torso_up = quat_apply(self.body_quat(self.torso_id), self.up.expand(self.num_envs, 3))
        upright = (torso_up * self.up).sum(-1).clamp(min=0.0)
        downhill = self.body_lin_vel(self.pelvis_id)[:, 0].clamp(0.0, w["v_max"]) / w["v_max"]
        return w["upright"] * upright + w["downhill"] * downhill

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = self.body_pos(self.pelvis_id)[:, 2] < self.cfg.fall_height
        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.slip_z[env_ids] = 0.0
        n = len(env_ids)

        # board = yaw about the snow normal, then roll about its own long axis (edge angle, θ < 0: heel edge down)
        yaw, edge, root_lin_vel, y0 = self._start_state(env_ids)
        q_yaw, q_roll = torch.zeros(n, 4, device=self.device), torch.zeros(n, 4, device=self.device)
        q_yaw[:, 2], q_yaw[:, 3] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        q_roll[:, 0], q_roll[:, 3] = torch.sin(edge / 2), torch.cos(edge / 2)
        board_quat = quat_mul(q_yaw, q_roll)
        board_pos = self.scene.env_origins[env_ids].clone()  # the env grid, laid out on the slope
        board_pos[:, 1] += y0
        # lowest edge just clear of the snow: lands within a step
        board_pos[:, 2] = self.board_half_width * edge.sin().abs() + self.board_half_thickness * edge.cos() + 0.0005

        # pelvis = board ∘ stance offset
        pelvis_pos = board_pos + quat_apply(board_quat, self.pelvis_in_board_pos.expand(n, 3))
        pelvis_quat = quat_mul(board_quat, self.pelvis_in_board_quat.expand(n, 4))
        root_vel = torch.zeros(n, 6, device=self.device)
        root_vel[:, :3] = self.to_world(root_lin_vel)

        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([self.to_world(pelvis_pos), self.quat_to_world(pelvis_quat)], dim=-1), env_ids=env_ids
        )
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=self.stance_pos[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self.stance_pos[env_ids]), env_ids=env_ids
        )

    def _start_state(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Board yaw [rad] (0 = nose down the fall line, +x), edge angle [rad], velocity (n, 3, slope frame) and
        sideways offset from the env origin [m] for the envs being reset.

        Default: flat, heading down the fall line ± start_yaw_deg, moving nose-first at start_speed. Override for
        other starts (e.g. sideways on the heel edge for a side slip).
        """
        n = len(env_ids)
        yaw = math.radians(self.cfg.start_yaw_deg) * (2 * torch.rand(n, device=self.device) - 1)
        lo, hi = self.cfg.start_speed
        speed = lo + (hi - lo) * torch.rand(n, device=self.device)
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        return yaw, torch.zeros_like(yaw), speed.unsqueeze(-1) * heading, torch.zeros_like(yaw)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    """The AMP humanoid observation (same layout as Isaac Lab's humanoid_amp, so the AMP discriminator matches)."""
    return torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
