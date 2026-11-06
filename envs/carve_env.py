"""Rider + board on an infinite groomed slope, forked from Isaac Lab's ``humanoid_amp`` env.

Two ways to make the slope (``CarveEnvCfg.mountain``): tilted gravity over flat ground, or a tilted ground plane
dressed as a mountain. Either way every body state is read and written in the slope frame (x downhill, y across,
z snow normal), so a policy trained in one rides in the other. Actions are joint targets around the stance.

Import after Isaac Sim launched.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

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

from . import physical
from . import rider as amp_rider
from .rider import camber_angle, rider_cfg, rider_scale
from .snow import SnowParams, air_drag, board_wrench, snow_wrench

GRAVITY = 9.81
POLICY_HZ = 30
DEFAULT_SLOPE_DEG = 9.0  # the demo's stage 3 slope
KEY_BODIES = ["right_hand", "left_hand", "right_foot", "left_foot"]
STOCK_JOINTS = 28
AMP_OBS_EXTRA = 25  # root height, tangent/normal, root velocities, key body offsets
BOARD_OBS = 6
FLEX_BOARD_OBS = 4
SYNERGIES = 2
PASSIVE_GROUPS = {"ankle": 6, "neck": 3, "knee:2": 2}  # DoF-name substring -> DoF count
IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


def profile(cfg) -> SimpleNamespace:
    """Robot-specific names and counts: the edited AMP humanoid or the stock Unitree G1."""
    if cfg.robot == "g1":
        from . import rider_g1 as g1

        return SimpleNamespace(rider_cfg=g1.rider_cfg, stock_joints=g1.STOCK_JOINTS, passive=g1.PASSIVE_GROUPS,
                               torso=g1.TORSO, key_bodies=g1.KEY_BODIES, rescale=False)
    if cfg.robot != "amp":
        raise ValueError(f"robot must be amp or g1, got {cfg.robot}")
    return SimpleNamespace(rider_cfg=amp_rider.rider_cfg, stock_joints=STOCK_JOINTS, passive=PASSIVE_GROUPS,
                           torso="torso", key_bodies=KEY_BODIES, rescale=True)


def stance_file(binding_angles: tuple[float, float] | None = None, tag: str = ""):
    front, back = binding_angles if binding_angles is not None else physical.load()["rider"]["binding_angles_deg"]
    suffix = f"_{tag}" if tag else ""
    return physical.stance_path(f"stance_{front:g}_{back:g}{suffix}.yaml")


def load_stance(binding_angles: tuple[float, float] | None = None, tag: str = "") -> dict:
    path = stance_file(binding_angles, tag)
    if not path.exists():
        hint = (f" --angles {binding_angles[0]:g} {binding_angles[1]:g}" if binding_angles else "") + (f" --tag {tag}" if tag else "")
        raise FileNotFoundError(f"{path} missing: run scripts/py scripts/make_stance.py{hint} first")
    return yaml.safe_load(path.read_text())


def tilted_gravity(angle_deg: float) -> tuple[float, float, float]:
    a = math.radians(angle_deg)
    return (GRAVITY * math.sin(a), 0.0, -GRAVITY * math.cos(a))


def slope_quat(angle_deg: float) -> tuple[float, float, float, float]:
    a = math.radians(angle_deg)
    return (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))


@configclass
class CarveSceneCfg(InteractiveSceneCfg):
    robot = None  # set from the stance in CarveEnvCfg
    board_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/board")
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        collision_group=-1,
        spawn=sim_utils.GroundPlaneCfg(
            size=(2000.0, 2000.0),
            # the snow model supplies all tangential forces; "min" overrides the bodies' own friction
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=0.0, dynamic_friction=0.0, restitution=0.0, friction_combine_mode="min"
            ),
        ),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class CarveEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4

    observation_space = 81 + 6  # recomputed from the rider options
    action_space = 28
    state_space = 0
    action_scale = 0.5  # fraction of each joint's half-range per unit action

    mountain: bool = False
    slope_deg: float = DEFAULT_SLOPE_DEG  # stage 2 rides 7, set by its script
    fall_height = 0.45  # m, pelvis above the snow
    start_speed = (2.0, 5.0)  # m/s
    start_yaw_deg = 0.0

    # rider model, see docs/HUMANOID_MODEL.md
    robot: str = "amp"  # amp (edited AMP humanoid) or g1 (stock Unitree G1, envs/rider_g1.py, envs/physical_g1.yaml)
    physics_hz: int = 120
    solver_iterations: tuple[int, int] = (32, 4)
    thoracic: bool = False
    neck_stiffness_scale: float = 1.0
    spine_damping_scale: float = 1.0
    pelvis_mass_frac: float | None = None
    arm_action_scale: float | None = None  # not reduced by synergy_residual
    arm_stiffness_scale: float = 1.0
    leg_action_scale: float | None = None
    free_action_scale: dict = {}  # {joint-name substring: scale}, like arm_action_scale: not reduced by synergy_residual
    knee_rotation: bool = False
    passive_joints: tuple[str, ...] = ()  # PASSIVE_GROUPS left to their drives at the stance target

    # synergy actions: two extra actions ahead of the joints, which become a residual at synergy_residual
    synergy: bool = False
    synergy_edge_rad: dict = {"left_ankle_y": 0.5, "right_ankle_y": 0.5, "left_knee": 0.6, "right_knee": 0.6}
    synergy_lean_rad: dict = {"abdomen_x": 0.9}
    synergy_residual: float = 0.3
    joint_smooth: float = 1.0  # EMA weight of a new action; 1 = none
    synergy_smooth: float = 1.0
    synergy_heel_rad: dict = {}  # added × edge⁻, so the hips sit back behind the heel edge
    synergy_toe_rad: dict = {}  # added × edge⁺, so the toe edge comes from the knees, not the ankles

    boot_range_deg: dict = {}
    rider_to_height: bool = False
    stance_width: float | None = None
    torso_twist_deg: float = 0.0
    ankle_range_deg: dict = {}
    stiff_boots: bool = False
    compliant_drives: bool = False
    board_flex: bool = False  # see docs/BOARD_MODEL.md
    stance_tag: str = ""
    binding_angles_deg: tuple[float, float] | None = None

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
        # placeholder rider; apply_rider_options rebuilds it once the overrides (robot, stance tag) are in. A saved
        # run's physical.yaml can name binding angles with no untagged stance (a G1 run's 60/50), so it may be absent
        try:
            self.scene.robot = rider_cfg(stance=load_stance())
        except FileNotFoundError:
            self.scene.robot = None


def configured_binding_angles(cfg: CarveEnvCfg) -> tuple[float, float] | None:
    return tuple(cfg.binding_angles_deg) if cfg.binding_angles_deg is not None else None


def num_joints(cfg: CarveEnvCfg) -> int:
    """Every DoF in the articulation, the board's spring included."""
    extra = sum(2 for enabled in (cfg.thoracic, cfg.knee_rotation, cfg.board_flex) if enabled)
    return profile(cfg).stock_joints + extra


def is_passive(cfg: CarveEnvCfg, joint: str) -> bool:
    return joint.startswith("board_flex") or any(group in joint for group in cfg.passive_joints)


def num_dof(cfg: CarveEnvCfg) -> int:
    """Policy-driven DoF."""
    passive = sum(count for group, count in profile(cfg).passive.items()
                  if group in cfg.passive_joints and (group != "knee:2" or cfg.knee_rotation))
    board_spring = 2 if cfg.board_flex else 0
    return num_joints(cfg) - board_spring - passive


def base_obs_size(cfg: CarveEnvCfg) -> int:
    return 2 * num_joints(cfg) + AMP_OBS_EXTRA + BOARD_OBS + (FLEX_BOARD_OBS if cfg.board_flex else 0)


def apply_rider_options(cfg: CarveEnvCfg) -> None:
    """Runs in __init__ because --set overrides land after __post_init__."""
    if cfg.physics_hz % POLICY_HZ:
        raise ValueError(f"physics_hz must be a multiple of the {POLICY_HZ} Hz policy rate, got {cfg.physics_hz}")
    cfg.decimation = cfg.physics_hz // POLICY_HZ
    cfg.sim.dt = 1.0 / cfg.physics_hz
    cfg.sim.render_interval = cfg.decimation
    cfg.action_space = num_dof(cfg) + (SYNERGIES if cfg.synergy else 0)
    if cfg.observation_space == 81 + 6:
        cfg.observation_space = base_obs_size(cfg)
    physical.select(cfg.robot)
    cfg.scene.robot = profile(cfg).rider_cfg(
        stance=load_stance(configured_binding_angles(cfg), cfg.stance_tag),
        binding_angles=configured_binding_angles(cfg),
        solver_iterations=tuple(cfg.solver_iterations),
        stance_width=cfg.stance_width,
        scale_factor=rider_scale() if cfg.rider_to_height else 1.0,
        thoracic=cfg.thoracic,
        knee_rotation=cfg.knee_rotation,
        board_flex=cfg.board_flex,
        stiff_boots=cfg.stiff_boots,
        compliant_drives=cfg.compliant_drives,
        neck_stiffness_scale=cfg.neck_stiffness_scale,
        spine_damping_scale=cfg.spine_damping_scale,
        arm_stiffness_scale=cfg.arm_stiffness_scale,
        ankle_range_deg=dict(cfg.ankle_range_deg),
        torso_twist_deg=cfg.torso_twist_deg,
        boot_range_deg={axis: tuple(r) for axis, r in cfg.boot_range_deg.items()},
    )
    cfg.scene.board_contact.prim_path = "{ENV_REGEX_NS}/Robot/board" + (".*" if cfg.board_flex else "")


def apply_slope(cfg: CarveEnvCfg, slope_deg: float) -> tuple[float, float, float, float]:
    """Set gravity, ground and light for the slope; returns the slope frame → world rotation."""
    if not cfg.mountain:
        cfg.sim.gravity = tilted_gravity(slope_deg)
        return IDENTITY_XYZW
    cfg.sim.gravity = (0.0, 0.0, -GRAVITY)
    cfg.scene.ground.init_state.rot = slope_quat(slope_deg)
    cfg.scene.light.spawn.color = (0.62, 0.76, 1.0)
    cfg.scene.light.spawn.intensity = 1500.0  # low enough that the sun shades the slopes
    return slope_quat(slope_deg)


def joint_index(joints: list[str], name: str) -> int:
    """A hinge that became a D6 (knee_rotation) is addressed by its first DoF."""
    return joints.index(name if name in joints else f"{name}:0")


class CarveEnv(DirectRLEnv):
    cfg: CarveEnvCfg

    def __init__(self, cfg: CarveEnvCfg, render_mode: str | None = None, **kwargs):
        apply_rider_options(cfg)
        self.slope_deg = cfg.slope_deg
        self.q_slope = torch.tensor(apply_slope(cfg, self.slope_deg), device=cfg.sim.device)
        self.q_slope_inv = self.q_slope * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=cfg.sim.device)
        super().__init__(cfg, render_mode, **kwargs)

        params = physical.load()
        self.robot = self.scene["robot"]
        self.contact = self.scene["board_contact"]
        self.up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._find_bodies()
        self._scale_to_rider_mass(params["rider"]["mass"])

        self.snow = SnowParams.from_config(params["snow"], params["board"], params["rider"]["mass"])
        self.snow_info: dict[str, torch.Tensor] = {}
        self.slip_z = torch.zeros(self.num_envs, len(self.board_ids), device=self.device).squeeze(-1)
        if self.cfg.board_flex:
            self.camber_rest = camber_angle(params["board"]["effective_edge"])

        stance = load_stance(configured_binding_angles(self.cfg), self.cfg.stance_tag)
        self.stance_pos = self.robot.data.default_joint_pos.torch.clone()
        self.pelvis_in_board_pos = torch.tensor(stance["pelvis_pos"], device=self.device)
        self.pelvis_in_board_quat = torch.tensor(stance["pelvis_quat"], device=self.device)
        self.board_half_thickness = params["board"]["thickness"] / 2
        self.board_half_width = params["board"]["waist_width"] / 2

        self._setup_actions()

    def _find_bodies(self):
        bodies = self.robot.data.body_names
        joints = self.robot.joint_names
        self.board_id = bodies.index("board")
        self.board_ids = [self.board_id] + ([bodies.index("board_tail")] if self.cfg.board_flex else [])
        self.contact_ids = [self.contact.body_names.index(bodies[i]) for i in self.board_ids]
        self.pelvis_id = bodies.index("pelvis")
        robot = profile(self.cfg)
        self.torso_id = bodies.index(robot.torso)
        self.key_ids = [bodies.index(name) for name in robot.key_bodies]
        self.actuated = torch.tensor([i for i, j in enumerate(joints) if not is_passive(self.cfg, j)],
                                     device=self.device)
        assert len(self.actuated) == num_dof(self.cfg), (len(self.actuated), num_dof(self.cfg))
        self.bend_id = joints.index("board_flex:1") if self.cfg.board_flex else None

    def _setup_actions(self):
        joints = self.robot.joint_names
        limits = self.robot.data.soft_joint_pos_limits.torch[0]
        self.joint_lower, self.joint_upper = limits[:, 0], limits[:, 1]
        scale = torch.full_like(self.joint_lower, self.cfg.action_scale)
        if self.cfg.leg_action_scale is not None:
            scale[[i for i, j in enumerate(joints) if "hip" in j or "knee" in j]] = self.cfg.leg_action_scale
        self.arm_joint = torch.tensor(["shoulder" in j or "elbow" in j for j in joints], device=self.device)
        if self.cfg.arm_action_scale is not None:
            scale[self.arm_joint] = self.cfg.arm_action_scale
        self.free_joint = self.arm_joint & (self.cfg.arm_action_scale is not None)
        for part, value in self.cfg.free_action_scale.items():
            mask = torch.tensor([part in j for j in joints], device=self.device)
            scale[mask] = value
            self.free_joint |= mask
        self.action_scale = (scale * 0.5 * (self.joint_upper - self.joint_lower))[self.actuated]
        self.actions = torch.zeros(self.num_envs, num_dof(self.cfg), device=self.device)
        self.synergy = torch.zeros(self.num_envs, SYNERGIES, device=self.device)
        if not self.cfg.synergy:
            return

        def joint_vector(table: dict) -> torch.Tensor:
            vector = torch.zeros(len(joints), device=self.device)
            for name, rad in table.items():
                vector[joint_index(joints, name)] = rad
            return vector

        self.synergy_rad = torch.stack([joint_vector(self.cfg.synergy_edge_rad), joint_vector(self.cfg.synergy_lean_rad)])
        self.toe_rad = joint_vector(self.cfg.synergy_toe_rad)
        self.heel_rad = joint_vector(self.cfg.synergy_heel_rad)

    def _scale_to_rider_mass(self, rider_mass: float) -> None:
        """The AMP humanoid is ~36 kg: scale masses, inertias and joint gains to the rider's mass. A stock robot keeps its own."""
        if not profile(self.cfg).rescale:
            self.mass_scale = 1.0
            return
        mass = self.robot.data.body_mass.torch.clone()
        human = [i for i in range(mass.shape[1]) if i not in self.board_ids]
        k = rider_mass / mass[0, human].sum().item()
        per_body = torch.full((mass.shape[1],), k, device=mass.device)
        per_body[self.board_ids] = 1.0
        if self.cfg.pelvis_mass_frac is not None:
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
        gain[[i for i, j in enumerate(self.robot.joint_names) if j.startswith("board_flex")]] = 1.0
        self.robot.write_joint_stiffness_to_sim_index(stiffness=self.robot.data.joint_stiffness.torch * gain)
        self.robot.write_joint_damping_to_sim_index(damping=self.robot.data.joint_damping.torch * gain)
        self.mass_scale = k

    def _setup_scene(self):
        if self.cfg.mountain:
            from .mountain import add_mountain

            add_mountain(self.slope_deg)

    # -- slope frame ------------------------------------------------------------------------------------------------

    def to_slope(self, v: torch.Tensor) -> torch.Tensor:
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

    def edge_angle(self) -> torch.Tensor:
        return self.snow_info["edge_angle"] if self.snow_info else torch.zeros(self.num_envs, device=self.device)

    # -- stepping ---------------------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.clamp(-1.0, 1.0)
        if self.cfg.synergy:
            synergy, actions = actions[:, :SYNERGIES], actions[:, SYNERGIES:]
            self.synergy = smooth(self.synergy, synergy, self.cfg.synergy_smooth)
        self.actions = smooth(self.actions, actions, self.cfg.joint_smooth)

    def _apply_action(self):
        target = self.stance_pos.clone()
        if self.cfg.synergy:
            residual = torch.where(self.free_joint[self.actuated], 1.0, self.cfg.synergy_residual)
            target[:, self.actuated] += residual * self.action_scale * self.actions
            edge = self.synergy[:, :1]
            target += self.synergy @ self.synergy_rad
            target += (-edge).clamp(min=0.0) * self.heel_rad
            target += edge.clamp(min=0.0) * self.toe_rad
        else:
            target[:, self.actuated] += self.action_scale * self.actions
        target = target.clamp(self.joint_lower, self.joint_upper)
        self.robot.actuators.target_command.set_position_index(value=target)
        self._apply_snow()

    def _apply_snow(self):
        contact = self.contact.data.net_forces_w.torch[:, self.contact_ids]
        normal_force = (self.to_slope(contact) * self.up).sum(-1)
        if self.cfg.board_flex:
            bend = self.robot.data.joint_pos.torch[:, self.bend_id]  # -camber at rest, 0 pressed flat
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

    # -- MDP --------------------------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        data = self.robot.data
        amp = compute_obs(
            data.joint_pos.torch,
            data.joint_vel.torch,
            self.body_pos(self.torso_id),
            self.body_quat(self.torso_id),
            self.body_lin_vel(self.torso_id),
            self.body_ang_vel(self.torso_id),
            self.body_pos(self.key_ids),
        )
        return {"policy": torch.cat([amp, self._board_obs()], dim=-1)}

    def _board_obs(self) -> torch.Tensor:
        info = self.snow_info or self._zero_snow_info()
        board = torch.stack(
            [
                info["edge_angle"],
                info["v_long"],
                info["v_lat"],
                (self.body_ang_vel(self.board_id) * self.up).sum(-1),
                info["normal_force"] / (GRAVITY * self.snow.system_mass),
                info["grip_usage"].clamp(max=5.0),
            ],
            dim=-1,
        )
        if not self.cfg.board_flex:
            return board
        return torch.cat([board, info["edge_halves"], info["load_nose"].unsqueeze(-1),
                          info["camber_engaged"].unsqueeze(-1)], dim=-1)

    def _zero_snow_info(self) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(self.num_envs, device=self.device)
        info = {k: zeros for k in ("edge_angle", "v_long", "v_lat", "normal_force", "grip_usage")}
        if self.cfg.board_flex:
            info |= {"edge_halves": torch.zeros(self.num_envs, 2, device=self.device), "load_nose": zeros + 0.5,
                     "camber_engaged": zeros}
        return info

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

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

        yaw, edge, lin_vel, y_offset = self._start_state(env_ids)
        board_quat = quat_mul(yaw_quat(yaw), roll_quat(edge))
        board_pos = self.scene.env_origins[env_ids].clone()
        board_pos[:, 1] += y_offset
        lowest_edge_clearance = 0.0005
        board_pos[:, 2] = (self.board_half_width * edge.sin().abs() + self.board_half_thickness * edge.cos()
                           + lowest_edge_clearance)

        pelvis_pos = board_pos + quat_apply(board_quat, self.pelvis_in_board_pos.expand(n, 3))
        pelvis_quat = quat_mul(board_quat, self.pelvis_in_board_quat.expand(n, 4))
        root_vel = torch.zeros(n, 6, device=self.device)
        root_vel[:, :3] = self.to_world(lin_vel)

        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([self.to_world(pelvis_pos), self.quat_to_world(pelvis_quat)], dim=-1), env_ids=env_ids
        )
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=self.stance_pos[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self.stance_pos[env_ids]), env_ids=env_ids
        )

    def _start_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(board yaw, edge angle, slope-frame velocity, sideways offset) for the envs being reset."""
        n = len(env_ids)
        yaw = math.radians(self.cfg.start_yaw_deg) * (2 * torch.rand(n, device=self.device) - 1)
        lo, hi = self.cfg.start_speed
        speed = lo + (hi - lo) * torch.rand(n, device=self.device)
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        return yaw, torch.zeros_like(yaw), speed.unsqueeze(-1) * heading, torch.zeros_like(yaw)


def smooth(previous: torch.Tensor, new: torch.Tensor, weight: float) -> torch.Tensor:
    return new if weight >= 1.0 else weight * new + (1.0 - weight) * previous


def yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(*angle.shape, 4, device=angle.device)
    q[..., 2], q[..., 3] = torch.sin(angle / 2), torch.cos(angle / 2)
    return q


def roll_quat(angle: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(*angle.shape, 4, device=angle.device)
    q[..., 0], q[..., 3] = torch.sin(angle / 2), torch.cos(angle / 2)
    return q


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
    """Same layout as Isaac Lab's humanoid_amp, so an AMP discriminator matches."""
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
