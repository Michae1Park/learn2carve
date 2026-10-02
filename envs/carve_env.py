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
from .snow import SnowParams, air_drag, snow_wrench

STANCE_FILE = config.ROOT / "envs/assets/stance.yaml"
KEY_BODIES = ["right_hand", "left_hand", "right_foot", "left_foot"]


def load_stance() -> dict:
    if not STANCE_FILE.exists():
        raise FileNotFoundError(f"{STANCE_FILE} missing: run scripts/py scripts/make_stance.py first")
    return yaml.safe_load(STANCE_FILE.read_text())


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


def slope_quat(angle_deg: float) -> tuple[float, float, float, float]:
    """Slope frame → world rotation (x, y, z, w): about +y, so the slope's x (downhill) points to +x and down."""
    a = math.radians(angle_deg)
    return (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))


class CarveEnv(DirectRLEnv):
    cfg: CarveEnvCfg

    def __init__(self, cfg: CarveEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.board_id = names.index("board")
        self.pelvis_id = names.index("pelvis")
        self.torso_id = names.index("torso")
        self.key_ids = [names.index(n) for n in KEY_BODIES]
        self.up = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        self._scale_to_rider_mass(params["rider"]["mass"])
        self.snow = SnowParams.from_config(params["snow"], params["board"], params["rider"]["mass"])
        self.snow_info: dict[str, torch.Tensor] = {}
        self.slip_z = torch.zeros(self.num_envs, device=self.device)  # grip spring state (snow.py)

        # stance: default joint targets, and the pelvis pose relative to the board
        stance = load_stance()
        self.stance_pos = self.robot.data.default_joint_pos.torch.clone()
        self.pelvis_in_board_pos = torch.tensor(stance["pelvis_pos"], device=self.device)
        self.pelvis_in_board_quat = torch.tensor(stance["pelvis_quat"], device=self.device)
        self.board_half_thickness = params["board"]["thickness"] / 2
        self.board_half_width = params["board"]["waist_width"] / 2

        limits = self.robot.data.soft_joint_pos_limits.torch[0]
        self.joint_lower, self.joint_upper = limits[:, 0], limits[:, 1]
        self.action_scale = self.cfg.action_scale * 0.5 * (self.joint_upper - self.joint_lower)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    def _scale_to_rider_mass(self, rider_mass: float) -> None:
        """The AMP humanoid is ~36 kg. Scale masses, inertias and joint gains so dynamics stay similar at 65 kg."""
        mass = self.robot.data.body_mass.torch.clone()
        human = [i for i in range(mass.shape[1]) if i != self.board_id]
        k = rider_mass / mass[0, human].sum().item()
        mass[:, human] *= k
        inertia = self.robot.data.body_inertia.torch.clone()
        inertia[:, human] *= k
        self.robot.set_masses_index(masses=mass)
        self.robot.set_inertias_index(inertias=inertia)
        self.robot.write_joint_stiffness_to_sim_index(stiffness=self.robot.data.joint_stiffness.torch * k)
        self.robot.write_joint_damping_to_sim_index(damping=self.robot.data.joint_damping.torch * k)
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
        target = (self.stance_pos + self.action_scale * self.actions).clamp(self.joint_lower, self.joint_upper)
        self.robot.actuators.target_command.set_position_index(value=target)
        self._apply_snow()

    def _apply_snow(self):
        # all in the slope frame; the resulting forces go back to the world frame
        normal_force = (self.to_slope(self.contact.data.net_forces_w.torch[:, 0]) * self.up).sum(-1)
        force, torque, self.snow_info = snow_wrench(
            self.body_quat(self.board_id),
            self.body_lin_vel(self.board_id),
            self.body_ang_vel(self.board_id),
            normal_force,
            self.up,
            self.physics_dt,
            self.snow,
            self.slip_z,
        )
        self.slip_z = self.snow_info["slip_z"]
        drag = air_drag(self.body_lin_vel(self.pelvis_id), self.snow)
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=self.to_world(torch.stack([force, drag], dim=1)),
            torques=self.to_world(torch.stack([torque, torch.zeros_like(drag)], dim=1)),
            body_ids=[self.board_id, self.pelvis_id],
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
        return {"policy": torch.cat([obs, board], dim=-1)}

    def _zero_snow_info(self) -> dict[str, torch.Tensor]:
        z = torch.zeros(self.num_envs, device=self.device)
        return {k: z for k in ("edge_angle", "v_long", "v_lat", "normal_force", "grip_usage")}

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
