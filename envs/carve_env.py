"""Carve environment: rider + board on an infinite groomed slope (Stage 1; Stage 3 adds AMP on top).

Forked from Isaac Lab's ``contrib/humanoid_amp`` env. Changes:

- Rider = humanoid + snowboard articulation (envs/rider.py), scaled to the rider's mass.
- Slope = flat frictionless ground with gravity tilted by ``slope.angle_deg``: downhill is +x, the snow normal is
  +z. Infinite and edge-free, so riders can go as far as they like.
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
        cfg = config.load()
        self.sim.gravity = tilted_gravity(cfg["slope"]["angle_deg"])
        self.scene.robot = rider_cfg(stance=load_stance())


class CarveEnv(DirectRLEnv):
    cfg: CarveEnvCfg

    def __init__(self, cfg: CarveEnvCfg, render_mode: str | None = None, **kwargs):
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

    # -- stepping ------------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        target = (self.stance_pos + self.action_scale * self.actions).clamp(self.joint_lower, self.joint_upper)
        self.robot.actuators.target_command.set_position_index(value=target)
        self._apply_snow()

    def _apply_snow(self):
        data = self.robot.data
        normal_force = (self.contact.data.net_forces_w.torch[:, 0] * self.up).sum(-1)
        force, torque, self.snow_info = snow_wrench(
            data.body_quat_w.torch[:, self.board_id],
            data.body_lin_vel_w.torch[:, self.board_id],
            data.body_ang_vel_w.torch[:, self.board_id],
            normal_force,
            self.up,
            self.physics_dt,
            self.snow,
            self.slip_z,
        )
        self.slip_z = self.snow_info["slip_z"]
        drag = air_drag(data.body_lin_vel_w.torch[:, self.pelvis_id], self.snow)
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=torch.stack([force, drag], dim=1),
            torques=torch.stack([torque, torch.zeros_like(drag)], dim=1),
            body_ids=[self.board_id, self.pelvis_id],
            is_global=True,
        )

    # -- MDP -----------------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        data = self.robot.data
        obs = compute_obs(
            data.joint_pos.torch,
            data.joint_vel.torch,
            data.body_pos_w.torch[:, self.torso_id],
            data.body_quat_w.torch[:, self.torso_id],
            data.body_lin_vel_w.torch[:, self.torso_id],
            data.body_ang_vel_w.torch[:, self.torso_id],
            data.body_pos_w.torch[:, self.key_ids],
        )
        info = self.snow_info or self._zero_snow_info()
        board = torch.stack(
            [
                info["edge_angle"],
                info["v_long"],
                info["v_lat"],
                (data.body_ang_vel_w.torch[:, self.board_id] * self.up).sum(-1),
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
        torso_up = quat_apply(self.robot.data.body_quat_w.torch[:, self.torso_id], self.up.expand(self.num_envs, 3))
        upright = (torso_up * self.up).sum(-1).clamp(min=0.0)
        downhill = self.robot.data.body_lin_vel_w.torch[:, self.pelvis_id, 0].clamp(0.0, w["v_max"]) / w["v_max"]
        return w["upright"] * upright + w["downhill"] * downhill

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = self.robot.data.body_pos_w.torch[:, self.pelvis_id, 2] < self.cfg.fall_height
        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.slip_z[env_ids] = 0.0
        n = len(env_ids)

        # board on the snow at the env origin, heading down the fall line (+x) ± start_yaw
        yaw = math.radians(self.cfg.start_yaw_deg) * (2 * torch.rand(n, device=self.device) - 1)
        board_quat = torch.zeros(n, 4, device=self.device)
        board_quat[:, 2], board_quat[:, 3] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        board_pos = self.scene.env_origins[env_ids].clone()
        board_pos[:, 2] = self.board_half_thickness + 0.005

        # pelvis = board ∘ stance offset
        pelvis_pos = board_pos + quat_apply(board_quat, self.pelvis_in_board_pos.expand(n, 3))
        pelvis_quat = quat_mul(board_quat, self.pelvis_in_board_quat.expand(n, 4))
        lo, hi = self.cfg.start_speed
        speed = lo + (hi - lo) * torch.rand(n, device=self.device)
        heading = quat_apply(board_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(n, 3))
        root_vel = torch.zeros(n, 6, device=self.device)
        root_vel[:, :3] = speed.unsqueeze(-1) * heading

        self.robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([pelvis_pos, pelvis_quat], dim=-1), env_ids=env_ids
        )
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=self.stance_pos[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self.stance_pos[env_ids]), env_ids=env_ids
        )


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
