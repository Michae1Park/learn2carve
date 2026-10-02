"""Go2 velocity-tracking env with an optional gait clock.

Task: follow a random (vx, vy, yaw_rate) command on flat ground. Actions are joint position targets
around the default stance.

Gaits:
  free             no clock; the gait emerges from the rewards (usually a trot)
  walk/trot/pace   a clock gives each foot a phase; feet are rewarded for touching the ground only in their
                   stance part of the cycle. Gaits differ only in per-foot phase offsets (see GAITS).

Import this module only after Isaac Sim has launched.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, index_fill_, replace
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Per-foot phase offsets (FL, FR, RL, RR) in cycles, stance fraction ("duty"), and cycle frequency [Hz].
# A foot is in stance while (clock + offset) mod 1 < duty.
GAITS = {
    "walk": dict(offsets=(0.25, 0.75, 0.0, 0.5), duty=0.75, freq=1.5),  # 4-beat: RL, FL, RR, FR; 3 feet down
    "trot": dict(offsets=(0.0, 0.5, 0.5, 0.0), duty=0.5, freq=2.0),  # diagonal pairs together
    "pace": dict(offsets=(0.0, 0.5, 0.0, 0.5), duty=0.5, freq=2.0),  # same-side pairs together
}


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.14, 0.18)),  # dark, so the white Go2 stands out
    )
    robot = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4  # policy at 50 Hz, physics at 200 Hz
    action_space = 12
    observation_space = 56  # 48 proprioception + 8 clock (sin, cos per foot)
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # free camera starting with an overview of the env grid (centred on the origin); orbit/zoom in the client.
    # To follow one robot instead: ViewerCfg(eye=(1.5, 1.5, 0.8), origin_type="asset_root", asset_name="robot")
    viewer: ViewerCfg = ViewerCfg(eye=(30.0, 30.0, 20.0), lookat=(0.0, 0.0, 0.0))

    gait: str = "free"  # free | walk | trot | pace
    action_scale = 0.25  # rad per unit action
    action_clip = 5.0
    base_height_target = 0.30  # m; the default pose sags to ~0.29 under gravity

    # command ranges: forward [m/s], sideways [m/s], yaw rate [rad/s]
    cmd_vx = (-1.0, 1.0)
    cmd_vy = (-0.5, 0.5)
    cmd_yaw = (-1.0, 1.0)

    # reward weights (each term is multiplied by step_dt)
    rew_alive = 1.0  # per second survived; without it the penalties make falling early the best option
    rew_track_lin_vel = 1.5  # exp(-|v_xy error|² / 0.25)
    rew_track_yaw = 0.75  # exp(-yaw rate error² / 0.25)
    rew_lin_vel_z = -2.0  # bouncing
    rew_ang_vel_xy = -0.05  # rolling / pitching
    rew_torque = -2e-4
    rew_joint_acc = -2.5e-7
    rew_action_rate = -0.01  # jerky actions
    rew_air_time = 0.25  # long steps (rewarded at touchdown, only when commanded to move)
    rew_orientation = -2.5  # body tilt
    rew_base_height = -30.0  # crouching
    rew_gait = 1.0  # fraction of feet whose contact matches the clock (clock gaits only)


class Go2Env(DirectRLEnv):
    cfg: Go2EnvCfg

    def __init__(self, cfg: Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.contact = self.scene["contact"]
        self.feet, _ = self.contact.find_sensors(FEET, preserve_order=True)
        self.base, _ = self.contact.find_sensors("base")

        n = self.num_envs
        self.actions = torch.zeros(n, 12, device=self.device)
        self.prev_actions = torch.zeros(n, 12, device=self.device)
        self.cmd = torch.zeros(n, 3, device=self.device)
        self.clock = torch.zeros(n, device=self.device)  # gait cycle phase in [0, 1)
        gait = GAITS.get(cfg.gait)
        if cfg.gait != "free" and gait is None:
            raise ValueError(f"unknown gait {cfg.gait!r}, pick one of: free, {', '.join(GAITS)}")
        self.gait = gait
        if gait:
            self.offsets = torch.tensor(gait["offsets"], device=self.device)
        self.episode_sums: dict[str, torch.Tensor] = {}

    def foot_phase(self) -> torch.Tensor:
        return (self.clock.unsqueeze(1) + self.offsets) % 1.0  # (n, 4)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        self.targets = self.robot.data.default_joint_pos.torch + self.cfg.action_scale * self.actions

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.targets)

    def _get_observations(self) -> dict:
        self.prev_actions = self.actions.clone()
        if self.gait:
            self.clock = (self.clock + self.gait["freq"] * self.step_dt) % 1.0
            angle = 2 * math.pi * self.foot_phase()
            clock_obs = torch.cat([angle.sin(), angle.cos()], dim=1)
        else:
            clock_obs = torch.zeros(self.num_envs, 8, device=self.device)
        d = self.robot.data
        obs = torch.cat(
            [
                d.root_lin_vel_b.torch,
                d.root_ang_vel_b.torch,
                d.projected_gravity_b.torch,
                self.cmd,
                d.joint_pos.torch - d.default_joint_pos.torch,
                d.joint_vel.torch,
                self.actions,
                clock_obs,
            ],
            dim=1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        lin_vel, ang_vel = d.root_lin_vel_b.torch, d.root_ang_vel_b.torch
        moving = self.cmd[:, :2].norm(dim=1) > 0.1

        first_contact = self.contact.compute_first_contact(self.step_dt).torch[:, self.feet]
        last_air_time = self.contact.data.last_air_time.torch[:, self.feet]
        air_time = ((last_air_time - 0.5) * first_contact).sum(dim=1) * moving

        terms = {
            "alive": (~self.reset_terminated).float(),
            "track_lin_vel": torch.exp(-((self.cmd[:, :2] - lin_vel[:, :2]) ** 2).sum(dim=1) / 0.25),
            "track_yaw": torch.exp(-((self.cmd[:, 2] - ang_vel[:, 2]) ** 2) / 0.25),
            "lin_vel_z": lin_vel[:, 2] ** 2,
            "ang_vel_xy": (ang_vel[:, :2] ** 2).sum(dim=1),
            "torque": (self.robot.actuators.applied_effort.torch ** 2).sum(dim=1),
            "joint_acc": (d.joint_acc.torch ** 2).sum(dim=1),
            "action_rate": ((self.actions - self.prev_actions) ** 2).sum(dim=1),
            "air_time": air_time,
            "orientation": (d.projected_gravity_b.torch[:, :2] ** 2).sum(dim=1),
            "base_height": (d.root_pos_w.torch[:, 2] - self.cfg.base_height_target) ** 2,
        }
        if self.gait:
            in_contact = self.contact.data.net_normal_forces_w.torch[:, self.feet].norm(dim=-1) > 1.0
            want_contact = self.foot_phase() < self.gait["duty"]
            terms["gait"] = (in_contact == want_contact).float().mean(dim=1)

        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            r = getattr(self.cfg, f"rew_{name}") * value * self.step_dt
            reward += r
            self.episode_sums.setdefault(name, torch.zeros_like(r))
            self.episode_sums[name] += r
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_force = self.contact.data.net_normal_forces_w_history.torch[:, :, self.base].norm(dim=-1)
        fell = (base_force.max(dim=1).values > 1.0).any(dim=1)
        return fell, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:  # spread resets so all envs don't time out together
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, int(self.max_episode_length))

        # log each reward term's per-second average over the finished episodes (shows up in TensorBoard)
        self.extras["log"] = {
            f"Episode_Reward/{k}": v[env_ids].mean() / self.max_episode_length_s for k, v in self.episode_sums.items()
        }
        self.extras["log"]["Episode_Termination/fell"] = self.reset_terminated[env_ids].float().mean()
        for v in self.episode_sums.values():
            index_fill_(v, env_ids, 0.0)

        index_fill_(self.actions, env_ids, 0.0)
        index_fill_(self.prev_actions, env_ids, 0.0)
        k = len(env_ids)
        for i, (lo, hi) in enumerate((self.cfg.cmd_vx, self.cfg.cmd_vy, self.cfg.cmd_yaw)):
            self.cmd[env_ids, i] = torch.empty(k, device=self.device).uniform_(lo, hi)
        self.clock[env_ids] = torch.rand(k, device=self.device)

        d = self.robot.data
        root_pose = d.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=d.default_root_vel.torch[env_ids], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=d.default_joint_pos.torch[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=d.default_joint_vel.torch[env_ids], env_ids=env_ids)
