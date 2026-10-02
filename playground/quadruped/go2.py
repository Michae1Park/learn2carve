"""Go2 scene + env base shared by walk/env.py and backflip/env.py.

The base owns everything that isn't task-specific: the scene, joint-position actions, reward bookkeeping
(`rew_<name>` weights × dt, logged per term to TensorBoard) and resetting the robot to its default stance.
Subclasses add observations, reward terms and terminations.

Import this module only after Isaac Sim has launched.
"""

from __future__ import annotations

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
class Go2BaseEnvCfg(DirectRLEnvCfg):
    decimation = 4  # policy at 50 Hz, physics at 200 Hz
    action_space = 12
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # free camera starting with an overview of the env grid (centred on the origin); orbit/zoom in the client.
    # To follow one robot instead: ViewerCfg(eye=(1.5, 1.5, 0.8), origin_type="asset_root", asset_name="robot")
    viewer: ViewerCfg = ViewerCfg(eye=(30.0, 30.0, 20.0), lookat=(0.0, 0.0, 0.0))

    action_scale = 0.25  # rad per unit action
    action_clip = 5.0
    fail_bodies = ["base"]  # episode ends (fell) when any of these touches the ground


class Go2BaseEnv(DirectRLEnv):
    cfg: Go2BaseEnvCfg

    def __init__(self, cfg: Go2BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.contact = self.scene["contact"]
        self.feet, _ = self.contact.find_sensors(FEET, preserve_order=True)
        self.fail_bodies, _ = self.contact.find_sensors(cfg.fail_bodies)

        n = self.num_envs
        self.actions = torch.zeros(n, 12, device=self.device)
        self.prev_actions = torch.zeros(n, 12, device=self.device)
        self.episode_sums: dict[str, torch.Tensor] = {}

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        self.targets = self.robot.data.default_joint_pos.torch + self.cfg.action_scale * self.actions

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.targets)

    def feet_in_contact(self) -> torch.Tensor:
        return self.contact.data.net_normal_forces_w.torch[:, self.feet].norm(dim=-1) > 1.0  # (n, 4)

    def fell(self) -> torch.Tensor:
        force = self.contact.data.net_normal_forces_w_history.torch[:, :, self.fail_bodies].norm(dim=-1)
        return (force.max(dim=1).values > 1.0).any(dim=1)

    def sum_rewards(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted sum of the terms (weight = cfg.rew_<name>, × step_dt), also accumulated for logging."""
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            r = getattr(self.cfg, f"rew_{name}") * value * self.step_dt
            reward += r
            self.episode_sums.setdefault(name, torch.zeros_like(r))
            self.episode_sums[name] += r
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # log each reward term's per-second average over the finished episodes (shows up in TensorBoard)
        self.extras["log"] = {
            f"Episode_Reward/{k}": v[env_ids].mean() / self.max_episode_length_s for k, v in self.episode_sums.items()
        }
        self.extras["log"]["Episode_Termination/fell"] = self.reset_terminated[env_ids].float().mean()
        for v in self.episode_sums.values():
            index_fill_(v, env_ids, 0.0)

        index_fill_(self.actions, env_ids, 0.0)
        index_fill_(self.prev_actions, env_ids, 0.0)

        d = self.robot.data
        root_pose = d.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=d.default_root_vel.torch[env_ids], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=d.default_joint_pos.torch[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=d.default_joint_vel.torch[env_ids], env_ids=env_ids)
        return env_ids
