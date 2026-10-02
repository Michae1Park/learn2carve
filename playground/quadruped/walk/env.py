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

from isaaclab.utils import configclass

from playground.quadruped.go2 import Go2BaseEnv, Go2BaseEnvCfg

# Per-foot phase offsets (FL, FR, RL, RR) in cycles, stance fraction ("duty"), and cycle frequency [Hz].
# A foot is in stance while (clock + offset) mod 1 < duty.
GAITS = {
    "walk": dict(offsets=(0.25, 0.75, 0.0, 0.5), duty=0.75, freq=1.5),  # 4-beat: RL, FL, RR, FR; 3 feet down
    "trot": dict(offsets=(0.0, 0.5, 0.5, 0.0), duty=0.5, freq=2.0),  # diagonal pairs together
    "pace": dict(offsets=(0.0, 0.5, 0.0, 0.5), duty=0.5, freq=2.0),  # same-side pairs together
}


@configclass
class Go2EnvCfg(Go2BaseEnvCfg):
    episode_length_s = 20.0
    observation_space = 56  # 48 proprioception + 8 clock (sin, cos per foot)

    gait: str = "free"  # free | walk | trot | pace
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


class Go2Env(Go2BaseEnv):
    cfg: Go2EnvCfg

    def __init__(self, cfg: Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        self.cmd = torch.zeros(n, 3, device=self.device)
        self.clock = torch.zeros(n, device=self.device)  # gait cycle phase in [0, 1)
        gait = GAITS.get(cfg.gait)
        if cfg.gait != "free" and gait is None:
            raise ValueError(f"unknown gait {cfg.gait!r}, pick one of: free, {', '.join(GAITS)}")
        self.gait = gait
        if gait:
            self.offsets = torch.tensor(gait["offsets"], device=self.device)

    def foot_phase(self) -> torch.Tensor:
        return (self.clock.unsqueeze(1) + self.offsets) % 1.0  # (n, 4)

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
            want_contact = self.foot_phase() < self.gait["duty"]
            terms["gait"] = (self.feet_in_contact() == want_contact).float().mean(dim=1)
        return self.sum_rewards(terms)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.fell(), time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        env_ids = super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:  # spread resets so all envs don't time out together
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, int(self.max_episode_length))
        k = len(env_ids)
        for i, (lo, hi) in enumerate((self.cfg.cmd_vx, self.cfg.cmd_vy, self.cfg.cmd_yaw)):
            self.cmd[env_ids, i] = torch.empty(k, device=self.device).uniform_(lo, hi)
        self.clock[env_ids] = torch.rand(k, device=self.device)
