"""Go2 backflip env.

Task: stand, jump, rotate backwards once (nose up and over, 360°) and land on four feet, all within one short
episode. Every episode follows the same timeline, so the policy sees the time as a phase in [0, 1]:

    0 ─── flip_start ════ flip_end ───────── episode_length_s
      stand / crouch    jump + rotate     land, stand still

The main signal is `track_flip`: the flip angle (total backward rotation since reset, unwrapped, so it reaches
2π after one full flip) should follow a target that ramps 0 → 2π across the flip window. `spin` and `height`
give extra push during the window, `feet_down` and `pose` reward a clean stance before and after.

Import this module only after Isaac Sim has launched.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from playground.quadruped.go2 import Go2BaseEnv, Go2BaseEnvCfg


@configclass
class BackflipEnvCfg(Go2BaseEnvCfg):
    episode_length_s = 2.0
    observation_space = 47  # 45 proprioception + phase + flip progress
    action_scale = 0.5  # rad per unit action; the walk's 0.25 is too little leg travel to jump high
    fail_bodies = ["base", "Head_.*"]  # landing on the back or head ends the attempt

    # timeline [s]
    flip_start = 0.5
    flip_end = 1.0
    base_height_target = 0.30  # m, standing height

    # reward weights (each term is multiplied by step_dt)
    rew_alive = 1.0
    rew_track_flip = 5.0  # exp(-(flip angle - target)² / 1.0); the main term
    rew_spin = 0.5  # backward pitch rate [rad/s], capped at max_spin, flip window only
    rew_height = 10.0  # base height above standing [m], flip window only
    rew_feet_down = 1.0  # fraction of feet on the ground, outside the flip window
    rew_pose = -0.5  # joint distance from the default stance, after landing
    rew_ang_vel_roll_yaw = -0.1  # roll / yaw rates (a flip is pitch only)
    rew_torque = -1e-5
    rew_joint_acc = -2.5e-7
    rew_action_rate = -0.01

    max_spin = 12.0  # rad/s, cap for `spin` (a 0.5 s flip averages 2π / 0.5 ≈ 12.6)
    success_angle = 1.75 * math.pi  # rad; counts as a flip in the logs and in play.py


class BackflipEnv(Go2BaseEnv):
    cfg: BackflipEnvCfg

    def __init__(self, cfg: BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        self.flip_angle = torch.zeros(n, device=self.device)  # unwrapped backward rotation since reset [rad]
        self.prev_pitch = torch.zeros(n, device=self.device)  # wrapped pitch at the last step [rad]
        self.last_flip_angle = torch.zeros(n, device=self.device)  # flip angle of the previous episode (for play)
        self.last_fell = torch.zeros(n, dtype=torch.bool, device=self.device)

    def time_s(self) -> torch.Tensor:
        return self.episode_length_buf * self.step_dt

    def in_flip_window(self) -> torch.Tensor:
        t = self.time_s()
        return (t >= self.cfg.flip_start) & (t < self.cfg.flip_end)

    def target_flip_angle(self) -> torch.Tensor:
        ramp = (self.time_s() - self.cfg.flip_start) / (self.cfg.flip_end - self.cfg.flip_start)
        return 2 * math.pi * ramp.clamp(0.0, 1.0)

    def pitch(self) -> torch.Tensor:
        """Nose-up angle from the gravity direction in the body frame, wrapped to (-π, π]."""
        g = self.robot.data.projected_gravity_b.torch
        return torch.atan2(-g[:, 0], -g[:, 2])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # first callback after each physics step, so the flip angle is integrated here
        pitch = self.pitch()
        self.flip_angle += torch.remainder(pitch - self.prev_pitch + math.pi, 2 * math.pi) - math.pi
        self.prev_pitch = pitch

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.fell(), time_out

    def _get_observations(self) -> dict:
        self.prev_actions = self.actions.clone()
        d = self.robot.data
        phase = (self.time_s() / self.cfg.episode_length_s).unsqueeze(1)
        progress = (self.flip_angle / (2 * math.pi)).unsqueeze(1)
        obs = torch.cat(
            [
                d.root_lin_vel_b.torch,
                d.root_ang_vel_b.torch,
                d.projected_gravity_b.torch,
                d.joint_pos.torch - d.default_joint_pos.torch,
                d.joint_vel.torch,
                self.actions,
                phase,
                progress,
            ],
            dim=1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        ang_vel = d.root_ang_vel_b.torch
        window = self.in_flip_window().float()
        landed = (self.time_s() >= self.cfg.flip_end).float()
        height = d.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]

        terms = {
            "alive": (~self.reset_terminated).float(),
            "track_flip": torch.exp(-((self.flip_angle - self.target_flip_angle()) ** 2) / 1.0),
            "spin": (-ang_vel[:, 1]).clamp(max=self.cfg.max_spin) * window,  # nose up = negative body-y rate
            "height": (height - self.cfg.base_height_target).clamp(min=0.0) * window,
            "feet_down": self.feet_in_contact().float().mean(dim=1) * (1.0 - window),
            "pose": ((d.joint_pos.torch - d.default_joint_pos.torch) ** 2).sum(dim=1) * landed,
            "ang_vel_roll_yaw": ang_vel[:, 0] ** 2 + ang_vel[:, 2] ** 2,
            "torque": (self.robot.actuators.applied_effort.torch ** 2).sum(dim=1),
            "joint_acc": (d.joint_acc.torch ** 2).sum(dim=1),
            "action_rate": ((self.actions - self.prev_actions) ** 2).sum(dim=1),
        }
        return self.sum_rewards(terms)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # remember how the finished episodes went, before the base resets the robot (the very first reset has no
        # finished episodes: their length is still 0)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        done = env_ids[self.episode_length_buf[env_ids] > 0]
        flip_log = {}
        if len(done):
            self.last_flip_angle[done] = self.flip_angle[done]
            self.last_fell[done] = self.reset_terminated[done]
            flipped = (self.flip_angle[done] > self.cfg.success_angle) & ~self.reset_terminated[done]
            flip_log = {
                "Episode_Flip/rotations": (self.flip_angle[done] / (2 * math.pi)).mean(),
                "Episode_Flip/success": flipped.float().mean(),
            }

        # no reset spreading: every episode must start at t = 0 for the timeline to mean anything
        env_ids = super()._reset_idx(env_ids)
        self.extras["log"].update(flip_log)
        self.flip_angle[env_ids] = 0.0
        self.prev_pitch[env_ids] = 0.0
