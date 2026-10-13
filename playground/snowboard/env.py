"""Snowboard env: the rider (envs/carve_env.py) learns to side slip, skid or carve down a fenced piste.

The rider follows an S line laid on the snow, y_ref(x) = Y·sin(2πx/λ + φ), with λ the turn length in metres.
Each step the env turns "where the board is" into two targets:

    path heading  the direction to travel: the line's direction, pulled back onto the line (pure pursuit)
    slip angle    angle between the board's long axis and its travel direction

    task      line       slip target
    sideslip  straight   -90° (sliding sideways on the heel edge)
    skid      S          slip_max · |sin(2πx/λ + φ)|
    carve     S          0

The policy only moves its body; the snow model (envs/snow.py) turns the board. Crossing a fence ends the episode.
See README.md for the observation, reward and knobs.

Import this module only after Isaac Sim has launched.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from isaaclab.utils import configclass, index_fill_
from isaaclab.utils.math import quat_apply

from envs.carve_env import CarveEnv, CarveEnvCfg
from envs.mountain import FENCE_HALF_WIDTH

TASKS = ("sideslip", "skid", "carve")


@configclass
class SnowboardEnvCfg(CarveEnvCfg):
    episode_length_s = 12.0
    observation_space = 87 + 28 + 14  # carve env obs + last action + task obs (see _task_obs)
    action_scale = 0.25  # fraction of each joint's half-range per unit action; the carve env's 0.5 flails

    task: str = "sideslip"  # sideslip | skid | carve
    mountain = True

    # the S line
    turn_length = (40.0, 40.0)  # λ, metres per full S, sampled per episode (short ≈ 20, medium 40, long 80)
    heading_amp_deg = 35.0  # the line crosses the fall line at this angle (sets the swing Y)
    lookahead = 6.0  # m, pure-pursuit distance pulling the heading target back onto the line
    slip_max_deg = 25.0  # skid: slip angle at each turn's apex
    fixed_phase: float | None = None  # rad; None = random start phase per episode (play.py fixes it)
    speed_cmd = (0.5, 2.5)  # m/s, sampled per episode (turn tasks: see __post_init__)
    fence_half_width = FENCE_HALF_WIDTH  # m; crossing it ends the episode
    run_length: float | None = None  # m; end the episode (as a time-out) this far down (play.py sets it)

    # reward weights (each term is multiplied by step_dt)
    rew_alive = 1.0
    rew_upright = 1.0  # torso up · snow normal
    rew_heading = 2.0  # exp(-(heading error / σ)²)
    rew_line = 2.0  # exp(-(line error / σ_line)²): be on the line, not just parallel to it
    rew_slip = 2.0  # exp(-(slip error / σ)²)
    rew_speed = 1.0  # exp(-((|v| - v_cmd) / σ_v)²)
    rew_pose = -0.1  # joint distance from the snowboard stance
    rew_action_rate = -0.01
    rew_torso_ang_vel = -0.05  # wobbling
    sigma_deg = 30.0  # tolerance of the heading and slip terms; tighter and going straight earns ~nothing on a turn
    sigma_line = 2.0  # m
    sigma_speed = 2.0  # m/s

    def __post_init__(self):
        super().__post_init__()
        if self.task != "sideslip":
            self.speed_cmd = (4.0, 7.0)


class Tracking(NamedTuple):
    """How the board is doing against the line, per env. Angles in rad, line error in m."""

    heading_ref: torch.Tensor  # direction the board should travel
    slip_ref: torch.Tensor  # slip angle the task asks for
    e_heading: torch.Tensor  # travel direction - heading_ref
    e_slip: torch.Tensor  # slip angle - slip_ref (turns: |slip| - slip_ref, either way of skidding counts)
    e_line: torch.Tensor  # y - y_ref


class SnowboardEnv(CarveEnv):
    cfg: SnowboardEnvCfg

    def __init__(self, cfg: SnowboardEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.task not in TASKS:
            raise ValueError(f"unknown task {cfg.task!r}, pick one of: {', '.join(TASKS)}")
        n, dev = cfg.scene.num_envs, cfg.sim.device
        # task state must exist before super().__init__, which resets every env
        self.phase = torch.zeros(n, device=dev)  # φ [rad]
        self.wavelength = torch.full((n,), 40.0, device=dev)  # λ [m]
        self.swing = torch.zeros(n, device=dev)  # Y [m]
        self.v_cmd = torch.zeros(n, device=dev)  # commanded speed [m/s]
        self.prev_actions = torch.zeros(n, cfg.action_space, device=dev)
        self.out_of_bounds = torch.zeros(n, dtype=torch.bool, device=dev)
        self.episode_sums: dict[str, torch.Tensor] = {}  # per-episode totals for TensorBoard
        super().__init__(cfg, render_mode, **kwargs)
        self.sigma = math.radians(cfg.sigma_deg)

    # -- the line and how the board tracks it -----------------------------------------------------------------

    def position(self) -> torch.Tensor:
        """Board (x, y) relative to the env's start [m]: x downhill, y across (0 = piste centre)."""
        return self.body_pos(self.board_id)[:, :2] - self.scene.env_origins[:, :2]

    def line_phase(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * math.pi * x / self.wavelength + self.phase

    def line(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(y_ref, direction) of the S line at distance x down the slope."""
        phase = self.line_phase(x)
        k = 2 * math.pi / self.wavelength
        return self.swing * phase.sin(), torch.atan(self.swing * k * phase.cos())

    def board_speed(self) -> torch.Tensor:
        return self.body_lin_vel(self.board_id).norm(dim=-1)

    def path_heading(self) -> torch.Tensor:
        """Direction the board travels [rad], 0 = straight down the fall line."""
        v = self.body_lin_vel(self.board_id)
        return torch.atan2(v[:, 1], v[:, 0])

    def slip_angle(self) -> torch.Tensor:
        """atan2(v_lat, v_long) [rad]: 0 = nose-first, no slip; -90° = sliding toward the toe edge."""
        info = self.snow_info or self._zero_snow_info()
        return torch.atan2(info["v_lat"], info["v_long"])

    def board_yaw(self) -> torch.Tensor:
        """Direction the nose points [rad], 0 = down the fall line."""
        forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        nose = quat_apply(self.body_quat(self.board_id), forward)
        return torch.atan2(nose[:, 1], nose[:, 0])

    def tracking(self) -> Tracking:
        x, y = self.position().unbind(-1)
        y_ref, direction = self.line(x)
        e_line = y - y_ref
        heading_ref = direction - torch.atan(e_line / self.cfg.lookahead)

        task, beta = self.cfg.task, self.slip_angle()
        if task == "sideslip":
            slip_ref = torch.full_like(x, -math.pi / 2)
            e_slip = _wrap(beta - slip_ref)
        else:
            slip_max = math.radians(self.cfg.slip_max_deg) if task == "skid" else 0.0
            slip_ref = slip_max * self.line_phase(x).sin().abs()
            e_slip = beta.abs() - slip_ref
        return Tracking(heading_ref, slip_ref, _wrap(self.path_heading() - heading_ref), e_slip, e_line)

    # -- MDP --------------------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self.prev_actions = self.actions.clone()
        obs = super()._get_observations()["policy"]
        return {"policy": torch.cat([obs, self.actions, self._task_obs()], dim=-1)}

    def _task_obs(self) -> torch.Tensor:
        x, y = self.position().unbind(-1)
        phase, yaw, t = self.line_phase(x), self.board_yaw(), self.tracking()
        return torch.stack(
            [phase.sin(), phase.cos(), t.heading_ref, t.slip_ref, t.e_heading, t.e_slip, self.board_speed(),
             self.v_cmd, yaw.sin(), yaw.cos(), t.e_line / 5.0, y / self.cfg.fence_half_width,
             self.wavelength / 40.0, self.swing / 5.0],
            dim=-1,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        fallen, time_out = super()._get_dones()
        x, y = self.position().unbind(-1)
        self.out_of_bounds = y.abs() > self.cfg.fence_half_width
        if self.cfg.run_length is not None:
            time_out = time_out | (x > self.cfg.run_length)
        return fallen | self.out_of_bounds, time_out

    def _get_rewards(self) -> torch.Tensor:
        torso_up = quat_apply(self.body_quat(self.torso_id), self.up.expand(self.num_envs, 3))
        t, speed = self.tracking(), self.board_speed()
        terms = {
            "alive": (~self.reset_terminated).float(),
            "upright": (torso_up * self.up).sum(-1).clamp(min=0.0),
            "heading": torch.exp(-(t.e_heading / self.sigma).square()),
            "line": torch.exp(-(t.e_line / self.cfg.sigma_line).square()),
            "slip": torch.exp(-(t.e_slip / self.sigma).square()),
            "speed": torch.exp(-((speed - self.v_cmd) / self.cfg.sigma_speed).square()),
            "pose": (self.robot.data.joint_pos.torch - self.stance_pos).square().sum(-1),
            "action_rate": (self.actions - self.prev_actions).square().sum(-1),
            "torso_ang_vel": self.body_ang_vel(self.torso_id).square().sum(-1),
        }
        # for TensorBoard: how well each target is met, averaged per episode like the reward terms
        self._track("err_heading_deg", t.e_heading.abs().rad2deg())
        self._track("err_slip_deg", t.e_slip.abs().rad2deg())
        self._track("err_line_m", t.e_line.abs())
        self._track("speed", speed)
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            r = getattr(self.cfg, f"rew_{name}") * value * self.step_dt
            reward += r
            self._track(f"rew_{name}", r)
        return reward

    def _track(self, name: str, value: torch.Tensor):
        self.episode_sums.setdefault(name, torch.zeros_like(value))
        self.episode_sums[name] += value

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._log_episodes(env_ids)

        def sample(lo_hi):
            lo, hi = lo_hi
            return lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)

        if self.cfg.fixed_phase is None:
            self.phase[env_ids] = 2 * math.pi * torch.rand(len(env_ids), device=self.device)
        else:
            self.phase[env_ids] = self.cfg.fixed_phase
        self.wavelength[env_ids] = sample(self.cfg.turn_length)
        slope_tan = 0.0 if self.cfg.task == "sideslip" else math.tan(math.radians(self.cfg.heading_amp_deg))
        self.swing[env_ids] = self.wavelength[env_ids] * slope_tan / (2 * math.pi)
        self.v_cmd[env_ids] = sample(self.cfg.speed_cmd)
        super()._reset_idx(env_ids)
        index_fill_(self.actions, env_ids, 0.0)
        index_fill_(self.prev_actions, env_ids, 0.0)

    def _log_episodes(self, env_ids: torch.Tensor):
        """Per-episode averages to TensorBoard: rewards per second, errors and speed per step."""
        if not self.episode_sums:
            return
        steps = self.episode_length_buf[env_ids].clamp_min(1).float()
        log = {}
        for name, total in self.episode_sums.items():
            if name.startswith("rew_"):
                log[f"Episode_Reward/{name[4:]}"] = total[env_ids].mean() / self.max_episode_length_s
            else:
                log[f"Episode_Snow/{name}"] = (total[env_ids] / steps).mean()
            index_fill_(total, env_ids, 0.0)
        log["Episode_Termination/fell"] = (self.reset_terminated[env_ids] & ~self.out_of_bounds[env_ids]).float().mean()
        log["Episode_Termination/fence"] = self.out_of_bounds[env_ids].float().mean()
        log["Episode_Snow/seconds"] = (steps * self.step_dt).mean()
        self.extras["log"] = log

    def _start_state(self, env_ids: torch.Tensor):
        """Start already doing the task: sideways at rest on the heel edge, or on the S line at the commanded speed."""
        n = len(env_ids)
        jitter = math.radians(5.0) * (2 * torch.rand(n, device=self.device) - 1)
        if self.cfg.task == "sideslip":
            # At rest, with the centre of mass right above the heel edge: slope angle plus the lean to the board's
            # edge. From flat the rider rolls onto the toe edge and falls in ~0.7 s; with any downhill speed the
            # biting edge stops the board in one step and the body trips over it.
            lean = math.atan2(self.board_half_width, self.pelvis_in_board_pos[2].item())
            yaw = math.pi / 2 + jitter
            edge = torch.full_like(yaw, -(math.radians(self.slope_deg) + lean))
            return yaw, edge, torch.zeros(n, 3, device=self.device), torch.zeros(n, device=self.device)
        # on the line at x = 0, heading along it
        phase, swing, wavelength = self.phase[env_ids], self.swing[env_ids], self.wavelength[env_ids]
        heading = torch.atan(swing * (2 * math.pi / wavelength) * phase.cos())
        direction = torch.stack([heading.cos(), heading.sin(), torch.zeros_like(heading)], dim=-1)
        velocity = self.v_cmd[env_ids].unsqueeze(-1) * direction
        return heading + jitter, torch.zeros_like(heading), velocity, swing * phase.sin()


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(angle.sin(), angle.cos())
