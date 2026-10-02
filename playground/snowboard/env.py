"""Snowboard playground env: the Stage 1 rider (envs/carve_env.py) learns to side slip, skid and carve on a fenced
piste (envs/mountain.py).

Turns follow an S line laid on the snow, so turn size is a distance, not a time:

    y_ref(x) = Y · sin(2π x / λ + φ)        λ = turn length (metres per full S: one toe + one heel turn)
                                            Y = λ · tan(A) / 2π, so the line crosses the fall line at ±A

Every task is then two targets, recomputed each step from where the board is:

    path heading  ψ_ref    direction the board should travel: the line's direction, plus a pull back onto it
                           (pure pursuit over `lookahead` metres)
    slip angle    β*       angle between the board's long axis and its travel direction

    task       line          β*                            track left on the snow
    sideslip   straight      -90° (board across, heelside)  a wide straight band
    skid       S, λ          slip_max · |sin(2πx/λ + φ)|    a wide S (slip peaks at each turn's apex)
    carve      S, λ          0                              a thin S line

The turn's apex (path pointing down the fall line, line at ±Y) is where the S curves hardest, so that's where the
skid slides most; edges change while crossing the piste. Short / medium / long turns are λ ≈ 20 / 40 / 80 m:
set `turn_length` (sampled per episode; the policy observes λ). A sine with λ²/(4π²Y) below the board's sidecut
radius (~5–7 m) can't be carved, only skidded: short turns are a skid exercise.

The piste is fenced at ±fence_half_width (envs/mountain.py): crossing it ends the episode like a fall, and the
policy observes its position between the fences. Lines are centred on the piste, so a start phase φ ≠ 0 starts
off-centre (on the line).

The policy steers only through its body: tipping the board on edge makes the snow model turn it (sidecut), a flat
or over-pivoted board skids (grip limit). See envs/snow.py.

Heelside side slip: the rider faces downhill, the board's heel edge (+y) points uphill, and the board slides
toward its toe edge, so v_lat < 0 and β = atan2(v_lat, v_long) = -90°. The board's yaw is +90° (nose to +y).

Import this module only after Isaac Sim has launched.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass, index_fill_
from isaaclab.utils.math import quat_apply

from envs.carve_env import CarveEnv, CarveEnvCfg
from envs.mountain import FENCE_HALF_WIDTH

TASKS = ("sideslip", "skid", "carve")


@configclass
class SnowboardEnvCfg(CarveEnvCfg):
    episode_length_s = 12.0
    observation_space = 87 + 28 + 14  # carve env obs + last action + task (see _task_obs)
    action_scale = 0.25  # fraction of each joint's half-range per unit action; Stage 1's 0.5 flails

    task: str = "sideslip"  # sideslip | skid | carve
    mountain = True  # tilted snowy mountain with real gravity (False: Stage 1's flat ground + tilted gravity)
    # slope_deg (CarveEnvCfg): None = config.yaml slope (15°)

    # the S line
    turn_length = (40.0, 40.0)  # λ, m per full S, sampled per episode. Short ≈ 20, medium 40, long 80
    heading_amp_deg = 35.0  # A: the line crosses the fall line at this angle (sets the swing Y)
    lookahead = 6.0  # m, pure-pursuit distance pulling the heading target back onto the line
    slip_max_deg = 25.0  # skid: slip angle at each turn's apex
    fixed_phase: float | None = None  # rad; None = random start phase per episode (training). play.py fixes it
    speed_cmd = (0.5, 2.5)  # m/s, sampled per episode. sideslip: slow; the turn tasks override it in __post_init__
    fence_half_width = FENCE_HALF_WIDTH  # m; crossing it ends the episode
    run_length: float | None = None  # m; end the episode (as a time-out) after this far down the slope. play.py sets it

    # reward weights (each term is multiplied by step_dt)
    rew_alive = 1.0
    rew_upright = 1.0  # torso up · snow normal
    rew_heading = 2.0  # exp(-(ψ - ψ_ref)² / σ²)
    rew_line = 2.0  # exp(-(y - y_ref)² / σ_line²): be on the line, not just parallel to it
    rew_slip = 2.0  # exp(-(β - β*)² / σ²)
    rew_speed = 1.0  # exp(-(|v| - v_cmd)² / σ_v²)
    rew_pose = -0.1  # joint distance from the snowboard stance
    rew_action_rate = -0.01
    rew_torso_ang_vel = -0.05  # wobbling
    sigma_deg = 30.0  # tolerance of the heading and slip terms. 15° is too tight for turns: going straight
                      # (≈ 25° off a ±35° sine) then earns ~nothing, so nothing pulls the policy toward edging
    sigma_line = 2.0  # m
    sigma_speed = 2.0  # m/s, tolerance of the speed term

    def __post_init__(self):
        super().__post_init__()
        if self.task != "sideslip":
            self.speed_cmd = (4.0, 7.0)


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
        self.v_cmd = torch.zeros(n, device=dev)
        self.prev_actions = torch.zeros(n, cfg.action_space, device=dev)
        self.out_of_bounds = torch.zeros(n, dtype=torch.bool, device=dev)
        self.episode_sums: dict[str, torch.Tensor] = {}
        super().__init__(cfg, render_mode, **kwargs)
        self.sigma = math.radians(cfg.sigma_deg)

    # -- targets and measurements ---------------------------------------------------------------------------

    def position(self) -> torch.Tensor:
        """Board (x, y) on the slope relative to the env's start line [m]: x downhill, y across (0 = piste centre)."""
        return self.body_pos(self.board_id)[:, :2] - self.scene.env_origins[:, :2]

    def line_phase(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * math.pi * x / self.wavelength + self.phase

    def line(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(y_ref, direction) of the S line at distance x down the slope."""
        th = self.line_phase(x)
        k = 2 * math.pi / self.wavelength
        return self.swing * th.sin(), torch.atan(self.swing * k * th.cos())

    def targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(ψ_ref, β*, line error y - y_ref) [rad, rad, m] for every env."""
        x, y = self.position().unbind(-1)
        y_ref, direction = self.line(x)
        e_line = y - y_ref
        heading = direction - torch.atan(e_line / self.cfg.lookahead)
        if self.cfg.task == "sideslip":
            return heading, torch.full_like(x, -math.pi / 2), e_line
        slip_max = math.radians(self.cfg.slip_max_deg) if self.cfg.task == "skid" else 0.0
        return heading, slip_max * self.line_phase(x).sin().abs(), e_line

    def path_heading(self) -> torch.Tensor:
        """Direction the board travels [rad], 0 = straight down the fall line."""
        v = self.body_lin_vel(self.board_id)
        return torch.atan2(v[:, 1], v[:, 0])

    def slip_angle(self) -> torch.Tensor:
        """β = atan2(v_lat, v_long) [rad]: 0 = nose-first, no slip; -90° = sliding toward the toe edge."""
        info = self.snow_info or self._zero_snow_info()
        return torch.atan2(info["v_lat"], info["v_long"])

    def errors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(heading error [rad], slip error [rad], line error [m]). Turns score |β|: either way of skidding counts."""
        heading_ref, slip_ref, e_line = self.targets()
        e_heading = _wrap(self.path_heading() - heading_ref)
        beta = self.slip_angle()
        e_slip = _wrap(beta - slip_ref) if self.cfg.task == "sideslip" else beta.abs() - slip_ref
        return e_heading, e_slip, e_line

    # -- MDP ------------------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self.prev_actions = self.actions.clone()
        obs = super()._get_observations()["policy"]
        return {"policy": torch.cat([obs, self.actions, self._task_obs()], dim=-1)}

    def _task_obs(self) -> torch.Tensor:
        x, y = self.position().unbind(-1)
        th = self.line_phase(x)
        heading_ref, slip_ref, _ = self.targets()
        e_heading, e_slip, e_line = self.errors()
        speed = self.body_lin_vel(self.board_id).norm(dim=-1)
        return torch.stack(
            [th.sin(), th.cos(), heading_ref, slip_ref, e_heading, e_slip, speed, self.v_cmd, *self._board_yaw_sc(),
             e_line / 5.0, y / self.cfg.fence_half_width, self.wavelength / 40.0, self.swing / 5.0],
            dim=-1,
        )

    def _board_yaw_sc(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.body_quat(self.board_id)
        ex = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3))
        yaw = torch.atan2(ex[:, 1], ex[:, 0])
        return yaw.sin(), yaw.cos()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        fallen, time_out = super()._get_dones()
        x, y = self.position().unbind(-1)
        self.out_of_bounds = y.abs() > self.cfg.fence_half_width
        if self.cfg.run_length is not None:
            time_out = time_out | (x > self.cfg.run_length)
        return fallen | self.out_of_bounds, time_out

    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        torso_up = quat_apply(self.body_quat(self.torso_id), self.up.expand(self.num_envs, 3))
        e_heading, e_slip, e_line = self.errors()
        speed = self.body_lin_vel(self.board_id).norm(dim=-1)
        terms = {
            "alive": (~self.reset_terminated).float(),
            "upright": (torso_up * self.up).sum(-1).clamp(min=0.0),
            "heading": torch.exp(-(e_heading / self.sigma).square()),
            "line": torch.exp(-(e_line / self.cfg.sigma_line).square()),
            "slip": torch.exp(-(e_slip / self.sigma).square()),
            "speed": torch.exp(-((speed - self.v_cmd) / self.cfg.sigma_speed).square()),
            "pose": (d.joint_pos.torch - self.stance_pos).square().sum(-1),
            "action_rate": (self.actions - self.prev_actions).square().sum(-1),
            "torso_ang_vel": self.body_ang_vel(self.torso_id).square().sum(-1),
        }
        # for the logs: how well each target is met, averaged per episode like the reward terms
        self._track("err_heading_deg", e_heading.abs().rad2deg())
        self._track("err_slip_deg", e_slip.abs().rad2deg())
        self._track("err_line_m", e_line.abs())
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
        k = len(env_ids)
        if self.cfg.fixed_phase is None:
            self.phase[env_ids] = 2 * math.pi * torch.rand(k, device=self.device)
        else:
            self.phase[env_ids] = self.cfg.fixed_phase
        lo, hi = self.cfg.turn_length
        self.wavelength[env_ids] = lo + (hi - lo) * torch.rand(k, device=self.device)
        amp = 0.0 if self.cfg.task == "sideslip" else math.tan(math.radians(self.cfg.heading_amp_deg))
        self.swing[env_ids] = self.wavelength[env_ids] * amp / (2 * math.pi)
        lo, hi = self.cfg.speed_cmd
        self.v_cmd[env_ids] = lo + (hi - lo) * torch.rand(k, device=self.device)
        super()._reset_idx(env_ids)
        index_fill_(self.actions, env_ids, 0.0)
        index_fill_(self.prev_actions, env_ids, 0.0)

    def _log_episodes(self, env_ids: torch.Tensor):
        """Per-episode averages to TensorBoard: rewards per second, errors and speed per step."""
        if not self.episode_sums:
            return
        steps = self.episode_length_buf[env_ids].clamp_min(1).float()
        log = {}
        for k, v in self.episode_sums.items():
            if k.startswith("rew_"):
                log[f"Episode_Reward/{k[4:]}"] = v[env_ids].mean() / self.max_episode_length_s
            else:
                log[f"Episode_Snow/{k}"] = (v[env_ids] / steps).mean()
            index_fill_(v, env_ids, 0.0)
        log["Episode_Termination/fell"] = (self.reset_terminated[env_ids] & ~self.out_of_bounds[env_ids]).float().mean()
        log["Episode_Termination/fence"] = self.out_of_bounds[env_ids].float().mean()
        log["Episode_Snow/seconds"] = (steps * self.step_dt).mean()
        self.extras["log"] = log

    def _start_state(self, env_ids: torch.Tensor):
        """Start already doing the task: sideways at rest on the heel edge, or on the S line at the commanded speed.

        The side slip starts on its heel edge, with the rider's centre of mass right above that edge: tipped by the
        slope angle (board level) plus atan(half width / pelvis height). Bound rider + board then tip as one
        inverted pendulum on a 0.25 m base; from flat it rolls downhill onto the toe edge and falls in ~0.7 s.
        """
        n = len(env_ids)
        jitter = math.radians(5.0) * (2 * torch.rand(n, device=self.device) - 1)
        if self.cfg.task == "sideslip":
            yaw = math.pi / 2 + jitter
            lean = math.atan2(self.board_half_width, self.pelvis_in_board_pos[2].item())
            edge = torch.full_like(yaw, -(math.radians(self.slope_deg) + lean))
            # from rest: with any downhill speed the biting edge stops the board in one step and the body trips over it
            zeros = torch.zeros(n, device=self.device)
            return yaw, edge, torch.zeros(n, 3, device=self.device), zeros
        # on the line at x = 0
        th = self.phase[env_ids]
        k = 2 * math.pi / self.wavelength[env_ids]
        y0 = self.swing[env_ids] * th.sin()
        heading = torch.atan(self.swing[env_ids] * k * th.cos())
        speed = self.v_cmd[env_ids].unsqueeze(-1)
        vel = speed * torch.stack([heading.cos(), heading.sin(), torch.zeros_like(heading)], dim=-1)
        return heading + jitter, torch.zeros_like(heading), vel, y0


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(a.sin(), a.cos())
