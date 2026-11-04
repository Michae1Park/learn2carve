"""Snowboard env: the rider (envs/carve_env.py) learns to side slip, skid or carve down a fenced piste.

No line is laid on the snow. The turn tasks command a speed and reward linked turns; on a slope, the only way to hold
a speed below the straight-run speed is to turn across the hill, so the S (its length, width and rhythm) comes out of
the physics: speed, slope, edge angle and grip. The tasks differ in how the board may slide:

    task      heading                         slip target β*                 track
    sideslip  straight down the fall line     -90° (sliding on the heel edge)  wide straight band
    skid      free: linked turns              20° (the board pivots and slides) wide S
    carve     free: linked turns              0 (clean edge)                   thin S

A turn counts when the board's travel direction crosses the fall line after swinging at least min_turn_deg to one
side. The policy only moves its body; the snow model (envs/snow.py) turns the board. Crossing a fence ends the episode.
See docs/RL_TRAINING.md for the observation, reward and knobs.

Import this module only after Isaac Sim has launched.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass, index_fill_
from isaaclab.utils.math import quat_apply

from envs.carve_env import CarveEnv, CarveEnvCfg, base_obs_size, num_dof
from envs.mountain import FENCE_HALF_WIDTH

TASKS = ("sideslip", "skid", "carve")
TASK_OBS = 16  # see _task_obs


@configclass
class SnowboardEnvCfg(CarveEnvCfg):
    episode_length_s = 12.0
    observation_space = 87 + 28 + TASK_OBS  # carve env obs + last action + task obs; recomputed from the DoF
    action_scale = 0.25  # fraction of each joint's half-range per unit action; the carve env's 0.5 flails

    task: str = "sideslip"  # sideslip | skid | carve
    mountain = True

    speed_cmd = (0.5, 2.5)  # m/s, sampled per episode (turn tasks: see __post_init__)
    # a turn counts once the path swung this far to one side and crossed back over the fall line. 15° was gamed:
    # ±15° wiggles every 0.7 s collected the bonus without slowing down; 45° is a real turn, across the hill
    min_turn_deg = 45.0
    skid_slip_deg = 20.0  # skid: slip angle to hold
    start_heading_deg = 20.0  # turn tasks: start heading sampled in ± this from the fall line
    fence_half_width = FENCE_HALF_WIDTH  # m; crossing it ends the episode

    # guide S line (turn tasks): something to trace while learning to turn. Adaptive: it fades only while the policy
    # follows it (mean error < guide_fade_below), and comes back if it loses it (> guide_restore_above). On a fixed
    # schedule it faded while the policy was still learning not to fall, and the turn was never learned. Each episode
    # draws its own S, so no one shape is learned; once the guide is gone, turn length and shape are the policy's.
    guide_steps: int = 3000  # steps of good tracking to fade the guide from full to 0; 0 = off (play, eval)
    guide_fade_below_deg = 15.0
    guide_restore_above_deg = 30.0
    # Each episode: crossing angle A, then a turn length λ that keeps the S carvable: tightest bend
    # λ / (2π tan A) ≥ guide_min_radius (sidecut 5–7 m; 2.8 m bends made riders throw themselves), swing
    # λ tan A / 2π ≤ guide_max_swing (fences at ±20 m).
    guide_amp_deg = (35.0, 50.0)  # angle at which the guide crosses the fall line
    guide_min_radius = 9.0  # m
    guide_max_swing = 15.0  # m
    guide_lookahead = 6.0  # m, pure pursuit back onto the guide
    run_length: float | None = None  # m; end the episode (as a time-out) this far down (play.py sets it)

    # Reward: what a good run achieves, not how to move. An earlier 17-term version (upright, low, facing,
    # angulation, pose, ...) hand-designed the technique, its terms fought each other and each one was gamed.
    # Each term is multiplied by step_dt, except turn and fall (once per event).
    rew_alive = 1.0
    rew_speed = 2.0  # exp(-((|v| - v_cmd) / σ_v)²): the reason to turn
    rew_speed_cap = -3.0  # × clip((|v| - speed_cap) / 2, 0, 1): over the cap is clearly worse than turning
    speed_cap = 12.0  # m/s; a clean carve on 15° settles at ~19-21 m/s, so the cap forces turns finished across the hill
    rew_carve = 2.0  # turn tasks: edged × turning × clean = clip(|θ|/edge_target) · clip(|ψ̇_board|/0.5) · exp(-(β/σ)²)
    edge_target_deg = 20.0  # full carve credit at or above this edge angle. Carve: 40 (see __post_init__)
    rew_slip = 2.0  # skid / side slip only (the carve term covers slip for carving): exp(-(slip error / σ_slip)²)
    rew_heading = 2.0  # side slip only: exp(-(heading / σ)²), straight down the fall line
    rew_turn = 1.0  # per completed turn (turn tasks): links the turns
    rew_fall = -10.0  # once, when the episode ends in a fall or past a fence
    rew_guide = 4.0  # exp(-(heading - guide heading)² / σ²) × fade (turn tasks, adaptive, see guide_fade)
    rew_action_rate = -0.01
    start_speed_turns = (4.0, 6.0)  # m/s: slow, so the first turn comes before the fence (20 m away)
    sigma_slip_deg = 15.0
    sigma_heading_deg = 30.0  # also the guide's tolerance
    sigma_speed = 5.0  # m/s; 2 gave no gradient: at 15 m/s vs 7 commanded, exp(-16) ≈ 0, so nothing taught it to turn

    def __post_init__(self):
        super().__post_init__()
        if self.task != "sideslip":
            # what a carved S on 15° can hold inside the fences: a clean carve settles at ~19-21 m/s on any S that
            # fits, so 5-9 m/s (the first try) was unreachable without skidding
            self.speed_cmd = (8.0, 11.0)  # below the cap
        if self.task == "carve":
            # carve aggressively: full reward from a 40° edge up, no penalty beyond (up to ~85° if it wants)
            self.edge_target_deg = 40.0


class SnowboardEnv(CarveEnv):
    cfg: SnowboardEnvCfg

    def __init__(self, cfg: SnowboardEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.task not in TASKS:
            raise ValueError(f"unknown task {cfg.task!r}, pick one of: {', '.join(TASKS)}")
        cfg.observation_space = base_obs_size(cfg) + num_dof(cfg) + TASK_OBS
        n, dev = cfg.scene.num_envs, cfg.sim.device
        # task state must exist before super().__init__, which resets every env
        self.v_cmd = torch.zeros(n, device=dev)  # commanded speed [m/s]
        self.turn_side = torch.zeros(n, device=dev)  # side the path has swung to: -1, 0 (not yet), +1
        self.turn_swing = torch.zeros(n, device=dev)  # largest |heading| on that side so far [rad]
        self.turn_count = torch.zeros(n, device=dev)
        self.last_turns = torch.zeros(n, device=dev)  # turns of each env's previous episode (read after a reset)
        self.since_turn = torch.zeros(n, device=dev)  # s since the last completed turn (or the start)
        self.turn_done = torch.zeros(n, dtype=torch.bool, device=dev)  # a turn completed this step
        self.guide_k = torch.zeros(n, device=dev)  # 2π / λ of this episode's guide [1/m]
        self.guide_y = torch.zeros(n, device=dev)  # its swing Y [m]
        self.guide_phase = torch.zeros(n, device=dev)
        self.guide_level = 1.0  # adaptive fade, see guide_fade()
        self.guide_err_ema = 90.0  # deg, running mean of the guide heading error over all envs
        self.prev_actions = torch.zeros(n, num_dof(cfg), device=dev)
        self.out_of_bounds = torch.zeros(n, dtype=torch.bool, device=dev)
        self.episode_sums: dict[str, torch.Tensor] = {}  # per-episode totals for TensorBoard
        super().__init__(cfg, render_mode, **kwargs)
        self.min_turn = math.radians(cfg.min_turn_deg)

    # -- where the board is and how it moves --------------------------------------------------------------------

    def position(self) -> torch.Tensor:
        """Board (x, y) relative to the env's start [m]: x downhill, y across (0 = piste centre)."""
        return self.body_pos(self.board_id)[:, :2] - self.scene.env_origins[:, :2]

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

    def guide_fade(self) -> float:
        """Guide strength 0..1: starts at 1, adapted by _adapt_guide (always 0 for side slip or guide_steps = 0)."""
        if self.cfg.task == "sideslip" or self.cfg.guide_steps <= 0:
            return 0.0
        return self.guide_level

    def _adapt_guide(self, e_guide_deg: torch.Tensor):
        """Fade the guide while it's followed, bring it back when it's lost (once per step, over all envs)."""
        if self.cfg.guide_steps <= 0:
            return
        alive = ~self.reset_terminated
        if alive.any():
            self.guide_err_ema = 0.99 * self.guide_err_ema + 0.01 * e_guide_deg[alive].mean().item()
        step = 1.0 / self.cfg.guide_steps
        if self.guide_err_ema < self.cfg.guide_fade_below_deg:
            self.guide_level = max(0.0, self.guide_level - step)
        elif self.guide_err_ema > self.cfg.guide_restore_above_deg:
            self.guide_level = min(1.0, self.guide_level + step)

    def guide_heading(self) -> torch.Tensor:
        """Heading toward the guide S: its direction at the board, pulled back onto it (pure pursuit)."""
        x, y = self.position().unbind(-1)
        arg = self.guide_k * x + self.guide_phase
        y_ref = self.guide_y * arg.sin()
        return torch.atan(self.guide_y * self.guide_k * arg.cos()) - torch.atan((y - y_ref) / self.cfg.guide_lookahead)

    def slip_target(self) -> torch.Tensor:
        target = {"sideslip": -math.pi / 2, "skid": math.radians(self.cfg.skid_slip_deg), "carve": 0.0}[self.cfg.task]
        return torch.full((self.num_envs,), target, device=self.device)

    def slip_error(self) -> torch.Tensor:
        """Sideslip: signed (heel edge only). Turns: |slip| - target, so skidding either way counts."""
        beta = self.slip_angle()
        if self.cfg.task == "sideslip":
            return _wrap(beta - self.slip_target())
        return beta.abs() - self.slip_target()

    def _update_turns(self):
        """Count linked turns: the path swings ≥ min_turn to one side, then crosses the fall line to the other."""
        psi = self.path_heading()
        side = torch.sign(psi) * (psi.abs() > 0.02)
        crossed = (side != 0) & (self.turn_side != 0) & (side != self.turn_side)
        self.turn_done = crossed & (self.turn_swing >= self.min_turn)
        self.turn_count += self.turn_done.float()
        self.since_turn = torch.where(self.turn_done, torch.zeros_like(self.since_turn), self.since_turn + self.step_dt)
        new_side = torch.where(side != 0, side, self.turn_side)
        self.turn_swing = torch.where(crossed, psi.abs(), torch.maximum(self.turn_swing, psi.abs()))
        self.turn_side = new_side

    # -- MDP --------------------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self.prev_actions = self.actions.clone()
        obs = super()._get_observations()["policy"]
        return {"policy": torch.cat([obs, self.actions, self._task_obs()], dim=-1)}

    def _task_obs(self) -> torch.Tensor:
        psi, beta, yaw = self.path_heading(), self.slip_angle(), self.board_yaw()
        y = self.position()[:, 1]
        fade = self.guide_fade()
        e_guide = _wrap(self.guide_heading() - psi)  # faded with the reward: zeros once the guide is gone
        return torch.stack(
            [self.board_speed(), self.v_cmd, psi.sin(), psi.cos(), beta.sin(), beta.cos(), yaw.sin(), yaw.cos(),
             y / self.cfg.fence_half_width, self.turn_side, self.turn_swing, self.since_turn.clamp(max=10.0) / 5.0,
             self.slip_target(), fade * e_guide.sin(), fade * (1 - e_guide.cos()), torch.full_like(psi, fade)],
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
        self._update_turns()
        cfg = self.cfg
        speed, e_slip, psi = self.board_speed(), self.slip_error(), self.path_heading()
        turns = cfg.task != "sideslip"
        zeros = torch.zeros_like(speed)
        edge = self.snow_info["edge_angle"].abs() if self.snow_info else zeros
        clean = torch.exp(-(self.slip_angle() / math.radians(cfg.sigma_slip_deg)).square())
        terms = {
            "alive": (~self.reset_terminated).float(),
            "speed": torch.exp(-((speed - self.v_cmd) / cfg.sigma_speed).square()),
            "speed_cap": ((speed - cfg.speed_cap) / 2.0).clamp(0.0, 1.0),
            # edged, turning and clean at once: tipped but straight (leaning back into the fence), or turning flat, or
            # skidding, all earn ~0
            "carve": (edge / math.radians(cfg.edge_target_deg)).clamp(0.0, 1.0) * self.turning() * clean
            if cfg.task == "carve" else zeros,
            "slip": zeros if cfg.task == "carve" else torch.exp(-(e_slip / math.radians(cfg.sigma_slip_deg)).square()),
            "heading": zeros if turns else torch.exp(-(psi / math.radians(cfg.sigma_heading_deg)).square()),
            "action_rate": (self.actions - self.prev_actions).square().sum(-1),
        }
        if turns:
            self._technique()  # logged only (pelvis height, angulation), not rewarded
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            r = getattr(cfg, f"rew_{name}") * value * self.step_dt
            reward += r
            self._track(f"rew_{name}", r)
        r = cfg.rew_fall * self.reset_terminated.float()  # a fall or a fence, not a time-out
        reward += r
        self._track("rew_fall", r)
        if turns:  # the guide, faded over training
            e_guide = _wrap(psi - self.guide_heading())
            x, y = self.position().unbind(-1)
            e_line = y - self.guide_y * (self.guide_k * x + self.guide_phase).sin()
            r = self.guide_fade() * cfg.rew_guide * torch.exp(-(e_guide / math.radians(cfg.sigma_heading_deg)).square())
            reward += r * self.step_dt
            self._track("rew_guide", r * self.step_dt)
            self._track("err_guide_m", e_line.abs())
            self._track("err_guide_deg", e_guide.abs().rad2deg())
            self._adapt_guide(e_guide.abs().rad2deg())
        if turns:  # a bonus per completed turn, not a rate
            r = cfg.rew_turn * self.turn_done.float()
            reward += r
            self._track("rew_turn", r)

        # for TensorBoard: per-episode averages
        self._track("err_slip_deg", e_slip.abs().rad2deg())
        self._track("err_speed", (speed - self.v_cmd).abs())
        self._track("speed", speed)
        self._track("heading_abs_deg", psi.abs().rad2deg())
        self._track("edge_abs_deg", self.snow_info["edge_angle"].abs().rad2deg() if self.snow_info
                    else torch.zeros_like(speed))
        return reward

    def turning(self) -> torch.Tensor:
        """0..1: how hard the board is turning, from its yaw rate about the snow normal (full at 0.5 rad/s)."""
        yaw_rate = self.body_ang_vel(self.board_id)[:, 2].abs()
        return (yaw_rate / 0.5).clamp(0.0, 1.0)

    def _technique(self) -> dict[str, torch.Tensor]:
        """Low, facing forward, angulated. Slope frame: x downhill, z = snow normal."""
        n = self.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(n, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(n, 3)
        up = self.up.expand(n, 3)
        height = self.body_pos(self.pelvis_id)[:, 2] - self.body_pos(self.board_id)[:, 2]
        # full in 0.55–0.70 m (stance 0.80), falling off below: low, not lying down
        low = ((self.pelvis_in_board_pos[2] - height) / 0.10).clamp(0.0, 1.0) * ((height - 0.40) / 0.15).clamp(0.0, 1.0)

        chest = quat_apply(self.body_quat(self.torso_id), ex)  # the humanoid faces its torso's +x
        chest = chest - (chest * up).sum(-1, keepdim=True) * up
        v = self.body_lin_vel(self.board_id)
        v = v - (v * up).sum(-1, keepdim=True) * up
        facing = ((chest * v).sum(-1) / (chest.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-6)).clamp(0.0, 1.0)

        across = quat_apply(self.body_quat(self.board_id), ey)  # the board's edge-to-edge axis
        torso_up = quat_apply(self.body_quat(self.torso_id), up)
        torso_roll = torch.atan2((torso_up * across).sum(-1), (torso_up * up).sum(-1)).abs()
        edge = self.snow_info["edge_angle"].abs() if self.snow_info else torch.zeros(n, device=self.device)
        angulation = (((edge - torso_roll) / math.radians(20.0)).clamp(0.0, 1.0) * (edge > math.radians(10.0))
                      * self.turning())
        self._track("pelvis_height_m", height)
        self._track("angulation_deg", (edge - torso_roll).rad2deg())
        return {"low": low, "facing": facing, "angulation": angulation}

    def _track(self, name: str, value: torch.Tensor):
        self.episode_sums.setdefault(name, torch.zeros_like(value))
        self.episode_sums[name] += value

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._log_episodes(env_ids)
        self.last_turns[env_ids] = self.turn_count[env_ids]
        lo, hi = self.cfg.speed_cmd
        self.v_cmd[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
        for buf in (self.turn_side, self.turn_swing, self.turn_count, self.since_turn):
            buf[env_ids] = 0.0
        m = len(env_ids)
        rand = lambda lo_hi: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(m, device=self.device)
        tan_a = torch.tan(torch.deg2rad(rand(self.cfg.guide_amp_deg)))
        shortest = 2 * math.pi * self.cfg.guide_min_radius * tan_a
        longest = torch.minimum(1.3 * shortest, 2 * math.pi * self.cfg.guide_max_swing / tan_a)
        wavelength = shortest + (longest - shortest).clamp(min=0.0) * torch.rand(m, device=self.device)
        self.guide_k[env_ids] = 2 * math.pi / wavelength
        self.guide_y[env_ids] = wavelength * tan_a / (2 * math.pi)
        self.guide_phase[env_ids] = math.pi * (torch.rand(m, device=self.device) < 0.5).float()  # swing left or right
        super()._reset_idx(env_ids)
        index_fill_(self.actions, env_ids, 0.0)
        index_fill_(self.prev_actions, env_ids, 0.0)

    def _log_episodes(self, env_ids: torch.Tensor):
        """Per-episode averages to TensorBoard: rewards per second, errors and speed per step, turns per episode."""
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
        seconds = steps * self.step_dt
        log["Episode_Snow/turns"] = self.turn_count[env_ids].mean()
        log["Episode_Snow/turns_per_10s"] = (self.turn_count[env_ids] / seconds * 10).mean()
        log["Episode_Termination/fell"] = (self.reset_terminated[env_ids] & ~self.out_of_bounds[env_ids]).float().mean()
        log["Episode_Termination/fence"] = self.out_of_bounds[env_ids].float().mean()
        log["Episode_Snow/seconds"] = seconds.mean()
        log["Guide/level"] = torch.tensor(self.guide_fade())
        log["Guide/err_ema_deg"] = torch.tensor(self.guide_err_ema)
        self.extras["log"] = log

    def _start_state(self, env_ids: torch.Tensor):
        """Start already doing the task: sideways at rest on the heel edge, or riding at the commanded speed."""
        n = len(env_ids)
        jitter = math.radians(5.0) * (2 * torch.rand(n, device=self.device) - 1)
        zeros = torch.zeros(n, device=self.device)
        if self.cfg.task == "sideslip":
            # At rest, with the centre of mass right above the heel edge: slope angle plus the lean to the board's
            # edge. From flat the rider rolls onto the toe edge and falls in ~0.7 s; with any downhill speed the
            # biting edge stops the board in one step and the body trips over it.
            lean = math.atan2(self.board_half_width, self.pelvis_in_board_pos[2].item())
            yaw = math.pi / 2 + jitter
            edge = torch.full_like(yaw, -(math.radians(self.slope_deg) + lean))
            return yaw, edge, torch.zeros(n, 3, device=self.device), zeros
        # piste centre, nose first, at start_speed_turns: along the guide while it's on, else within
        # ± start_heading_deg of the fall line
        if self.guide_fade() > 0:
            heading = torch.atan(self.guide_y[env_ids] * self.guide_k[env_ids] * self.guide_phase[env_ids].cos())
        else:
            heading = math.radians(self.cfg.start_heading_deg) * (2 * torch.rand(n, device=self.device) - 1)
        direction = torch.stack([heading.cos(), heading.sin(), zeros], dim=-1)
        lo, hi = self.cfg.start_speed_turns
        speed = lo + (hi - lo) * torch.rand(n, device=self.device)
        velocity = speed.unsqueeze(-1) * direction
        return heading + jitter, zeros, velocity, zeros


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(angle.sin(), angle.cos())
