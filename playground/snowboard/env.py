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
    min_turn_deg = 55.0  # a turn finishes 50-60° across the hill (80° was too much: the rider headed back up the hill)
    turn_bonus_from_deg = 25.0  # the turn bonus ramps from here ...
    turn_bonus_full_deg = 60.0  # ... to full here, so deeper turns always pay more and early half-turns still teach
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
    guide_amp_deg = (50.0, 60.0)  # angle at which the guide crosses the fall line (was 35-50: a narrow S; 60 is the most the swing limit allows)
    guide_min_radius = 4.5  # m: a 50° edge turns a ~6 m sidecut in ~4 m (was 9 for gentle edges)
    guide_max_swing = 17.0  # m (fences at ±20)
    guide_lookahead = 6.0  # m, pure pursuit back onto the guide
    run_length: float | None = None  # m; end the episode (as a time-out) this far down (play.py sets it)

    # Reward: what a good run achieves, not how to move. An earlier 17-term version (upright, low, facing,
    # angulation, pose, ...) hand-designed the technique, its terms fought each other and each one was gamed.
    # Each term is multiplied by step_dt, except turn and fall (once per event).
    rew_alive = 1.0
    rew_speed = 2.0  # exp(-((|v| - v_cmd) / σ_v)²): the reason to turn
    rew_speed_cap = -10.0  # × clip((|v| - speed_cap) / 2, 0, 1): over the cap is clearly worse than turning
    speed_cap = 12.0  # m/s; a clean carve on 15° settles at ~19-21 m/s, so the cap forces turns finished across the hill
    # carve: edged × turning × clean × low = clip(|θ|/edge_target) · clip(|ψ̇_board|/0.5) · exp(-(β/σ)²) · (0.5 + 0.5 low)
    fall_height = 0.15  # pelvis above the snow [m]: at the apex of a deep carve the hips are ~0.2 m from it (was 0.45)
    rew_carve = 2.0
    edge_target_deg = 20.0  # full carve credit at or above this edge angle. Carve: 40 (see __post_init__)
    rew_low = 1.0  # turn tasks, always: pelvis 0.30–0.45 m above the board, ramping up from the 0.80 m stance; never sit
                   # down (0 at 0.12 m: an ungated version was gamed by leaning back)
    rew_edge_switch = 10.0  # turn tasks: per edge change, the held edge (smoothed, ≥20°) flips from heel to toe or back
    rew_wrong_edge = -4.0  # turn tasks: held edge on the side that turns the wrong way for the S (heel when it needs toe and back): clip(angle / 30°)
    rew_release = 2.0  # turn tasks: once the turn has swung (≥30-55°), rolling the held edge toward the OTHER side, dense: clip(roll rate / 1 rad/s)
    rew_deep_edge = -3.0  # turn tasks: held edge beyond 60°, ramping over 15°: too deep to roll back out of (a rider sat at -75°)
    rew_edge_hold = -8.0  # turn tasks: staying on one edge longer than edge_hold_after_s: per second, ramping over edge_hold_ramp_s
    edge_hold_after_s = 3.5
    edge_hold_ramp_s = 2.0
    rew_traverse = -3.0  # turn tasks: no completed turn for traverse_after_s: per second, ramping to full over traverse_ramp_s
    traverse_after_s = 3.0
    traverse_ramp_s = 3.0
    rew_height = 1.0  # turn tasks: pelvis height follows the edge, tall (0.50 m) at the edge change, 0.25 m at 45° and up
    height_sigma = 0.10
    rew_swing = 2.0  # carve: heading across the hill while edged: clip(|heading| / min_turn) × clip(edge / edge_target). The turn bonus only pays at 80°; this is the slope to it
    rew_edge_rate = -2.0  # turn tasks: flipping the edge faster than ~3 rad/s (chatter; a real edge change is slower): clip((|d edge/dt| - 3) / 2, 0, 1)
    rew_hip_tail = 1.0  # turn tasks: hips pushed back toward the tail, behind the feet's midpoint along the board: clip(offset / 0.10 m)
    rew_balance = 1.0  # turn tasks: centre of mass between the feet, front/back load split ~50/50 (static estimate): exp(-((back share - 0.5) / 0.15)²)
    rew_uphill = -5.0  # turn tasks: heading more than 90° from the fall line is riding back up the hill: clip((|heading| - 90°) / 15°, 0, 1)
    rew_bow = 1.0  # turn tasks: upper body bowed forward, torso inclined from the snow normal: 45° flat -> 65° at 45° edge; 0 by 80° (lying)
    rew_arm = 0.5  # turn tasks: one hand low, the other high (the outside arm lifts for balance) while edged
    rew_hand = 1.0  # turn tasks: the lowest hand near the snow while edged (the hand drags at the apex, hangs low at the edge change)
    hand_scale = 0.30  # m: exp(-(lowest hand height / scale)²) × clip(edge / edge_target)
    rew_straight = -3.0  # turn tasks: riding straight down the fall line, exp(-(heading / straight_sigma)²), costs this per second
    straight_sigma_deg = 15.0
    rew_facing = 1.0  # turn tasks: shoulders open, chest toward the travel direction (the nose): max(cos, 0)
    rew_hip = 1.0  # turn tasks: the folded L from head to knee: torso-thigh angle 180° (straight) -> 0, 45° or tighter -> 1
    rew_slip = 2.0  # skid / side slip only (the carve term covers slip for carving): exp(-(slip error / σ_slip)²)
    rew_heading = 2.0  # side slip only: exp(-(heading / σ)²), straight down the fall line
    rew_turn = 5.0  # per completed turn (turn tasks): links the turns
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
        self.prev_psi = torch.zeros(n, device=dev)  # path heading / edge angle of the previous step, for their rates
        self.prev_edge = torch.zeros(n, device=dev)
        self.psi_rate = torch.zeros(n, device=dev)  # smoothed path heading rate [rad/s] (~0.3 s): chatter averages out
        self.edge_rate = torch.zeros(n, device=dev)  # edge angle rate [rad/s]
        self.edge_side = torch.zeros(n, device=dev)  # last held edge side (-1 heel / +1 toe, 0 none yet) and time on it
        self.since_switch = torch.zeros(n, device=dev)
        self.edge_switch = torch.zeros(n, device=dev)  # 1 on the step the held edge flips
        self.edge_ema_rate = torch.zeros(n, device=dev)  # d(edge_ema)/dt [rad/s]
        self.edge_ema = torch.zeros(n, device=dev)  # signed edge angle smoothed over ~0.6 s: chatter cancels, a held edge stays
        self.turn_side = torch.zeros(n, device=dev)  # side the path has swung to: -1, 0 (not yet), +1
        self.turn_swing = torch.zeros(n, device=dev)  # largest |heading| on that side so far [rad]
        self.turn_count = torch.zeros(n, device=dev)
        self.last_turns = torch.zeros(n, device=dev)  # turns of each env's previous episode (read after a reset)
        self.since_turn = torch.zeros(n, device=dev)  # s since the last completed turn (or the start)
        self.turn_done = torch.zeros(n, dtype=torch.bool, device=dev)  # a turn completed this step
        self.turn_credit = torch.zeros(n, device=dev)  # its bonus, 0..1 by how deep it swung
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
        first = self.episode_length_buf <= 1  # right after a reset: no previous step to difference against
        rate = torch.where(first, torch.zeros_like(psi), _wrap(psi - self.prev_psi) / self.step_dt).clamp(-3.0, 3.0)
        self.psi_rate = torch.where(first, torch.zeros_like(psi), 0.9 * self.psi_rate + 0.1 * rate)
        self.prev_psi = psi
        theta = self.snow_info["edge_angle"] if self.snow_info else torch.zeros_like(psi)
        self.edge_rate = torch.where(first, torch.zeros_like(psi), (theta - self.prev_edge) / self.step_dt)
        self.prev_edge = theta
        new_ema = torch.where(first, theta, 0.95 * self.edge_ema + 0.05 * theta)
        self.edge_ema_rate = torch.where(first, torch.zeros_like(psi), (new_ema - self.edge_ema) / self.step_dt)
        self.edge_ema = new_ema
        held = torch.sign(self.edge_ema) * (self.edge_ema.abs() > math.radians(20.0))
        self.edge_switch = ((held != 0) & (self.edge_side != 0) & (held != self.edge_side)).float()
        self.edge_side = torch.where(held != 0, held, self.edge_side)
        self.since_switch = torch.where(self.edge_switch > 0, torch.zeros_like(self.since_switch),
                                        self.since_switch + self.step_dt)
        side = torch.sign(psi) * (psi.abs() > 0.02)
        crossed = (side != 0) & (self.turn_side != 0) & (side != self.turn_side)
        crossed = crossed & (self.board_speed() > 5.0)  # a turn at speed: spinning in place doesn't count
        self.turn_done = crossed & (self.turn_swing >= self.min_turn)  # counted (turns_per_10s, eval)
        lo, hi = math.radians(self.cfg.turn_bonus_from_deg), math.radians(self.cfg.turn_bonus_full_deg)
        self.turn_credit = crossed.float() * ((self.turn_swing - lo) / (hi - lo)).clamp(0.0, 1.0)  # rewarded
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
        # also fallen: lying on the snow with the board on its side was never caught by the pelvis-height check (the
        # board's edge held the pelvis above 0.45 m), and the policy learned to lie there and spin for carve reward
        torso_up = quat_apply(self.body_quat(self.torso_id), self.up.expand(self.num_envs, 3))
        # thresholds leave room for deep carves (reference-image/): torso inclined ~60-70°, hips just above the snow
        lying = (torso_up * self.up).sum(-1) < math.cos(math.radians(80.0))
        sitting = self.pelvis_height() < 0.12  # below the low band (0.30-0.45 m, reward 0 at 0.15)
        on_side = self.snow_info["edge_angle"].abs() > math.radians(85.0) if self.snow_info else torch.zeros_like(lying)
        fallen = fallen | lying | sitting | on_side
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
        edge_abs = self.edge_ema.abs()  # the held edge: flipping it back and forth (chatter) averages out to nothing
        # which way the S asks for the path to turn: away from the fall line until the swing reaches min_turn, then
        # back through it (the turn side and swing are in the observation). The heading rate has the sign opposite
        # to the edge angle (heel edge, θ < 0: heading up), so the edge the S needs has sign -d.
        d = torch.where(self.turn_swing >= self.min_turn, -self.turn_side, self.turn_side)
        right_edge = -d * self.edge_ema  # > 0 on the edge that turns the right way
        edge = torch.where(d != 0, right_edge.clamp(min=0.0), edge_abs)  # only the right edge counts for the edge-gated terms
        wrong = torch.where(d != 0, (-right_edge).clamp(min=0.0), torch.zeros_like(edge_abs))
        self._track("edge_dir_deg", torch.where(d != 0, right_edge, torch.zeros_like(edge_abs)).rad2deg())
        clean = torch.exp(-(self.slip_angle() / math.radians(cfg.sigma_slip_deg)).square())
        tech = self._technique() if turns else {}  # also logs pelvis height and angulation
        hip_tail, back_share = self.stance_balance() if turns else (zeros, zeros)
        terms = {
            "alive": (~self.reset_terminated).float(),
            "speed": torch.exp(-((speed - self.v_cmd) / cfg.sigma_speed).square()),
            "speed_cap": ((speed - cfg.speed_cap) / 2.0).clamp(0.0, 1.0),
            # edged, turning and clean at once: tipped but straight (leaning back into the fence), or turning flat, or
            # skidding, all earn ~0
            # × (0.5 + 0.5 low): being low pays only while carving (a low crouch riding straight earns nothing), and
            # carving tall still earns half, so the squat isn't required before the carve can be learned
            # × moving: no carve credit below ~4 m/s, full from 8 (spinning in place earned it before)
            "carve": (edge / math.radians(cfg.edge_target_deg)).clamp(0.0, 1.0) * self.path_turning() * clean
            * (0.5 + 0.5 * self.low()) * ((speed - 4.0) / 4.0).clamp(0.0, 1.0) if cfg.task == "carve" else zeros,
            "low": self.low() if turns else zeros,
            "facing": tech["facing"] if turns else zeros,
            "hand": ((edge / math.radians(cfg.edge_target_deg)).clamp(0.0, 1.0)
                     * torch.exp(-(self.body_pos(self.key_ids[:2])[..., 2].min(-1).values / cfg.hand_scale).square())
                     if turns else zeros),
            "bow": (lambda incl: (incl / (math.radians(45.0) + math.radians(20.0) * (edge / math.radians(45.0)).clamp(0.0, 1.0)))
                    .clamp(0.0, 1.0) * ((math.radians(80.0) - incl) / math.radians(10.0)).clamp(0.0, 1.0))(
                self.torso_incline()) if turns else zeros,
            "swing": ((psi.abs() / self.min_turn).clamp(0.0, 1.0) * (edge / math.radians(cfg.edge_target_deg)).clamp(0.0, 1.0)
                      * self.path_turning() if cfg.task == "carve" else zeros),
            "edge_rate": ((self.edge_rate.abs() - 3.0) / 2.0).clamp(0.0, 1.0) if turns else zeros,
            "hip_tail": (hip_tail / 0.10).clamp(0.0, 1.0) if turns else zeros,
            "balance": torch.exp(-((back_share - 0.5) / 0.15).square()) if turns else zeros,
            "uphill": ((psi.abs() - math.radians(90.0)) / math.radians(15.0)).clamp(0.0, 1.0) if turns else zeros,
            "release": ((-self.edge_side * self.edge_ema_rate / 1.0).clamp(0.0, 1.0)
                        * ((self.turn_swing - math.radians(30.0)) / math.radians(25.0)).clamp(0.0, 1.0) if turns else zeros),
            "wrong_edge": (wrong / math.radians(30.0)).clamp(0.0, 1.0) if turns else zeros,
            "deep_edge": ((edge_abs - math.radians(60.0)) / math.radians(15.0)).clamp(0.0, 1.0) if turns else zeros,
            "edge_hold": ((self.since_switch - cfg.edge_hold_after_s) / cfg.edge_hold_ramp_s).clamp(0.0, 1.0) if turns else zeros,
            "traverse": ((self.since_turn - cfg.traverse_after_s) / cfg.traverse_ramp_s).clamp(0.0, 1.0) if turns else zeros,
            "height": (torch.exp(-((self.pelvis_height() - (0.50 - 0.25 * (edge / math.radians(45.0)).clamp(0.0, 1.0)))
                                   / cfg.height_sigma).square()) if turns else zeros),
            "arm": (((lambda hz: (hz.max(-1).values - hz.min(-1).values) / 0.5)(self.body_pos(self.key_ids[:2])[..., 2]))
                    .clamp(0.0, 1.0) * (edge / math.radians(cfg.edge_target_deg)).clamp(0.0, 1.0) if turns else zeros),
            "straight": torch.exp(-(psi / math.radians(cfg.straight_sigma_deg)).square()) if turns else zeros,
            "hip": self.hip_fold() if turns else zeros,
            "slip": zeros if cfg.task == "carve" else torch.exp(-(e_slip / math.radians(cfg.sigma_slip_deg)).square()),
            "heading": zeros if turns else torch.exp(-(psi / math.radians(cfg.sigma_heading_deg)).square()),
            "action_rate": (self.actions - self.prev_actions).square().sum(-1),
        }
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
        if turns:  # a bonus per edge change, not a rate: the S needs both edges
            r = cfg.rew_edge_switch * self.edge_switch
            reward += r
            self._track("rew_edge_switch", r)
        if turns:  # a bonus per completed turn, not a rate
            r = cfg.rew_turn * self.turn_credit
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

    def pelvis_height(self) -> torch.Tensor:
        """Pelvis above the board along the snow normal [m] (stance 0.80)."""
        return self.body_pos(self.pelvis_id)[:, 2] - self.body_pos(self.board_id)[:, 2]

    def low(self) -> torch.Tensor:
        """0..1: a ramp from 0 at the 0.80 m stance to full at 0.45 m, full down to 0.25 m (hips near the snow, as at the
        apex of a deep carve), then 0 at 0.12 m (sitting). The ramp starts at the stance so a standing rider feels a gradient."""
        h = self.pelvis_height()
        return ((0.80 - h) / 0.35).clamp(0.0, 1.0) * ((h - 0.12) / 0.13).clamp(0.0, 1.0)

    def stance_balance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(hips behind the feet's midpoint along the board [m], share of the weight on the back foot [0..1]).

        t points from the front foot (left, toward the nose) to the back foot (right, toward the tail). The share is
        static: where the rider's centre of mass sits between the feet, 0 = over the front foot, 1 = over the back
        foot (it ignores accelerations, so it is a proxy for the front/back load split)."""
        if not hasattr(self, "_com_ids"):
            mass = self.robot.data.body_mass.torch[0]
            self._com_ids = [i for i in range(mass.shape[0]) if i not in self.board_ids]
            self._com_mass = mass[self._com_ids]
        back, front = self.body_pos(self.key_ids[2]), self.body_pos(self.key_ids[3])
        t = back - front
        t = t - (t * self.up).sum(-1, keepdim=True) * self.up
        span = t.norm(dim=-1).clamp_min(0.2)
        t = t / span.unsqueeze(-1)
        mid = (back + front) / 2
        hip_tail = ((self.body_pos(self.pelvis_id) - mid) * t).sum(-1)
        pos = self.body_pos(self._com_ids)
        com = (pos * self._com_mass.view(1, -1, 1)).sum(1) / self._com_mass.sum()
        share = (((com - front) * t).sum(-1) / span).clamp(0.0, 1.0)
        self._track("load_back", share)
        self._track("hip_tail_m", hip_tail)
        return hip_tail, share

    def torso_incline(self) -> torch.Tensor:
        """Angle of the torso axis from the snow normal [rad]: 0 upright, ~80° is lying (a fall)."""
        torso_up = quat_apply(self.body_quat(self.torso_id), self.up.expand(self.num_envs, 3))
        return (torso_up * self.up).sum(-1).clamp(-1.0, 1.0).acos()

    def hip_fold(self) -> torch.Tensor:
        """0..1: the folded L from head to knee. The angle between the torso line (hip to head) and the thigh line
        (hip to knee), per leg, averaged: 0 with torso and leg in one line (180°), 1 at 45° or tighter (a 30-45° L),
        tapering to 0 at 15° so the chest doesn't fold flat onto the thighs."""
        if not hasattr(self, "_fold_ids"):
            names = self.robot.body_names
            self._fold_ids = [(names.index(f"{s}_thigh"), names.index(f"{s}_shin")) for s in ("right", "left")]
            self._head_id = names.index("head")
        head = self.body_pos(self._head_id)
        fold = []
        for thigh, shin in self._fold_ids:
            hip = self.body_pos(thigh)  # the thigh link starts at the hip, the shin link at the knee
            torso, leg = head - hip, self.body_pos(shin) - hip
            cos = (torso * leg).sum(-1) / (torso.norm(dim=-1) * leg.norm(dim=-1)).clamp_min(1e-6)
            angle = cos.clamp(-1.0, 1.0).acos()
            fold.append(((math.pi - angle) / math.radians(135.0)).clamp(0.0, 1.0)
                        * ((angle - math.radians(15.0)) / math.radians(15.0)).clamp(0.0, 1.0))
        return torch.stack(fold).mean(0)

    def path_turning(self) -> torch.Tensor:
        """0..1: how hard the path itself turns, from the smoothed path heading rate (full at 0.5 rad/s). Edge chatter
        spins the board without changing where it goes, so it earns nothing here (board yaw rate paid it)."""
        return (self.psi_rate.abs() / 0.5).clamp(0.0, 1.0)

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
        low = self.low()

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
        for buf in (self.turn_side, self.turn_swing, self.turn_count, self.since_turn, self.edge_side, self.since_switch):
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
