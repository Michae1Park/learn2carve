"""Snowboard tasks on the fenced piste: side slip, skid or carve.

    task      heading                      slip target
    sideslip  straight down the fall line  -90° (heel edge)
    skid      linked turns                 20°
    carve     linked turns                 0

The turn tasks command a speed below the straight-run speed, so the S comes from the physics: the only way to hold
it is to turn across the hill. Crossing a fence ends the episode. See docs/RL_TRAINING.md.

Import after Isaac Sim launched.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass, index_fill_
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from envs.carve_env import GRAVITY, CarveEnv, CarveEnvCfg, base_obs_size, num_dof
from envs.mountain import FENCE_HALF_WIDTH
from envs.snow import board_frame

TASKS = ("sideslip", "skid", "carve")
TASK_OBS = 19
SYNERGY_OBS = 2
BALANCE_RADIUS = 7.6  # m, sidecut radius used to pick the edge that balances a speed
HELD_EDGE = math.radians(20.0)
TURN_MIN_SPEED = 5.0  # m/s
STANCE_PELVIS_HEIGHT = 0.80  # m above the board
SITTING_HEIGHT = 0.12
ON_SIDE_EDGE = math.radians(85.0)
REVERSE_MIN_SPEED = 2.0
FALL_REASONS = ("height", "lying", "sitting", "on_side", "reverse")
RSI_MIN_SAMPLES = 256
RSI_PER_STEP = 32

deg = math.radians


def ramp(x: torch.Tensor, start: float, width: float) -> torch.Tensor:
    """0 at start, 1 at start + width."""
    return ((x - start) / width).clamp(0.0, 1.0)


def gauss(error: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-(error / sigma).square())


def wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(angle.sin(), angle.cos())


def uniform(lo_hi, n: int, device) -> torch.Tensor:
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(n, device=device)


def coin(n: int, device, p: float = 0.5) -> torch.Tensor:
    return torch.rand(n, device=device) < p


@configclass
class SnowboardEnvCfg(CarveEnvCfg):
    episode_length_s = 12.0
    observation_space = 87 + 28 + TASK_OBS  # recomputed from the DoF
    action_scale = 0.25

    task: str = "sideslip"
    mountain = True

    speed_cmd = (0.5, 2.5)  # m/s; turn tasks: see __post_init__
    min_turn_deg = 55.0  # swing that makes a turn count when the path crosses back over the fall line
    turn_bonus_from_deg = 25.0
    turn_bonus_full_deg = 60.0
    skid_slip_deg = 20.0
    start_heading_deg = 20.0
    fence_half_width = FENCE_HALF_WIDTH

    # guide S line for the turn tasks: fades while followed, comes back when lost; each episode draws its own S
    guide_steps: int = 3000  # 0 = off
    guide_fade_below_deg = 15.0
    guide_restore_above_deg = 30.0
    guide_amp_deg = (50.0, 60.0)  # fall-line crossing angle
    guide_min_radius = 4.5  # m, tightest bend
    guide_max_swing = 17.0  # m
    guide_lookahead = 6.0  # m
    run_length: float | None = None  # m, ends the episode as a time-out

    # reward: each term × step_dt, except turn, edge switch and fall (per event)
    rew_alive = 1.0
    rew_speed = 2.0
    rew_speed_cap = -10.0
    speed_cap = 12.0  # m/s
    carve_start_toe_frac = 0.5
    # reset from recorded mid-episode toe carves, so toe → heel is practised where riders get stuck
    rsi_frac = 0.0
    rsi_buffer = 8192
    rsi_min_s = 3.0
    rsi_toe_deg = 12.0
    lying_deg = 80.0  # torso tilt; a deep toe carve bows it 55-70°
    fall_height = 0.15
    rew_carve = 2.0
    edge_target_deg = 20.0  # carve: 40, see __post_init__
    rew_low = 1.0
    rew_edge_switch = 10.0
    reverse_slip_deg = 100.0  # tail first at speed counts as a fall

    # edge command: a signed edge angle to track, heel → base → toe
    edge_cmd: bool = False
    rew_edge_cmd = 6.0
    edge_cmd_sigma_deg = 12.0
    rew_edge_side = 0.0
    edge_cmd_clean: bool = True
    rew_pose_ref = 0.0  # match the double pendulum's legs and COM lean (scripts/pendulum_rl.py)
    rew_tilt = 0.0
    tilt_free_deg = 50.0
    rew_grab = 0.0  # inside hand toward its grab point on the board
    grab_sigma_m = 0.25
    grab_linear_m = 0.0  # > 0: linear falloff instead of the Gaussian
    rew_rise = 0.0  # stand tall while the command crosses flat
    carve_start: bool = False  # start each episode balanced on an edge
    edge_cmd_hold: bool = False  # command that edge for the whole episode
    pose_ref_legs = 1.15
    pose_ref_com = 0.77
    pose_ref_sigma_deg = 15.0
    edge_cmd_amp_deg = (20.0, 50.0)  # curriculum level 0 → 1
    edge_cmd_period_s = (4.6, 4.0)  # level 0 → 1, ±15 % per episode
    edge_cmd_level_init = 0.0
    edge_cmd_level_fixed = False
    edge_cmd_mode = "timer"  # "timer": periodic. "heading": switch once the turn reaches edge_cmd_heading_deg
    edge_cmd_heading_deg = 50.0
    edge_cmd_min_s = 1.0
    edge_cmd_max_s = 3.0
    edge_cmd_ramp_s = 0.6
    edge_cmd_center_deg_per_m = 0.0  # heading mode: shift the switch heading to steer back to the piste centre
    edge_cmd_center_max_deg = 25.0
    edge_cmd_sharpness = 1.8
    edge_cmd_level_up_deg = 12.0
    edge_cmd_level_steps = 2000
    switch_flat_deg = 5.0  # an edge change counts only through a flat base
    rew_wrong_edge = -4.0
    rew_release = 2.0
    rew_deep_edge = -3.0
    rew_edge_hold = -8.0
    edge_hold_after_s = 3.5
    edge_hold_ramp_s = 2.0
    rew_traverse = -3.0
    traverse_after_s = 3.0
    traverse_ramp_s = 3.0
    rew_height = 1.0
    height_sigma = 0.10
    rew_swing = 2.0
    rew_edge_rate = -2.0
    rew_hip_tail = 1.0
    rew_balance = 1.0
    rew_uphill = -5.0
    rew_bow = 1.0
    rew_arm = 0.5
    rew_hand = 1.0
    hand_scale = 0.30  # m
    rew_straight = -3.0
    straight_sigma_deg = 15.0
    rew_facing = 1.0
    rew_hip = 1.0
    rew_slip = 2.0
    rew_heading = 2.0
    rew_turn = 5.0
    rew_fall = -10.0
    rew_guide = 4.0
    rew_action_rate = -0.01
    start_speed_turns = (4.0, 6.0)  # m/s
    sigma_slip_deg = 15.0
    sigma_heading_deg = 30.0
    sigma_speed = 5.0  # m/s; narrower gives no gradient far from the command

    def __post_init__(self):
        super().__post_init__()
        if self.task != "sideslip":
            self.speed_cmd = (8.0, 11.0)
        if self.task == "carve":
            self.edge_target_deg = 40.0


# per-step reward terms, summed in this order
STEP_TERMS = ("alive", "speed", "speed_cap", "carve", "low", "facing", "hand", "bow", "swing", "edge_rate", "hip_tail",
              "balance", "uphill", "release", "wrong_edge", "deep_edge", "edge_hold", "traverse", "height", "arm",
              "straight", "hip", "slip", "heading", "action_rate")


class SnowboardEnv(CarveEnv):
    cfg: SnowboardEnvCfg

    def __init__(self, cfg: SnowboardEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.task not in TASKS:
            raise ValueError(f"unknown task {cfg.task!r}, pick one of: {', '.join(TASKS)}")
        cfg.observation_space = base_obs_size(cfg) + num_dof(cfg) + TASK_OBS + (SYNERGY_OBS if cfg.synergy else 0)
        self._init_task_state(cfg.scene.num_envs, cfg.sim.device, cfg)  # super().__init__ resets every env
        super().__init__(cfg, render_mode, **kwargs)
        self.min_turn = deg(cfg.min_turn_deg)
        self.turns = cfg.task != "sideslip"
        self._find_task_bodies()

    def _init_task_state(self, n: int, device, cfg: SnowboardEnvCfg):
        def zeros(dtype=torch.float32):
            return torch.zeros(n, dtype=dtype, device=device)

        self.v_cmd = zeros()
        self.prev_psi = zeros()
        self.prev_edge = zeros()
        self.psi_rate = zeros()
        self.edge_rate = zeros()
        self.edge_side = zeros()  # last held edge: -1 heel, +1 toe, 0 none yet
        self.since_switch = zeros()
        self.base_touched = zeros(torch.bool)
        self.edge_switch = zeros()
        self.edge_ema_rate = zeros()
        self.edge_ema = zeros()  # chatter averages out, a held edge stays
        self.turn_side = zeros()
        self.turn_swing = zeros()
        self.turn_count = zeros()
        self.last_turns = zeros()
        self.since_turn = zeros()
        self.turn_done = zeros(torch.bool)
        self.turn_credit = zeros()
        self.guide_k = zeros()
        self.guide_y = zeros()
        self.guide_phase = zeros()
        self.guide_level = 1.0
        self.guide_err_ema = 90.0
        self.cmd_period = torch.ones(n, device=device)
        self.cmd_phase = zeros()
        self.cmd_level = cfg.edge_cmd_level_init
        self.cmd_hold = zeros()
        self.cmd_err_ema = 90.0
        self.cmd_side = zeros()  # heading mode: +1 toe, -1 heel
        self.cmd_from = zeros()
        self.cmd_since = zeros()
        self.rsi_started = zeros()
        self.rsi_count, self.rsi_next = 0, 0
        self.prev_actions = torch.zeros(n, num_dof(cfg), device=device)
        self.out_of_bounds = zeros(torch.bool)
        self.fall_why = torch.zeros(n, len(FALL_REASONS), dtype=torch.bool, device=device)
        self.episode_sums: dict[str, torch.Tensor] = {}

    def _find_task_bodies(self):
        names = self.robot.body_names
        self.right_hand_id, self.left_hand_id, self.back_foot_id, self.front_foot_id = self.key_ids
        self.hand_ids = self.key_ids[:2]
        self.head_id = names.index("head")
        self.hip_knee_ids = [(names.index(f"{side}_thigh"), names.index(f"{side}_shin")) for side in ("right", "left")]
        self.rider_ids = [i for i in range(len(names)) if i not in self.board_ids]
        self._init_rsi_buffer()

    # -- board state --------------------------------------------------------------------------------------------------

    def position(self) -> torch.Tensor:
        """Board (x downhill, y across) relative to the env's start."""
        return self.body_pos(self.board_id)[:, :2] - self.scene.env_origins[:, :2]

    def board_speed(self) -> torch.Tensor:
        return self.body_lin_vel(self.board_id).norm(dim=-1)

    def path_heading(self) -> torch.Tensor:
        """0 = straight down the fall line."""
        v = self.body_lin_vel(self.board_id)
        return torch.atan2(v[:, 1], v[:, 0])

    def slip_angle(self) -> torch.Tensor:
        info = self.snow_info or self._zero_snow_info()
        return torch.atan2(info["v_lat"], info["v_long"])

    def board_yaw(self) -> torch.Tensor:
        nose = quat_apply(self.body_quat(self.board_id), self.axis(0))
        return torch.atan2(nose[:, 1], nose[:, 0])

    def axis(self, i: int) -> torch.Tensor:
        unit = torch.zeros(3, device=self.device)
        unit[i] = 1.0
        return unit.expand(self.num_envs, 3)

    def slip_target(self) -> torch.Tensor:
        target = {"sideslip": -math.pi / 2, "skid": deg(self.cfg.skid_slip_deg), "carve": 0.0}[self.cfg.task]
        return torch.full((self.num_envs,), target, device=self.device)

    def slip_error(self) -> torch.Tensor:
        """Side slip: signed, heel edge only. Turns: |slip| - target, skidding either way."""
        beta = self.slip_angle()
        if self.cfg.task == "sideslip":
            return wrap(beta - self.slip_target())
        return beta.abs() - self.slip_target()

    def balanced_edge(self, speed: torch.Tensor) -> torch.Tensor:
        """|Edge| whose sidecut arc balances this speed."""
        g_normal = GRAVITY * math.cos(deg(self.slope_deg))
        return torch.asin((speed.square() / (g_normal * BALANCE_RADIUS)).clamp(max=0.9))

    # -- rider posture ------------------------------------------------------------------------------------------------

    def pelvis_height(self) -> torch.Tensor:
        """Above the board along the snow normal (stance 0.80 m)."""
        return self.body_pos(self.pelvis_id)[:, 2] - self.body_pos(self.board_id)[:, 2]

    def low(self) -> torch.Tensor:
        """Ramps up from the stance to 0.45 m, full to 0.25 m, then down to 0 at sitting height."""
        h = self.pelvis_height()
        return ramp(STANCE_PELVIS_HEIGHT - h, 0.0, 0.35) * ramp(h, SITTING_HEIGHT, 0.13)

    def torso_up_cos(self) -> torch.Tensor:
        torso_up = quat_apply(self.body_quat(self.torso_id), self.up.expand(self.num_envs, 3))
        return (torso_up * self.up).sum(-1)

    def torso_incline(self) -> torch.Tensor:
        return self.torso_up_cos().clamp(-1.0, 1.0).acos()

    def rider_com(self) -> torch.Tensor:
        mass = self.robot.data.body_mass.torch[0, self.rider_ids]  # scaled to the rider only once the sim has stepped
        return (self.body_pos(self.rider_ids) * mass.view(1, -1, 1)).sum(1) / mass.sum()

    def stance_balance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(hips behind the feet's midpoint along the board [m], static share of the weight on the back foot)."""
        back, front = self.body_pos(self.back_foot_id), self.body_pos(self.front_foot_id)
        toward_tail = back - front
        toward_tail = toward_tail - (toward_tail * self.up).sum(-1, keepdim=True) * self.up
        span = toward_tail.norm(dim=-1).clamp_min(0.2)
        toward_tail = toward_tail / span.unsqueeze(-1)
        mid = (back + front) / 2
        hip_tail = ((self.body_pos(self.pelvis_id) - mid) * toward_tail).sum(-1)
        back_share = (((self.rider_com() - front) * toward_tail).sum(-1) / span).clamp(0.0, 1.0)
        self._track("load_back", back_share)
        self._track("hip_tail_m", hip_tail)
        return hip_tail, back_share

    def roll_leans(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Leans of board → pelvis and board → rider COM in the board's roll plane, + toward the engaged θ > 0 edge."""
        normal = self.up.expand(self.num_envs, 3)
        _, across, _ = board_frame(self.body_quat(self.board_id), normal)
        toe_side = -across
        board, pelvis = self.body_pos(self.board_id), self.body_pos(self.pelvis_id)

        def lean(d):
            return torch.atan2((d * toe_side).sum(-1), (d * normal).sum(-1))

        return lean(pelvis - board), lean(self.rider_com() - board)

    def hip_fold(self) -> torch.Tensor:
        """Torso-thigh angle per leg, averaged: 0 straight, 1 at 45° or tighter, back to 0 at 15°."""
        head = self.body_pos(self.head_id)
        folds = []
        for hip_id, knee_id in self.hip_knee_ids:
            hip = self.body_pos(hip_id)
            torso, thigh = head - hip, self.body_pos(knee_id) - hip
            cos = (torso * thigh).sum(-1) / (torso.norm(dim=-1) * thigh.norm(dim=-1)).clamp_min(1e-6)
            angle = cos.clamp(-1.0, 1.0).acos()
            folds.append(ramp(math.pi - angle, 0.0, deg(135.0)) * ramp(angle, deg(15.0), deg(15.0)))
        return torch.stack(folds).mean(0)

    def path_turning(self) -> torch.Tensor:
        """From the smoothed path heading rate: edge chatter spins the board without turning the path."""
        return ramp(self.psi_rate.abs(), 0.0, 0.5)

    def turning(self) -> torch.Tensor:
        return ramp(self.body_ang_vel(self.board_id)[:, 2].abs(), 0.0, 0.5)

    def facing(self) -> torch.Tensor:
        """Chest toward the travel direction, both flattened onto the snow."""
        up = self.up.expand(self.num_envs, 3)
        chest = quat_apply(self.body_quat(self.torso_id), self.axis(0))
        chest = chest - (chest * up).sum(-1, keepdim=True) * up
        v = self.body_lin_vel(self.board_id)
        v = v - (v * up).sum(-1, keepdim=True) * up
        return ((chest * v).sum(-1) / (chest.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-6)).clamp(0.0, 1.0)

    def _log_angulation(self):
        up = self.up.expand(self.num_envs, 3)
        across = quat_apply(self.body_quat(self.board_id), self.axis(1))
        torso_up = quat_apply(self.body_quat(self.torso_id), up)
        torso_roll = torch.atan2((torso_up * across).sum(-1), (torso_up * up).sum(-1)).abs()
        self._track("pelvis_height_m", self.pelvis_height())
        self._track("angulation_deg", (self.edge_angle().abs() - torso_roll).rad2deg())

    # -- guide S ------------------------------------------------------------------------------------------------------

    def guide_fade(self) -> float:
        if self.cfg.task == "sideslip" or self.cfg.guide_steps <= 0:
            return 0.0
        return self.guide_level

    def guide_heading(self) -> torch.Tensor:
        """The guide's direction at the board, pulled back onto it by pure pursuit."""
        x, y = self.position().unbind(-1)
        arg = self.guide_k * x + self.guide_phase
        y_ref = self.guide_y * arg.sin()
        return torch.atan(self.guide_y * self.guide_k * arg.cos()) - torch.atan((y - y_ref) / self.cfg.guide_lookahead)

    def _adapt_guide(self, error_deg: torch.Tensor):
        if self.cfg.guide_steps <= 0:
            return
        self.guide_err_ema = self._update_alive_ema(self.guide_err_ema, error_deg)
        step = 1.0 / self.cfg.guide_steps
        if self.guide_err_ema < self.cfg.guide_fade_below_deg:
            self.guide_level = max(0.0, self.guide_level - step)
        elif self.guide_err_ema > self.cfg.guide_restore_above_deg:
            self.guide_level = min(1.0, self.guide_level + step)

    def _update_alive_ema(self, ema: float, value: torch.Tensor) -> float:
        alive = ~self.reset_terminated
        if alive.any():
            ema = 0.99 * ema + 0.01 * value[alive].mean().item()
        return ema

    def _sample_guide(self, env_ids: torch.Tensor):
        """A carvable S: tightest bend ≥ guide_min_radius, swing ≤ guide_max_swing."""
        m = len(env_ids)
        tan_a = torch.tan(torch.deg2rad(uniform(self.cfg.guide_amp_deg, m, self.device)))
        shortest = 2 * math.pi * self.cfg.guide_min_radius * tan_a
        longest = torch.minimum(1.3 * shortest, 2 * math.pi * self.cfg.guide_max_swing / tan_a)
        wavelength = shortest + (longest - shortest).clamp(min=0.0) * torch.rand(m, device=self.device)
        self.guide_k[env_ids] = 2 * math.pi / wavelength
        self.guide_y[env_ids] = wavelength * tan_a / (2 * math.pi)
        self.guide_phase[env_ids] = math.pi * coin(m, self.device).float()

    # -- edge command ------------------------------------------------------------------------------------------------

    def edge_command(self, ahead_s: float = 0.0) -> torch.Tensor:
        """Commanded signed edge angle `ahead_s` from now (θ < 0 heel, > 0 toe); 0 when off."""
        cfg = self.cfg
        if not cfg.edge_cmd:
            return torch.zeros(self.num_envs, device=self.device)
        if cfg.edge_cmd_hold:
            return torch.sign(self.cmd_hold) * self.balanced_edge(self.board_speed())
        lo, hi = cfg.edge_cmd_amp_deg
        amplitude = deg(lo + self.cmd_level * (hi - lo))
        if cfg.edge_cmd_mode == "heading":
            progress = ((self.cmd_since + ahead_s) / cfg.edge_cmd_ramp_s).clamp(0.0, 1.0)
            return self.cmd_from + (self.cmd_side * amplitude - self.cmd_from) * progress
        t = self.episode_length_buf.float() * self.step_dt + ahead_s
        phase = 2 * math.pi * t / self.cmd_period + self.cmd_phase
        return amplitude * (cfg.edge_cmd_sharpness * phase.sin()).clamp(-1.0, 1.0)

    def _step_heading_cmd(self):
        """Switch edges once the turn has come round (heel turns swing the heading +, toe turns -), or on timeout."""
        cfg = self.cfg
        now = self.edge_command()
        self.cmd_since += self.step_dt
        psi, switch_heading = self.path_heading(), deg(cfg.edge_cmd_heading_deg)
        cap = cfg.edge_cmd_center_max_deg
        centring = torch.deg2rad((cfg.edge_cmd_center_deg_per_m * self.position()[:, 1]).clamp(-cap, cap))
        heel_done = (self.cmd_side < 0) & (psi > switch_heading - centring)
        toe_done = (self.cmd_side > 0) & (psi < -switch_heading - centring)
        timed_out = self.cmd_since > cfg.edge_cmd_max_s
        switch = (self.cmd_since > cfg.edge_cmd_min_s) & (heel_done | toe_done | timed_out)
        self.cmd_from = torch.where(switch, now, self.cmd_from)
        self.cmd_side = torch.where(switch, -self.cmd_side, self.cmd_side)
        self.cmd_since = torch.where(switch, torch.zeros_like(self.cmd_since), self.cmd_since)

    def _adapt_edge_cmd(self, error_deg: torch.Tensor):
        self.cmd_err_ema = self._update_alive_ema(self.cmd_err_ema, error_deg)
        if self.cfg.edge_cmd_level_fixed:
            return
        step = 1.0 / self.cfg.edge_cmd_level_steps
        if self.cmd_err_ema < self.cfg.edge_cmd_level_up_deg:
            self.cmd_level = min(1.0, self.cmd_level + step)
        elif self.cmd_err_ema > 2 * self.cfg.edge_cmd_level_up_deg:
            self.cmd_level = max(0.0, self.cmd_level - step)

    def _sample_edge_cmd(self, env_ids: torch.Tensor):
        m = len(env_ids)
        p0, p1 = self.cfg.edge_cmd_period_s
        self.cmd_period[env_ids] = (p0 + self.cmd_level * (p1 - p0)) * (0.85 + 0.3 * torch.rand(m, device=self.device))
        self.cmd_phase[env_ids] = math.pi * coin(m, self.device).float()
        if self.cfg.edge_cmd_mode == "heading":
            self.cmd_side[env_ids] = torch.where(coin(m, self.device), -1.0, 1.0)
            self.cmd_from[env_ids] = 0.0
            self.cmd_since[env_ids] = 0.0

    # -- turn and edge tracking ---------------------------------------------------------------------------------------

    def _update_turns(self):
        psi = self.path_heading()
        edge = self.edge_angle()
        just_reset = self.episode_length_buf <= 1
        zeros = torch.zeros_like(psi)

        rate = torch.where(just_reset, zeros, wrap(psi - self.prev_psi) / self.step_dt).clamp(-3.0, 3.0)
        self.psi_rate = torch.where(just_reset, zeros, 0.9 * self.psi_rate + 0.1 * rate)
        self.prev_psi = psi
        self.edge_rate = torch.where(just_reset, zeros, (edge - self.prev_edge) / self.step_dt)
        self.prev_edge = edge
        new_ema = torch.where(just_reset, edge, 0.95 * self.edge_ema + 0.05 * edge)
        self.edge_ema_rate = torch.where(just_reset, zeros, (new_ema - self.edge_ema) / self.step_dt)
        self.edge_ema = new_ema

        self._update_edge_switch(edge)

        side = torch.sign(psi) * (psi.abs() > 0.02)
        crossed = (side != 0) & (self.turn_side != 0) & (side != self.turn_side) & (self.board_speed() > TURN_MIN_SPEED)
        self.turn_done = crossed & (self.turn_swing >= self.min_turn)
        bonus_from, bonus_full = deg(self.cfg.turn_bonus_from_deg), deg(self.cfg.turn_bonus_full_deg)
        self.turn_credit = crossed.float() * ramp(self.turn_swing, bonus_from, bonus_full - bonus_from)
        self.turn_count += self.turn_done.float()
        self.since_turn = torch.where(self.turn_done, zeros, self.since_turn + self.step_dt)
        self.turn_swing = torch.where(crossed, psi.abs(), torch.maximum(self.turn_swing, psi.abs()))
        self.turn_side = torch.where(side != 0, side, self.turn_side)

    def _update_edge_switch(self, edge: torch.Tensor):
        """An edge change: the held (smoothed) edge flips sides after the raw edge went flat on the base."""
        held = torch.sign(self.edge_ema) * (self.edge_ema.abs() > HELD_EDGE)
        flipped = (held != 0) & (self.edge_side != 0) & (held != self.edge_side)
        self.base_touched |= edge.abs() < deg(self.cfg.switch_flat_deg)
        self.edge_switch = (flipped & self.base_touched).float()
        self.base_touched &= held == 0
        self.edge_side = torch.where(held != 0, held, self.edge_side)
        self.since_switch = torch.where(self.edge_switch > 0, torch.zeros_like(self.since_switch),
                                        self.since_switch + self.step_dt)

    # -- MDP ----------------------------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self.prev_actions = self.actions.clone()
        obs = super()._get_observations()["policy"]
        parts = [obs, self.actions, self._task_obs()] + ([self.synergy] if self.cfg.synergy else [])
        return {"policy": torch.cat(parts, dim=-1)}

    def _task_obs(self) -> torch.Tensor:
        psi, beta, yaw = self.path_heading(), self.slip_angle(), self.board_yaw()
        y = self.position()[:, 1]
        fade = self.guide_fade()
        guide_error = wrap(self.guide_heading() - psi)
        return torch.stack(
            [self.board_speed(), self.v_cmd, psi.sin(), psi.cos(), beta.sin(), beta.cos(), yaw.sin(), yaw.cos(),
             y / self.cfg.fence_half_width, self.turn_side, self.turn_swing, self.since_turn.clamp(max=10.0) / 5.0,
             self.slip_target(), fade * guide_error.sin(), fade * (1 - guide_error.cos()), torch.full_like(psi, fade),
             self.edge_command(), self.edge_command(0.3), self.edge_command(0.6)],
            dim=-1,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        too_low, time_out = super()._get_dones()
        lying = self.torso_up_cos() < math.cos(deg(self.cfg.lying_deg))
        sitting = self.pelvis_height() < SITTING_HEIGHT
        on_side = self.edge_angle().abs() > ON_SIDE_EDGE if self.snow_info else torch.zeros_like(lying)
        reverse = (self.slip_angle().abs() > deg(self.cfg.reverse_slip_deg)) & (self.board_speed() > REVERSE_MIN_SPEED)
        self.fall_why = torch.stack([too_low, lying, sitting, on_side, reverse], -1)
        fallen = self.fall_why.any(-1)
        x, y = self.position().unbind(-1)
        self.out_of_bounds = y.abs() > self.cfg.fence_half_width
        if self.cfg.run_length is not None:
            time_out = time_out | (x > self.cfg.run_length)
        return fallen | self.out_of_bounds, time_out

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        if cfg.edge_cmd and cfg.edge_cmd_mode == "heading" and not cfg.edge_cmd_hold:
            self._step_heading_cmd()
        if cfg.rsi_frac > 0:
            self._rsi_record()
        self._update_turns()

        reward = torch.zeros(self.num_envs, device=self.device)

        def add(name: str, value: torch.Tensor):
            nonlocal reward
            reward += value
            self._track(f"rew_{name}", value)

        terms = self._step_terms(self._edges_for_the_s())
        for name in STEP_TERMS:
            add(name, getattr(cfg, f"rew_{name}") * terms[name] * self.step_dt)
        add("fall", cfg.rew_fall * self.reset_terminated.float())
        if self.turns:
            self._guide_reward(add)
        if cfg.edge_cmd:
            self._edge_cmd_rewards(add)
        if self.turns:
            add("edge_switch", cfg.rew_edge_switch * self.edge_switch)
            add("turn", cfg.rew_turn * self.turn_credit)
        self._log_step()
        return reward

    def _step_terms(self, s_edges) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        speed, psi = self.board_speed(), self.path_heading()
        zeros = torch.zeros_like(speed)
        clean = gauss(self.slip_angle(), deg(cfg.sigma_slip_deg))
        terms = {
            "alive": (~self.reset_terminated).float(),
            "speed": gauss(speed - self.v_cmd, cfg.sigma_speed),
            "speed_cap": ramp(speed, cfg.speed_cap, 2.0),
            "slip": zeros if cfg.task == "carve" else gauss(self.slip_error(), deg(cfg.sigma_slip_deg)),
            "heading": zeros if self.turns else gauss(psi, deg(cfg.sigma_heading_deg)),
            "action_rate": (self.actions - self.prev_actions).square().sum(-1),
        }
        if self.turns:
            terms |= self._turn_terms(speed, psi, clean, s_edges)
        return {name: terms.get(name, zeros) for name in STEP_TERMS}

    def _edges_for_the_s(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(edge that turns the path the way the S needs, edge that turns it the wrong way), both ≥ 0.

        The S swings away from the fall line until min_turn, then back through it; the heading rate has the
        opposite sign to the edge angle. With an edge command, the command picks the edge instead.
        """
        edge_abs = self.edge_ema.abs()
        zeros = torch.zeros_like(edge_abs)
        wanted = torch.where(self.turn_swing >= self.min_turn, -self.turn_side, self.turn_side)
        right_edge = -wanted * self.edge_ema
        self._track("edge_dir_deg", torch.where(wanted != 0, right_edge, zeros).rad2deg())
        if self.cfg.edge_cmd:
            return edge_abs, zeros
        right = torch.where(wanted != 0, right_edge.clamp(min=0.0), edge_abs)
        wrong = torch.where(wanted != 0, (-right_edge).clamp(min=0.0), zeros)
        return right, wrong

    def _turn_terms(self, speed, psi, clean, s_edges) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        edge, wrong = s_edges
        edge_abs = self.edge_ema.abs()
        edged = ramp(edge, 0.0, deg(cfg.edge_target_deg))
        self._log_angulation()
        hip_tail, back_share = self.stance_balance()
        hand_heights = self.body_pos(self.hand_ids)[..., 2]
        incline = self.torso_incline()
        bow_target = deg(45.0) + deg(20.0) * ramp(edge, 0.0, deg(45.0))
        carving = cfg.task == "carve"
        return {
            "carve": (edged * self.path_turning() * clean * (0.5 + 0.5 * self.low()) * ramp(speed, 4.0, 4.0)
                      if carving else torch.zeros_like(speed)),
            "swing": (ramp(psi.abs(), 0.0, self.min_turn) * edged * self.path_turning()
                      if carving else torch.zeros_like(speed)),
            "low": self.low(),
            "facing": self.facing(),
            "hand": edged * gauss(hand_heights.min(-1).values, cfg.hand_scale),
            "bow": ramp(incline, 0.0, bow_target) * ramp(deg(80.0) - incline, 0.0, deg(10.0)),
            "edge_rate": ramp(self.edge_rate.abs(), 3.0, 2.0),
            "hip_tail": ramp(hip_tail, 0.0, 0.10),
            "balance": gauss(back_share - 0.5, 0.15),
            "uphill": ramp(psi.abs(), deg(90.0), deg(15.0)),
            "release": ramp(-self.edge_side * self.edge_ema_rate, 0.0, 1.0) * ramp(self.turn_swing, deg(30.0), deg(25.0)),
            "wrong_edge": ramp(wrong, 0.0, deg(30.0)),
            "deep_edge": ramp(edge_abs, deg(60.0), deg(15.0)),
            "edge_hold": ramp(self.since_switch, cfg.edge_hold_after_s, cfg.edge_hold_ramp_s),
            "traverse": ramp(self.since_turn, cfg.traverse_after_s, cfg.traverse_ramp_s),
            "height": gauss(self.pelvis_height() - (0.50 - 0.25 * ramp(edge, 0.0, deg(45.0))), cfg.height_sigma),
            "arm": ramp(hand_heights.max(-1).values - hand_heights.min(-1).values, 0.0, 0.5) * edged,
            "straight": gauss(psi, deg(cfg.straight_sigma_deg)),
            "hip": self.hip_fold(),
        }

    def _guide_reward(self, add):
        cfg = self.cfg
        psi = self.path_heading()
        heading_error = wrap(psi - self.guide_heading())
        x, y = self.position().unbind(-1)
        line_error = y - self.guide_y * (self.guide_k * x + self.guide_phase).sin()
        add("guide", self.guide_fade() * cfg.rew_guide * gauss(heading_error, deg(cfg.sigma_heading_deg)) * self.step_dt)
        self._track("err_guide_m", line_error.abs())
        self._track("err_guide_deg", heading_error.abs().rad2deg())
        self._adapt_guide(heading_error.abs().rad2deg())

    def _edge_cmd_rewards(self, add):
        cfg = self.cfg
        edge = self.edge_angle()
        command = self.edge_command()
        error = edge - command
        edged = ramp(edge.abs(), deg(10.0), deg(10.0))
        clean = gauss(self.slip_angle(), deg(cfg.sigma_slip_deg)) if cfg.edge_cmd_clean else 1.0
        add("edge_cmd", cfg.rew_edge_cmd * gauss(error, deg(cfg.edge_cmd_sigma_deg)) * clean * edged * self.step_dt)
        if cfg.rew_edge_side:
            on = (command.abs() > deg(10.0)) & (edge.abs() > deg(3.0))
            agree = torch.where(on, torch.sign(edge) * torch.sign(command), torch.zeros_like(edge))
            add("edge_side", cfg.rew_edge_side * agree * self.step_dt)
        self._track("err_edge_cmd_deg", error.abs().rad2deg())
        self._adapt_edge_cmd(error.abs().rad2deg())
        if cfg.rew_tilt:
            add("tilt", cfg.rew_tilt * ramp(self.torso_incline(), deg(cfg.tilt_free_deg), deg(20.0)) * self.step_dt)
        if cfg.rew_grab:
            grab, distance = self._grab_reward(edge)
            add("grab", cfg.rew_grab * grab * self.step_dt)
            self._track("grab_active", (distance > 0).float())
            self._track("grab_dist_x_active_m", distance)
        if cfg.rew_rise:
            near_flat = (command.abs() < deg(8.0)).float()
            add("rise", cfg.rew_rise * near_flat * ramp(self.pelvis_height(), 0.35, 0.2) * self.step_dt)
        if cfg.rew_pose_ref:
            legs, com = self.roll_leans()
            legs_ref = cfg.pose_ref_legs * command
            com_ref = cfg.pose_ref_com * legs_ref
            sigma = deg(cfg.pose_ref_sigma_deg)
            on_edge = (command.abs() > deg(10.0)).float()
            match = gauss(legs - legs_ref, sigma) * gauss(com - com_ref, sigma)
            add("pose_ref", cfg.rew_pose_ref * match * on_edge * self.step_dt)
            self._track("legs_lean_deg", legs.rad2deg() * torch.sign(command))
            self._track("com_lean_deg", com.rad2deg() * torch.sign(command))

    def _grab_reward(self, edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inside hand to its grab point: heel turn, right hand → toe edge at the front foot; toe turn, left hand →
        heel edge at the back foot. Returns (reward, distance; 0 when no turn)."""
        board_quat, board_pos = self.body_quat(self.board_id), self.body_pos(self.board_id)

        def in_board(body_id):
            return quat_apply_inverse(board_quat, self.body_pos(body_id) - board_pos)

        toes = quat_apply_inverse(board_quat, quat_apply(self.body_quat(self.front_foot_id), self.axis(0)))
        toe_side = torch.where(toes[:, 1].abs() > 1e-3, torch.sign(toes[:, 1]), torch.ones_like(toes[:, 1]))
        lift = torch.zeros_like(toe_side) + 0.05
        heel_turn_target = torch.stack([in_board(self.front_foot_id)[:, 0], toe_side * self.board_half_width, lift], -1)
        toe_turn_target = torch.stack([in_board(self.back_foot_id)[:, 0], -toe_side * self.board_half_width, lift], -1)
        d_heel = (in_board(self.right_hand_id) - heel_turn_target).norm(dim=-1)
        d_toe = (in_board(self.left_hand_id) - toe_turn_target).norm(dim=-1)

        command = self.edge_command()
        heel_turn = (command < -deg(10.0)) & (edge < -deg(10.0))
        toe_turn = (command > deg(10.0)) & (edge > deg(10.0))
        distance = torch.where(heel_turn, d_heel, torch.where(toe_turn, d_toe, torch.zeros_like(d_heel)))
        if self.cfg.grab_linear_m > 0:
            reward = (1.0 - distance / self.cfg.grab_linear_m).clamp(0.0, 1.0)
        else:
            reward = gauss(distance, self.cfg.grab_sigma_m)
        self._track("grab_heel_active", heel_turn.float())
        self._track("grab_heel_dist_x_active_m", d_heel * heel_turn)
        self._track("grab_toe_active", toe_turn.float())
        self._track("grab_toe_dist_x_active_m", d_toe * toe_turn)
        return reward * (heel_turn | toe_turn).float(), distance

    def _log_step(self):
        speed = self.board_speed()
        self._track("err_slip_deg", self.slip_error().abs().rad2deg())
        self._track("err_speed", (speed - self.v_cmd).abs())
        self._track("speed", speed)
        self._track("heading_abs_deg", self.path_heading().abs().rad2deg())
        self._track("edge_abs_deg", self.edge_angle().abs().rad2deg())

    def _track(self, name: str, value: torch.Tensor):
        self.episode_sums.setdefault(name, torch.zeros_like(value))
        self.episode_sums[name] += value

    # -- reference state initialisation (RSI) --------------------------------------------------------------------------

    def _init_rsi_buffer(self):
        size, joints = self.cfg.rsi_buffer, self.stance_pos.shape[1]
        self.rsi_root = torch.zeros(size, 13, device=self.device)
        self.rsi_qpos = torch.zeros(size, joints, device=self.device)
        self.rsi_qvel = torch.zeros_like(self.rsi_qpos)
        self.rsi_slip = torch.zeros(size, *self.slip_z.shape[1:], device=self.device)
        self.rsi_ema = torch.zeros(size, device=self.device)

    def _rsi_record(self):
        cfg = self.cfg
        on_toe_edge = self.edge_angle() > deg(cfg.rsi_toe_deg)
        settled = self.episode_length_buf * self.step_dt > cfg.rsi_min_s
        ids = (settled & on_toe_edge & ~self.reset_buf).nonzero().squeeze(-1)
        if len(ids) == 0:
            return
        ids = ids[torch.randperm(len(ids), device=self.device)[:RSI_PER_STEP]]
        slots = (torch.arange(len(ids), device=self.device) + self.rsi_next) % cfg.rsi_buffer
        data = self.robot.data
        pose = data.root_link_pose_w.torch[ids].clone()
        pose[:, :3] = self.to_slope(pose[:, :3]) - self.scene.env_origins[ids]
        self.rsi_root[slots] = torch.cat([pose, data.root_com_vel_w.torch[ids]], dim=-1)
        self.rsi_qpos[slots] = data.joint_pos.torch[ids]
        self.rsi_qvel[slots] = data.joint_vel.torch[ids]
        self.rsi_slip[slots] = self.slip_z[ids]
        self.rsi_ema[slots] = self.edge_ema[ids]
        self.rsi_next = (self.rsi_next + len(ids)) % cfg.rsi_buffer
        self.rsi_count = min(cfg.rsi_buffer, self.rsi_count + len(ids))

    def _rsi_restore(self, env_ids: torch.Tensor):
        if self.rsi_count < RSI_MIN_SAMPLES:
            return
        picked = env_ids[coin(len(env_ids), self.device, self.cfg.rsi_frac)]
        if len(picked) == 0:
            return
        source = torch.randint(0, self.rsi_count, (len(picked),), device=self.device)
        root = self.rsi_root[source].clone()
        root[:, :3] = self.to_world(root[:, :3] + self.scene.env_origins[picked])
        self.robot.write_root_link_pose_to_sim_index(root_pose=root[:, :7], env_ids=picked)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=picked)
        self.robot.write_joint_position_to_sim_index(position=self.rsi_qpos[source], env_ids=picked)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.rsi_qvel[source], env_ids=picked)
        self.slip_z[picked] = self.rsi_slip[source]
        self.edge_ema[picked] = self.rsi_ema[source]
        self.cmd_phase[picked] = math.pi / 2  # toe peak
        if self.cfg.edge_cmd_mode == "heading":
            self.cmd_side[picked], self.cmd_from[picked], self.cmd_since[picked] = 1.0, 0.0, self.cfg.edge_cmd_ramp_s
        self.rsi_started[picked] = 1.0

    # -- reset ----------------------------------------------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._log_episodes(env_ids)
        self.last_turns[env_ids] = self.turn_count[env_ids]
        self.v_cmd[env_ids] = uniform(self.cfg.speed_cmd, len(env_ids), self.device)
        for buffer in (self.turn_side, self.turn_swing, self.turn_count, self.since_turn, self.edge_side,
                       self.since_switch, self.base_touched):
            buffer[env_ids] = 0.0
        self._sample_guide(env_ids)
        self._sample_edge_cmd(env_ids)
        self.rsi_started[env_ids] = 0.0
        super()._reset_idx(env_ids)
        if self.cfg.rsi_frac > 0:
            self._rsi_restore(env_ids)
        index_fill_(self.actions, env_ids, 0.0)
        if self.cfg.synergy:
            self.synergy = self.synergy.clone()
            index_fill_(self.synergy, env_ids, 0.0)
        index_fill_(self.prev_actions, env_ids, 0.0)

    def _log_episodes(self, env_ids: torch.Tensor):
        """TensorBoard: rewards per second, other tracked values per step, turns per episode."""
        if not self.episode_sums:
            return
        steps = self.episode_length_buf[env_ids].clamp_min(1).float()
        seconds = steps * self.step_dt
        terminated = self.reset_terminated[env_ids]
        fence = self.out_of_bounds[env_ids]
        log = {}
        for name, total in self.episode_sums.items():
            if name.startswith("rew_"):
                log[f"Episode_Reward/{name[4:]}"] = total[env_ids].mean() / self.max_episode_length_s
            else:
                log[f"Episode_Snow/{name}"] = (total[env_ids] / steps).mean()
            index_fill_(total, env_ids, 0.0)
        turns = self.turn_count[env_ids]
        log["Episode_Snow/turns"] = turns.mean()
        log["Episode_Snow/turns_per_10s"] = (turns / seconds * 10).mean()
        log["Episode_Termination/fell"] = (terminated & ~fence).float().mean()
        log["Episode_Termination/fence"] = fence.float().mean()
        log["Episode_Snow/seconds"] = seconds.mean()
        rsi = self.rsi_started[env_ids] > 0
        log["RSI/share"] = rsi.float().mean()
        if rsi.any():
            log["RSI/turns_per_10s"] = (turns[rsi] / seconds[rsi] * 10).mean()
            log["RSI/fell"] = terminated[rsi].float().mean()
        for i, reason in enumerate(FALL_REASONS):
            log[f"Episode_Termination/why_{reason}"] = self.fall_why[env_ids, i].float().mean()
        log["Guide/level"] = torch.tensor(self.guide_fade())
        log["EdgeCmd/level"] = torch.tensor(self.cmd_level)
        log["Guide/err_ema_deg"] = torch.tensor(self.guide_err_ema)
        self.extras["log"] = log

    def _start_state(self, env_ids: torch.Tensor):
        """Side slip: at rest across the hill on the heel edge. Turns: riding at speed, along the guide if it's on."""
        n = len(env_ids)
        jitter = deg(5.0) * (2 * torch.rand(n, device=self.device) - 1)
        zeros = torch.zeros(n, device=self.device)
        if self.cfg.task == "sideslip":
            # centre of mass right above the heel edge: from flat the rider rolls onto the toe edge and falls
            lean = math.atan2(self.board_half_width, self.pelvis_in_board_pos[2].item())
            yaw = math.pi / 2 + jitter
            edge = torch.full_like(yaw, -(deg(self.slope_deg) + lean))
            return yaw, edge, torch.zeros(n, 3, device=self.device), zeros

        if self.guide_fade() > 0:
            heading = torch.atan(self.guide_y[env_ids] * self.guide_k[env_ids] * self.guide_phase[env_ids].cos())
        else:
            heading = deg(self.cfg.start_heading_deg) * (2 * torch.rand(n, device=self.device) - 1)
        direction = torch.stack([heading.cos(), heading.sin(), zeros], dim=-1)
        speed = uniform(self.cfg.start_speed_turns, n, self.device)
        edge = zeros
        if self.cfg.carve_start:
            side = torch.where(coin(n, self.device, self.cfg.carve_start_toe_frac), 1.0, -1.0)
            edge = side * self.balanced_edge(speed)
            self.cmd_hold[env_ids] = edge
            if not self.cfg.edge_cmd_hold:
                self.cmd_phase[env_ids] = side * math.pi / 2
        return heading + jitter, edge, speed.unsqueeze(-1) * direction, zeros
