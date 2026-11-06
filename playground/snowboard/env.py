"""Carve on the fenced piste by tracking a commanded edge angle, heel ↔ toe.

The env commands a signed edge angle (heel < 0, toe > 0), on a timer or once the turn has come round; the board's
sidecut turns whatever edge the rider holds, so a command that alternates draws an S. Crossing a fence ends the
episode. Reward weights: RewardsCfg, set per stage from playground/snowboard/rewards/*.yaml. See docs/RL_TRAINING.md.

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

TASK_OBS = 19
SYNERGY_OBS = 2
BALANCE_RADIUS = 7.6  # m, sidecut radius used to pick the edge that balances a speed
TURN_MIN_SPEED = 5.0  # m/s
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


def uniform(lo_hi, n: int, device) -> torch.Tensor:
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(n, device=device)


def coin(n: int, device, p: float = 0.5) -> torch.Tensor:
    return torch.rand(n, device=device) < p


@configclass
class RewardsCfg:
    """Weights (per second, fall once) and term shapes; each stage sets them from rewards/<stage>.yaml."""

    edge_cmd = 30.0  # track the commanded edge
    edge_cmd_sigma_deg = 30.0
    edge_side = 20.0  # on the commanded side
    pose_ref = 10.0  # legs and COM lean of the double pendulum (scripts/pendulum_rl.py)
    pose_ref_legs = 1.15
    pose_ref_com = 0.77
    pose_ref_sigma_deg = 15.0
    rise = 30.0  # stand tall while the command crosses flat
    tilt = -10.0  # torso bowed past tilt_free_deg
    tilt_free_deg = 50.0
    grab = 10.0  # inside hand toward its grab point on the board
    grab_sigma_m = 0.6
    grab_linear_m = 1.2  # > 0: linear falloff instead of the Gaussian
    fall = -50.0  # once, a fall or a fence


@configclass
class SnowboardEnvCfg(CarveEnvCfg):
    episode_length_s = 12.0
    observation_space = 87 + 28 + TASK_OBS  # recomputed from the DoF
    action_scale = 0.25
    mountain = True
    fall_height = 0.15

    rewards: RewardsCfg = RewardsCfg()

    # rider: the same in every stage (docs/HUMANOID_MODEL.md §6); stance and bindings come from config.yaml
    stance_tag = "squat4"
    rider_to_height = True
    pelvis_mass_frac = 0.2
    knee_rotation = True
    passive_joints = ("knee:2",)
    ankle_range_deg = {"x": 45, "y": 80, "z": 55}
    torso_twist_deg = 90.0
    arm_stiffness_scale = 0.3
    arm_action_scale = 0.5
    stiff_boots = True
    boot_range_deg = {"y": [-5, 15], "x": [-5, 5], "z": [-5, 5]}
    synergy = True
    synergy_toe_rad = {"left_knee": 0.4, "right_knee": 0.4, "left_hip_y": -0.3, "right_hip_y": -0.3}
    synergy_smooth = 0.35
    joint_smooth = 0.35

    speed_cmd = (6.0, 8.0)  # m/s, observed only (saved policies were trained with it)
    start_heading_deg = 20.0
    start_speed_turns = (4.0, 6.0)  # m/s
    carve_start: bool = False  # start each episode balanced on an edge
    carve_start_toe_frac = 0.5
    fence_half_width = FENCE_HALF_WIDTH
    run_length: float | None = None  # m, ends the episode as a time-out
    min_turn_deg = 55.0  # swing that makes a turn count when the path crosses back over the fall line
    lying_deg = 95.0  # torso tilt; a deep toe carve bows it 55-70°
    reverse_slip_deg = 100.0  # tail first at speed counts as a fall

    # edge command: a signed edge angle to track, heel → base → toe
    edge_cmd_hold: bool = False  # hold the start edge for the whole episode (scripts/switch_test.py)
    edge_cmd_amp_deg = (20.0, 50.0)  # curriculum level 0 → 1
    edge_cmd_period_s = (4.6, 4.0)  # level 0 → 1, ±15 % per episode
    edge_cmd_level_init = 0.0
    edge_cmd_level_fixed = False
    edge_cmd_level_up_deg = 12.0
    edge_cmd_level_steps = 2000
    edge_cmd_mode = "timer"  # "timer": periodic. "heading": switch once the turn reaches edge_cmd_heading_deg
    edge_cmd_sharpness = 1.8
    edge_cmd_heading_deg = 50.0
    edge_cmd_min_s = 1.0
    edge_cmd_max_s = 3.0
    edge_cmd_ramp_s = 0.6
    edge_cmd_center_deg_per_m = 0.0  # heading mode: shift the switch heading to steer back to the piste centre
    edge_cmd_center_max_deg = 25.0

    # reset from recorded mid-episode toe carves, so toe → heel is practised where riders get stuck
    rsi_frac = 0.0
    rsi_buffer = 8192
    rsi_min_s = 3.0
    rsi_toe_deg = 12.0


class SnowboardEnv(CarveEnv):
    cfg: SnowboardEnvCfg

    def __init__(self, cfg: SnowboardEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = base_obs_size(cfg) + num_dof(cfg) + TASK_OBS + (SYNERGY_OBS if cfg.synergy else 0)
        self._init_task_state(cfg.scene.num_envs, cfg.sim.device, cfg)  # super().__init__ resets every env
        super().__init__(cfg, render_mode, **kwargs)
        self.min_turn = deg(cfg.min_turn_deg)
        self._find_task_bodies()

    def _init_task_state(self, n: int, device, cfg: SnowboardEnvCfg):
        def zeros(dtype=torch.float32):
            return torch.zeros(n, dtype=dtype, device=device)

        self.v_cmd = zeros()
        self.turn_side = zeros()
        self.turn_swing = zeros()
        self.turn_count = zeros()
        self.since_turn = zeros()
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
        self.out_of_bounds = zeros(torch.bool)
        self.fall_why = torch.zeros(n, len(FALL_REASONS), dtype=torch.bool, device=device)
        self.episode_sums: dict[str, torch.Tensor] = {}

    def _find_task_bodies(self):
        self.right_hand_id, self.left_hand_id, self.back_foot_id, self.front_foot_id = self.key_ids
        self.rider_ids = [i for i in range(len(self.robot.body_names)) if i not in self.board_ids]
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

    def balanced_edge(self, speed: torch.Tensor) -> torch.Tensor:
        """|Edge| whose sidecut arc balances this speed."""
        g_normal = GRAVITY * math.cos(deg(self.slope_deg))
        return torch.asin((speed.square() / (g_normal * BALANCE_RADIUS)).clamp(max=0.9))

    # -- rider posture ------------------------------------------------------------------------------------------------

    def pelvis_height(self) -> torch.Tensor:
        """Above the board along the snow normal (stance 0.80 m)."""
        return self.body_pos(self.pelvis_id)[:, 2] - self.body_pos(self.board_id)[:, 2]

    def torso_up_cos(self) -> torch.Tensor:
        torso_up = quat_apply(self.body_quat(self.torso_id), self.up.expand(self.num_envs, 3))
        return (torso_up * self.up).sum(-1)

    def torso_incline(self) -> torch.Tensor:
        return self.torso_up_cos().clamp(-1.0, 1.0).acos()

    def rider_com(self) -> torch.Tensor:
        mass = self.robot.data.body_mass.torch[0, self.rider_ids]  # scaled to the rider only once the sim has stepped
        return (self.body_pos(self.rider_ids) * mass.view(1, -1, 1)).sum(1) / mass.sum()

    def roll_leans(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Leans of board → pelvis and board → rider COM in the board's roll plane, + toward the engaged θ > 0 edge."""
        normal = self.up.expand(self.num_envs, 3)
        _, across, _ = board_frame(self.body_quat(self.board_id), normal)
        toe_side = -across
        board, pelvis = self.body_pos(self.board_id), self.body_pos(self.pelvis_id)

        def lean(d):
            return torch.atan2((d * toe_side).sum(-1), (d * normal).sum(-1))

        return lean(pelvis - board), lean(self.rider_com() - board)

    # -- edge command ------------------------------------------------------------------------------------------------

    def edge_command(self, ahead_s: float = 0.0) -> torch.Tensor:
        """Commanded signed edge angle `ahead_s` from now (θ < 0 heel, > 0 toe)."""
        cfg = self.cfg
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
        alive = ~self.reset_terminated
        if alive.any():
            self.cmd_err_ema = 0.99 * self.cmd_err_ema + 0.01 * error_deg[alive].mean().item()
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

    # -- turn tracking -------------------------------------------------------------------------------------------------

    def _update_turns(self):
        """A turn: the path crosses the fall line after swinging at least min_turn_deg to one side."""
        psi = self.path_heading()
        side = torch.sign(psi) * (psi.abs() > 0.02)
        crossed = (side != 0) & (self.turn_side != 0) & (side != self.turn_side) & (self.board_speed() > TURN_MIN_SPEED)
        turn_done = crossed & (self.turn_swing >= self.min_turn)
        self.turn_count += turn_done.float()
        self.since_turn = torch.where(turn_done, torch.zeros_like(psi), self.since_turn + self.step_dt)
        self.turn_swing = torch.where(crossed, psi.abs(), torch.maximum(self.turn_swing, psi.abs()))
        self.turn_side = torch.where(side != 0, side, self.turn_side)

    # -- MDP ----------------------------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"]
        parts = [obs, self.actions, self._task_obs()] + ([self.synergy] if self.cfg.synergy else [])
        return {"policy": torch.cat(parts, dim=-1)}

    def _task_obs(self) -> torch.Tensor:
        """The layout saved policies were trained on; the four zero slots held a slip target and a guide line."""
        psi, beta, yaw = self.path_heading(), self.slip_angle(), self.board_yaw()
        y = self.position()[:, 1]
        zeros = torch.zeros_like(psi)
        return torch.stack(
            [self.board_speed(), self.v_cmd, psi.sin(), psi.cos(), beta.sin(), beta.cos(), yaw.sin(), yaw.cos(),
             y / self.cfg.fence_half_width, self.turn_side, self.turn_swing, self.since_turn.clamp(max=10.0) / 5.0,
             zeros, zeros, zeros, zeros,
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
        if cfg.edge_cmd_mode == "heading" and not cfg.edge_cmd_hold:
            self._step_heading_cmd()
        if cfg.rsi_frac > 0:
            self._rsi_record()
        self._update_turns()

        reward = torch.zeros(self.num_envs, device=self.device)

        def add(name: str, value: torch.Tensor):
            nonlocal reward
            reward += value
            self._track(f"rew_{name}", value)

        terms = self._reward_terms()
        for name, value in terms.items():
            add(name, getattr(cfg.rewards, name) * value * self.step_dt)
        add("fall", cfg.rewards.fall * self.reset_terminated.float())
        self._log_step()
        return reward

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        """Per-second terms, each in [-1, 1]; docs/RL_TRAINING.md §4."""
        w = self.cfg.rewards
        edge, command = self.edge_angle(), self.edge_command()
        error = edge - command
        self._track("err_edge_cmd_deg", error.abs().rad2deg())
        self._adapt_edge_cmd(error.abs().rad2deg())

        edged = ramp(edge.abs(), deg(10.0), deg(10.0))
        side_on = (command.abs() > deg(10.0)) & (edge.abs() > deg(3.0))
        near_flat = (command.abs() < deg(8.0)).float()
        terms = {
            "edge_cmd": gauss(error, deg(w.edge_cmd_sigma_deg)) * edged,
            "edge_side": torch.where(side_on, torch.sign(edge) * torch.sign(command), torch.zeros_like(edge)),
            "rise": near_flat * ramp(self.pelvis_height(), 0.35, 0.2),
            "tilt": ramp(self.torso_incline(), deg(w.tilt_free_deg), deg(20.0)),
        }
        if w.pose_ref:
            terms["pose_ref"] = self._pose_ref(command)
        if w.grab:
            terms["grab"] = self._grab_reward(edge, command)
        return terms

    def _pose_ref(self, command: torch.Tensor) -> torch.Tensor:
        """Legs lean past the commanded edge, the COM less (angulation), as in the double pendulum."""
        w = self.cfg.rewards
        legs, com = self.roll_leans()
        legs_ref = w.pose_ref_legs * command
        com_ref = w.pose_ref_com * legs_ref
        sigma = deg(w.pose_ref_sigma_deg)
        on_edge = (command.abs() > deg(10.0)).float()
        self._track("legs_lean_deg", legs.rad2deg() * torch.sign(command))
        self._track("com_lean_deg", com.rad2deg() * torch.sign(command))
        return gauss(legs - legs_ref, sigma) * gauss(com - com_ref, sigma) * on_edge

    def _grab_reward(self, edge: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Inside hand to its grab point: heel turn, right hand → toe edge at the front foot; toe turn, left hand →
        heel edge at the back foot. 0 when no turn."""
        w = self.cfg.rewards
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

        heel_turn = (command < -deg(10.0)) & (edge < -deg(10.0))
        toe_turn = (command > deg(10.0)) & (edge > deg(10.0))
        distance = torch.where(heel_turn, d_heel, torch.where(toe_turn, d_toe, torch.zeros_like(d_heel)))
        if w.grab_linear_m > 0:
            reward = (1.0 - distance / w.grab_linear_m).clamp(0.0, 1.0)
        else:
            reward = gauss(distance, w.grab_sigma_m)
        self._track("grab_active", (distance > 0).float())
        self._track("grab_dist_x_active_m", distance)
        self._track("grab_heel_active", heel_turn.float())
        self._track("grab_heel_dist_x_active_m", d_heel * heel_turn)
        self._track("grab_toe_active", toe_turn.float())
        self._track("grab_toe_dist_x_active_m", d_toe * toe_turn)
        return reward * (heel_turn | toe_turn).float()

    def _log_step(self):
        self._track("err_slip_deg", self.slip_angle().abs().rad2deg())
        self._track("speed", self.board_speed())
        self._track("heading_abs_deg", self.path_heading().abs().rad2deg())
        self._track("edge_abs_deg", self.edge_angle().abs().rad2deg())
        self._track("pelvis_height_m", self.pelvis_height())

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
        self.cmd_phase[picked] = math.pi / 2  # toe peak
        if self.cfg.edge_cmd_mode == "heading":
            self.cmd_side[picked], self.cmd_from[picked], self.cmd_since[picked] = 1.0, 0.0, self.cfg.edge_cmd_ramp_s
        self.rsi_started[picked] = 1.0

    # -- reset ----------------------------------------------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._log_episodes(env_ids)
        self.v_cmd[env_ids] = uniform(self.cfg.speed_cmd, len(env_ids), self.device)
        for buffer in (self.turn_side, self.turn_swing, self.turn_count, self.since_turn):
            buffer[env_ids] = 0.0
        self._sample_edge_cmd(env_ids)
        self.rsi_started[env_ids] = 0.0
        super()._reset_idx(env_ids)
        if self.cfg.rsi_frac > 0:
            self._rsi_restore(env_ids)
        index_fill_(self.actions, env_ids, 0.0)
        if self.cfg.synergy:
            self.synergy = self.synergy.clone()
            index_fill_(self.synergy, env_ids, 0.0)

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
        log["EdgeCmd/level"] = torch.tensor(self.cmd_level)
        self.extras["log"] = log

    def _start_state(self, env_ids: torch.Tensor):
        """Riding at speed near the fall line; with carve_start, balanced on an edge with the command at its peak."""
        n = len(env_ids)
        jitter = deg(5.0) * (2 * torch.rand(n, device=self.device) - 1)
        zeros = torch.zeros(n, device=self.device)
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
