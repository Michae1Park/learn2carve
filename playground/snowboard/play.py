"""Ride a trained snowboard policy, draw its track on the snow and save a top-down picture of it.

    scripts/py playground/snowboard/play.py logs/playground/snowboard/<run>
    scripts/py playground/snowboard/play.py <run> --livestream 2 --seconds 120
    scripts/py playground/snowboard/play.py <run> --livestream 2 --follow    # camera tracks the rider
    scripts/py playground/snowboard/play.py --no_policy --task sideslip      # stance held, zero actions

Play is a series of short runs, each starting the same way (piste centre, phase 0, mid-range speed and turn length)
and ending --run_length metres down (default two S's; 25 m for the side slip). The rider then returns to the top.
--seconds is the total play time. One line is printed per run (env 0), and the longest run's track is saved to
<run>/trail.png.

The camera starts low on the snow at the bottom of the run, looking up at the start, then it is yours (orbit, pan,
zoom). --cam_x moves it along the run; --follow tracks the rider instead.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so this runs with plain python too

from playground.quadruped.common import ROOT, add_suppress_arg, load_policy, make_env, parse_overrides

from isaaclab.app import add_launcher_args, launch_simulation

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, nargs="?", help="run directory from train.py")
parser.add_argument("--checkpoint", default="best_agent.pt", help="file in <run>/checkpoints/")
parser.add_argument("--no_policy", action="store_true", help="hold the stance (zero actions) instead of a policy")
parser.add_argument("--task", choices=["sideslip", "skid", "carve"], default=None, help="default: the run's task")
parser.add_argument("--seconds", type=float, default=60.0, help="total play time")
parser.add_argument("--run_length", type=float, default=None, help="one run [m]; default: two S's (2 × turn length)")
parser.add_argument("--cam_x", type=float, default=None,
                    help="camera distance down the run from the start [m]; default: just past where a run ends")
parser.add_argument("--focal", type=float, default=35.0, help="camera focal length [mm] (Kit's default is ~18)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--follow", action="store_true", help="camera tracks the rider (default: free camera)")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="env cfg overrides on top of the run's")
add_suppress_arg(parser)
add_launcher_args(parser)
args = parser.parse_args()
if not args.no_policy and args.run is None:
    parser.error("give a run directory, or --no_policy")

import torch
import yaml

import isaaclab.sim as sim_utils


class RunStats:
    """Averages over one run of env 0, printed as one line."""

    HEADER = "run  end    seconds  speed  |edge|°  heading_err°  slip_err°  line_err_m  max|y|_m  turns"

    def __init__(self):
        self.steps = 0
        self.sums = dict.fromkeys(("speed", "edge", "heading_err", "slip_err", "line_err"), 0.0)
        self.max_y = 0.0
        self.turns = 0  # each time the path crosses the fall line
        self.side = 0  # which side of the fall line the path points to: -1, 0 (unknown yet), 1

    def add(self, env):
        t = env.tracking()
        heading = env.path_heading()[0].item()
        side = 1 if heading > 0.05 else (-1 if heading < -0.05 else 0)
        if side and self.side and side != self.side:
            self.turns += 1
        self.side = side or self.side

        self.steps += 1
        self.sums["speed"] += env.board_speed()[0].item()
        self.sums["edge"] += abs(math.degrees(env.snow_info["edge_angle"][0].item()))
        self.sums["heading_err"] += abs(math.degrees(t.e_heading[0].item()))
        self.sums["slip_err"] += abs(math.degrees(t.e_slip[0].item()))
        self.sums["line_err"] += abs(t.e_line[0].item())
        self.max_y = max(self.max_y, abs(env.position()[0, 1].item()))

    def line(self, number: int, end: str, step_dt: float) -> str:
        mean = {k: v / max(self.steps, 1) for k, v in self.sums.items()}
        return (f"{number:3d}  {end:5s}  {self.steps * step_dt:7.1f}  {mean['speed']:5.1f}  {mean['edge']:6.1f}  "
                f"{mean['heading_err']:12.1f}  {mean['slip_err']:9.1f}  {mean['line_err']:10.2f}  "
                f"{self.max_y:8.1f}  {self.turns:5d}")


def main():
    run = yaml.safe_load((args.run / "run.yaml").read_text()) if args.run else {"task": "sideslip", "overrides": {}}
    task = args.task or run["task"]
    out = args.run / "trail.png" if args.run else ROOT / f"logs/playground/snowboard/no_policy_{task}_trail.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from envs import config
        from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg
        from playground.snowboard.trail import SnowTrail

        overrides, wavelength, run_length = play_overrides(run["overrides"], SnowboardEnvCfg(task=task))
        raw_env = make_env(SnowboardEnv, SnowboardEnvCfg(task=task), overrides, args.num_envs, sim_cfg,
                           args.suppress_warnings)
        if args.no_policy:
            env, agent = raw_env, None
        else:
            env, agent = load_policy(raw_env, run, args.run, args.checkpoint, cfg_dir=HERE)
        if not args.follow:
            set_focal_length(raw_env.cfg.viewer.cam_prim_path, args.focal)
        board = config.load()["board"]
        trail = SnowTrail(board["length"], board["waist_width"], raw_env.q_slope)

        print(f"\n[play] task={task}  {'no policy' if args.no_policy else args.run.name}  run {run_length:.0f} m, "
              f"turn length {wavelength:.0f} m, fence ±{raw_env.cfg.fence_half_width:.0f} m")
        print(f"[play] {RunStats.HEADER}")
        longest_secs, longest_marks = ride(env, raw_env, agent, trail)

        trail.marks = longest_marks
        swing, phase = raw_env.swing[0].item(), raw_env.phase[0].item()
        trail.save_png(out, f"{task}: {longest_secs:.1f} s, turn length {wavelength:.0f} m",
                       line=lambda x: swing * math.sin(2 * math.pi * x / wavelength + phase),
                       fence=raw_env.cfg.fence_half_width, x_end=run_length)
        print(f"[play] track of the longest episode ({longest_secs:.1f} s) -> {out}")
        env.close()


def play_overrides(run_overrides: dict, defaults) -> tuple[dict, float, float]:
    """Env overrides for short, repeatable runs. Returns (overrides, turn length, run length).

    Training's cfg is the base; --set wins over everything. Unless set, every run starts the same way: mid-range
    speed and turn length, on the piste centre (phase 0) swinging out.
    """
    user = parse_overrides(args.set)
    overrides = run_overrides | user

    def middle(name: str) -> float:
        lo, hi = overrides.get(name, getattr(defaults, name))
        return (lo + hi) / 2

    if "speed_cmd" not in user:
        overrides["speed_cmd"] = (middle("speed_cmd"),) * 2
    wavelength = middle("turn_length")
    if "turn_length" not in user:
        overrides["turn_length"] = (wavelength,) * 2
    if "fixed_phase" not in user:
        overrides["fixed_phase"] = 0.0

    run_length = args.run_length or (25.0 if defaults.task == "sideslip" else 2 * wavelength)
    overrides["run_length"] = run_length
    overrides["episode_length_s"] = 60.0  # runs end by distance; this only catches a rider who stops
    if not args.follow:
        overrides["viewer"] = free_view_cfg(overrides.get("slope_deg"), args.cam_x or run_length + 10.0)
    return overrides, wavelength, run_length


def ride(env, raw_env, agent, trail) -> tuple[float, list]:
    """Play for --seconds, one printed line per run. Returns the longest run's (seconds, trail marks)."""
    steps = int(args.seconds / raw_env.step_dt)
    obs, _ = env.reset()
    stats, runs = RunStats(), 0
    longest = (-1.0, [])
    for t in range(steps):
        with torch.inference_mode():
            if agent is None:
                action = torch.zeros(raw_env.num_envs, raw_env.cfg.action_space, device=raw_env.device)
            else:
                action, outputs = agent.act(obs, None, timestep=t, timesteps=steps)
                action = outputs.get("mean_actions", action)
            obs, _, terminated, truncated, _ = env.step(action)
        terminated, truncated = bool(terminated.view(-1)[0]), bool(truncated.view(-1)[0])
        done = terminated or truncated

        if not done:  # on a done step the env has already reset to the start
            stats.add(raw_env)
            trail.add(raw_env.body_pos(raw_env.board_id)[0], raw_env.slip_angle()[0].item())
        if done or t == steps - 1:
            runs += 1
            end = "fence" if raw_env.out_of_bounds[0] else "fell" if terminated else "done" if truncated else "-"
            print(f"[play] {stats.line(runs, end, raw_env.step_dt)}")
            seconds = stats.steps * raw_env.step_dt
            if seconds > longest[0]:
                longest = (seconds, list(trail.marks))
            trail.marks.clear()
            trail.reset()
            stats = RunStats()
    return longest


def free_view_cfg(slope_deg: float | None, cam_x: float):
    """Camera placed once, low on the snow cam_x down the run, looking back up at env 0's start. Kit leaves it alone
    after that (origin_type "env"), so the mouse controls it. "asset_root" (--follow) re-aims it every frame."""
    from isaaclab.envs import ViewerCfg

    from envs import config
    from envs.mountain import slope_to_world

    a = slope_deg if slope_deg is not None else config.load()["slope"]["angle_deg"]
    # 0.6 m up: on a plane the horizon sits at eye height, so the 1.6 m rider rises above it into the sky
    return ViewerCfg(eye=slope_to_world(a, cam_x, 3.0, 0.6), lookat=slope_to_world(a, 0.0, 0.0, 1.0), origin_type="env")


def set_focal_length(cam_prim_path: str, focal_mm: float):
    """Longer lens than Kit's ~18 mm default: the rider fills more of the frame from the bottom of the run."""
    from isaaclab.sim.utils.stage import get_current_stage

    prim = get_current_stage().GetPrimAtPath(cam_prim_path)
    if prim.IsValid():  # only exists with a viewer (e.g. --livestream)
        prim.GetAttribute("focalLength").Set(focal_mm)


if __name__ == "__main__":
    main()
