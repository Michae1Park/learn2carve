"""Ride a trained snowboard policy, draw its track on the snow and save a top-down picture of it.

    scripts/py playground/snowboard/play.py logs/playground/snowboard/<run>
    scripts/py playground/snowboard/play.py <run> --livestream 2 --seconds 120
    scripts/py playground/snowboard/play.py --no_policy --task sideslip      # stance held, zero actions
    scripts/py playground/snowboard/play.py <run> --livestream 2 --follow    # camera tracks the rider

Short evaluation runs: each run ends --run_length metres down the piste (default two full S's = 2 × turn length;
25 m for the side slip), always starting the same way (centre of the piste, same phase, mid-range speed and turn
length), then the rider is put back at the top for the next run. Play lasts --seconds in total.

Camera: low on the snow at the bottom of the run, looking up at the start through a longer lens, so the rider
carves toward you (like a filmed carving clip). It then stays put: orbit, pan and zoom freely. --cam_x moves it
up or down the run, --follow tracks the rider instead.

Prints one line per run for env 0: how it ended (done / fell / fence), seconds, speed, mean heading / slip / line
errors, widest swing |y| and how many times the path crossed the fall line (each crossing is one turn). Saves the
longest run's track, with the target line and the fences, to <run>/trail.png (or logs/playground/snowboard/
no_policy_<task>_trail.png).
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so this runs with plain python too

from playground.quadruped.common import ROOT, add_suppress_arg, load_policy, make_env

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

from playground.quadruped.common import parse_overrides


def main():
    run = yaml.safe_load((args.run / "run.yaml").read_text()) if args.run else {"task": "sideslip", "overrides": {}}
    task = args.task or run["task"]
    overrides = run["overrides"] | parse_overrides(args.set)
    out = args.run / "trail.png" if args.run else ROOT / f"logs/playground/snowboard/no_policy_{task}_trail.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg
        from playground.snowboard.trail import SnowTrail

        # short, repeatable runs: fixed start phase, mid-range speed (unless --set overrides them)
        defaults = SnowboardEnvCfg(task=task)
        user = parse_overrides(args.set)
        lo, hi = overrides.get("speed_cmd", defaults.speed_cmd)
        if "speed_cmd" not in user:
            overrides["speed_cmd"] = ((lo + hi) / 2, (lo + hi) / 2)
        lo, hi = overrides.get("turn_length", defaults.turn_length)
        wavelength = (lo + hi) / 2
        if "turn_length" not in user:
            overrides["turn_length"] = (wavelength, wavelength)
        if "fixed_phase" not in user:
            overrides["fixed_phase"] = 0.0  # start on the piste centre, swinging out
        run_length = args.run_length or (25.0 if task == "sideslip" else 2 * wavelength)
        overrides["run_length"] = run_length
        overrides["episode_length_s"] = 60.0  # runs end by distance; this only catches a rider who stops
        if not args.follow:
            overrides["viewer"] = free_view_cfg(overrides.get("slope_deg"), args.cam_x or run_length + 10.0)
        raw_env = make_env(SnowboardEnv, SnowboardEnvCfg(task=task), overrides, args.num_envs, sim_cfg,
                           args.suppress_warnings)
        if args.no_policy:
            env, agent = raw_env, None
        else:
            env, agent = load_policy(raw_env, run, args.run, args.checkpoint, cfg_dir=HERE)
        from envs import config

        board = config.load()["board"]
        if not args.follow:
            _set_focal_length(raw_env.cfg.viewer.cam_prim_path, args.focal)
        trail = SnowTrail(board["length"], board["waist_width"], raw_env.q_slope)

        steps = int(args.seconds / raw_env.step_dt)
        obs, _ = env.reset()
        best: tuple[float, list] = (-1.0, [])
        ep = {"t": 0, "e_h": 0.0, "e_s": 0.0, "e_l": 0.0, "speed": 0.0, "edge": 0.0, "y_max": 0.0, "cross": 0, "side": 0}
        print(f"\n[play] task={task}  {'no policy' if args.no_policy else args.run.name}  run {run_length:.0f} m, "
              f"turn length {wavelength:.0f} m, fence ±{raw_env.cfg.fence_half_width:.0f} m")
        print("[play] run  end    seconds  speed  |edge|°  heading_err°  slip_err°  line_err_m  max|y|_m  turns")
        n_ep = 0
        for t in range(steps):
            with torch.inference_mode():
                if agent is None:
                    obs, _, terminated, truncated, _ = env.step(torch.zeros(raw_env.num_envs, 28, device=raw_env.device))
                else:
                    actions, outputs = agent.act(obs, None, timestep=t, timesteps=steps)
                    obs, _, terminated, truncated, _ = env.step(outputs.get("mean_actions", actions))
            done = bool(terminated.view(-1)[0] or truncated.view(-1)[0])
            if not done:
                e_h, e_s, e_l = raw_env.errors()
                heading = raw_env.path_heading()[0].item()
                trail.add(raw_env.body_pos(raw_env.board_id)[0], raw_env.slip_angle()[0].item())
                side = 1 if heading > 0.05 else (-1 if heading < -0.05 else 0)
                if side and ep["side"] and side != ep["side"]:
                    ep["cross"] += 1
                ep["side"] = side or ep["side"]
                ep["t"] += 1
                ep["e_h"] += abs(math.degrees(e_h[0].item()))
                ep["e_s"] += abs(math.degrees(e_s[0].item()))
                ep["e_l"] += abs(e_l[0].item())
                ep["y_max"] = max(ep["y_max"], abs(raw_env.position()[0, 1].item()))
                ep["speed"] += raw_env.body_lin_vel(raw_env.board_id)[0].norm().item()
                ep["edge"] += abs(math.degrees(raw_env.snow_info["edge_angle"][0].item()))
            if done or t == steps - 1:
                n_ep += 1
                k = max(ep["t"], 1)
                secs = ep["t"] * raw_env.step_dt
                if bool(raw_env.out_of_bounds[0]):
                    end = "fence"
                elif bool(terminated.view(-1)[0]):
                    end = "fell"
                else:
                    end = "done" if bool(truncated.view(-1)[0]) else "-"
                print(f"[play] {n_ep:3d}  {end:5s}  {secs:7.1f}  {ep['speed'] / k:5.1f}  {ep['edge'] / k:6.1f}  "
                      f"{ep['e_h'] / k:12.1f}  {ep['e_s'] / k:9.1f}  {ep['e_l'] / k:10.2f}  {ep['y_max']:8.1f}  "
                      f"{ep['cross']:5d}")
                if secs > best[0]:
                    best = (secs, list(trail.marks))
                trail.marks.clear()
                trail.reset()
                ep = {k_: 0 if isinstance(v, int) else 0.0 for k_, v in ep.items()}

        trail.marks = best[1]
        swing = raw_env.swing[0].item()
        trail.save_png(out, f"{task}: {best[0]:.1f} s, turn length {wavelength:.0f} m",
                       line=lambda x: swing * math.sin(2 * math.pi * x / wavelength + raw_env.phase[0].item()),
                       fence=raw_env.cfg.fence_half_width, x_end=run_length)
        print(f"[play] track of the longest episode ({best[0]:.1f} s) -> {out}")
        env.close()


def free_view_cfg(slope_deg: float | None, cam_x: float):
    """Camera placed once, low on the snow cam_x down the run, looking back up at env 0's start. Kit leaves it alone
    after that (origin_type "env"), so the mouse controls it. "asset_root" (--follow) re-aims it every frame."""
    from isaaclab.envs import ViewerCfg

    from envs import config
    from envs.mountain import slope_to_world

    a = slope_deg if slope_deg is not None else config.load()["slope"]["angle_deg"]
    # 0.6 m up: on a plane the horizon sits at eye height, so the 1.6 m rider rises above it into the sky
    return ViewerCfg(eye=slope_to_world(a, cam_x, 3.0, 0.6), lookat=slope_to_world(a, 0.0, 0.0, 1.0), origin_type="env")


def _set_focal_length(cam_prim_path: str, focal_mm: float):
    """Longer lens than Kit's ~18 mm default: the rider fills more of the frame from the bottom of the run."""
    from isaaclab.sim.utils.stage import get_current_stage

    prim = get_current_stage().GetPrimAtPath(cam_prim_path)
    if prim.IsValid():  # only exists with a viewer (e.g. --livestream)
        prim.GetAttribute("focalLength").Set(focal_mm)


if __name__ == "__main__":
    main()
