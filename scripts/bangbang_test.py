"""Scripted edge changes, no learning: can the rider change edges on a timer at all, and at what speed?

    scripts/py scripts/bangbang_test.py --slope 6
    scripts/py scripts/bangbang_test.py --slope 15

Each env follows a timed edge command θ*(t) = A · clamp(1.8 sin(2π t / T), -1, 1) (heel <-> base <-> toe) with a
saturated proportional controller on the lateral joints: u = clamp(k (θ* - θ) / 30°, -1, 1); u > 0 blends toward the
toe pattern, u < 0 toward the heel pattern (the max lateral patterns found by the reach test). k = 100 is bang-bang.
Envs span amplitude A × gain k; each row reports the first episode only: how long it stayed up, edge changes made,
tracking error, speed and slip.
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.common import make_env, parse_overrides

from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--slope", type=float, default=15.0)
p.add_argument("--period", type=float, default=4.0, help="full S [s] (a turn every period / 2)")
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--reps", type=int, default=4)
p.add_argument("--pattern", choices=["roll", "flex"], default="flex")
p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="env cfg overrides")
p.add_argument("--scale", type=float, default=None, help="action_scale override (env default 0.25)")
add_launcher_args(p)
args = p.parse_args()
args.headless = True

import torch

import isaaclab.sim as sim_utils

PATTERNS = {
    # the reach test's max lateral patterns: they reached ±80° mostly by falling over, they don't steer the edge
    "roll": (["right_hip_x", "left_hip_x", "right_ankle_x", "left_ankle_x", "abdomen_x"],
             (1, -1, -1, -1, -1), (-1, -1, 1, -1, -1)),
    # scripts/edge_sweep.py: in the posi-posi stance, ankle flex and knee bend tip the board (+1 -> heel edge)
    "flex": (["left_ankle_y", "left_knee", "right_ankle_y", "right_knee"], (-1, -1, -1, -1), (1, 1, 1, 1)),
}
AMPS = (20.0, 35.0, 50.0)
GAINS = (1.0, 2.0, 4.0, 100.0)

with launch_simulation(sim_utils.SimulationCfg(), args):
    from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

    combos = list(itertools.product(AMPS, GAINS)) * args.reps
    ov = {"stance_tag": "squat3", "binding_angles_deg": (54, 45), "pelvis_mass_frac": 0.2, "guide_steps": 0,
          "slope_deg": args.slope, "episode_length_s": args.seconds + 5, "start_heading_deg": 0.0}
    if args.scale:
        ov["action_scale"] = args.scale
    ov |= parse_overrides(args.set)
    LAT, TOE, HEEL = PATTERNS[args.pattern]
    raw = make_env(SnowboardEnv, SnowboardEnvCfg(task="carve"), ov, len(combos), sim_utils.SimulationCfg(), "true")
    names = [raw.robot.joint_names[i] for i in raw.actuated.tolist()]
    idx = torch.tensor([names.index(n) for n in LAT], device=raw.device)
    toe, heel = (torch.tensor(v, dtype=torch.float, device=raw.device) for v in (TOE, HEEL))
    amp = torch.tensor([math.radians(a) for a, _ in combos], device=raw.device)
    gain = torch.tensor([k for _, k in combos], device=raw.device)
    phase = torch.tensor([math.pi * (i % 2) for i in range(len(combos))], device=raw.device)  # heel or toe first

    raw.reset()
    n = len(combos)
    alive = torch.ones(n, dtype=torch.bool, device=raw.device)
    up = torch.zeros(n, device=raw.device)
    err = torch.zeros(n, device=raw.device)
    speed = torch.zeros(n, device=raw.device)
    slip = torch.zeros(n, device=raw.device)
    vmax = torch.zeros(n, device=raw.device)
    side = torch.zeros(n, device=raw.device)
    flips = torch.zeros(n, device=raw.device)
    act = torch.zeros(n, len(names), device=raw.device)
    for step in range(int(args.seconds / raw.step_dt)):
        t = step * raw.step_dt
        cmd = amp * (1.8 * torch.sin(2 * math.pi * t / args.period + phase)).clamp(-1.0, 1.0)
        theta = raw.snow_info["edge_angle"] if raw.snow_info else torch.zeros(n, device=raw.device)
        u = (gain * (cmd - theta) / math.radians(30.0)).clamp(-1.0, 1.0)
        pattern = torch.where(u.unsqueeze(-1) > 0, toe, heel) * u.abs().unsqueeze(-1)
        act.zero_()
        act[:, idx] = pattern
        _, _, term, trunc, _ = raw.step(act)
        theta = raw.snow_info["edge_angle"]
        v = raw.board_speed()
        a = alive.float()
        up += a * raw.step_dt
        err += a * (theta - cmd).abs().rad2deg()
        speed += a * v
        vmax = torch.where(alive, torch.maximum(vmax, v), vmax)
        slip += a * raw.slip_angle().abs().rad2deg()
        held = torch.sign(theta) * (theta.abs() > math.radians(15.0))
        flips += a * ((held != 0) & (side != 0) & (held != side)).float()
        side = torch.where(alive & (held != 0), held, side)
        alive = alive & ~(term | trunc)

    steps = (up / raw.step_dt).clamp_min(1)
    print(f"\n[bang] pattern {args.pattern}, overrides {parse_overrides(args.set)}, slope {args.slope}°, S period {args.period} s, {args.seconds} s max; mean over {args.reps} envs each")
    print("[bang]  amp°  gain  up_s  flips  err°  speed  vmax  slip°   (flips = edge changes ≥15° to ≥15°)")
    for a_deg, k in itertools.product(AMPS, GAINS):
        m = torch.tensor([c == (a_deg, k) for c in combos], device=raw.device)
        row = [x[m].mean().item() for x in (up, flips, err / steps, speed / steps, vmax, slip / steps)]
        print(f"[bang]  {a_deg:4.0f}  {k:4.0f}  {row[0]:4.1f}  {row[1]:5.1f}  {row[2]:4.1f}  {row[3]:5.1f}  {row[4]:4.1f}  {row[5]:5.1f}")
    raw.close()
