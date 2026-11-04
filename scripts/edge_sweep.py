"""Which joints tip the board? Hold one actuated joint at ±1 action (all others at the stance) and measure the edge
angle it produces, while the rider is still up.

    scripts/py scripts/edge_sweep.py
    scripts/py scripts/edge_sweep.py --set 'ankle_range_deg={x: 45, y: 80, z: 55}'

One env per (joint, sign). Prints the edge angle at --hold seconds (and whether the env was still up), sorted by
effect, so the strongest heel and toe joints come first.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.quadruped.common import make_env, parse_overrides

from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--slope", type=float, default=6.0)
p.add_argument("--hold", type=float, default=0.7, help="seconds to hold the action before reading the edge")
p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
add_launcher_args(p)
args = p.parse_args()
args.headless = True

import torch

import isaaclab.sim as sim_utils

with launch_simulation(sim_utils.SimulationCfg(), args):
    from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

    ov = {"stance_tag": "squat3", "binding_angles_deg": (54, 45), "pelvis_mass_frac": 0.2, "guide_steps": 0,
          "slope_deg": args.slope, "start_heading_deg": 0.0} | parse_overrides(args.set)
    probe = SnowboardEnvCfg(task="carve")
    names = None
    # one env per (joint, sign) + 1 reference at zero action
    raw = make_env(SnowboardEnv, probe, ov, 2 * 28 + 1, sim_utils.SimulationCfg(), "true")
    names = [raw.robot.joint_names[i] for i in raw.actuated.tolist()]
    n = 2 * len(names) + 1
    act = torch.zeros(raw.num_envs, len(names), device=raw.device)
    for j in range(len(names)):
        act[2 * j, j], act[2 * j + 1, j] = 1.0, -1.0
    raw.reset()
    alive = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    edge = torch.zeros(raw.num_envs, device=raw.device)
    for _ in range(int(args.hold / raw.step_dt)):
        _, _, term, trunc, _ = raw.step(act)
        e = raw.snow_info["edge_angle"].rad2deg()
        edge = torch.where(alive, e, edge)
        alive = alive & ~(term | trunc)
    ref = edge[n - 1].item()
    rows = [(edge[2 * j + k].item() - ref, names[j], "+1" if k == 0 else "-1", bool(alive[2 * j + k]))
            for j in range(len(names)) for k in (0, 1)]
    rows.sort()
    print(f"\n[sweep] slope {args.slope}°, hold {args.hold} s, overrides {parse_overrides(args.set)}; reference edge {ref:.1f}°")
    print("[sweep] Δedge°  joint            action  up")
    for d, j, s, up in rows:
        if abs(d) >= 1.0:
            print(f"[sweep] {d:6.1f}  {j:16s} {s:>4}  {'yes' if up else 'FELL'}")
    raw.close()
