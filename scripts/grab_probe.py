"""Hand grab per side: ride a policy and report how close the inside hand gets to its grab point in each turn.

    scripts/py scripts/grab_probe.py logs/playground/snowboard/<run> [--seconds 15]

Heel turn: right hand -> toe edge at the front foot. Toe turn: left hand -> heel edge at the back foot.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.common import load_policy, make_env, parse_overrides, run_overrides

from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("run", type=Path)
p.add_argument("--checkpoint", default="best_agent.pt")
p.add_argument("--seconds", type=float, default=15.0)
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
add_launcher_args(p)
args = p.parse_args()
args.headless = True

import torch
import yaml

import isaaclab.sim as sim_utils

with launch_simulation(sim_utils.SimulationCfg(), args):
    from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

    run = yaml.safe_load((args.run / "run.yaml").read_text())
    ov = run_overrides(run) | {"rsi_frac": 0.0} | parse_overrides(args.set)
    raw = make_env(SnowboardEnv, SnowboardEnvCfg(), ov, args.num_envs, sim_utils.SimulationCfg(), "true", legacy=True)
    env, agent = load_policy(raw, run, args.run, args.checkpoint)
    stats = {k: [] for k in ("heel", "toe")}
    obs, _ = env.reset()
    for t in range(int(args.seconds / raw.step_dt)):
        with torch.inference_mode():
            a, out = agent.act(obs, None, timestep=t, timesteps=1)
            obs, _, term, trunc, _ = env.step(out.get("mean_actions", a))
            theta = raw.snow_info["edge_angle"]
            d_heel, d_toe, heel, toe = raw._grab_parts(theta)
            stats["heel"].append(d_heel[heel])
            stats["toe"].append(d_toe[toe])
    for side, hand in (("heel", "right hand -> toe edge, front foot"), ("toe", "left hand -> heel edge, back foot")):
        d = torch.cat(stats[side]) if stats[side] else torch.zeros(0)
        if len(d) == 0:
            print(f"[grab] {side} turn ({hand}): never active")
            continue
        q = torch.quantile(d, torch.tensor([0.1, 0.5, 0.9], device=d.device)).tolist()
        print(f"[grab] {side} turn ({hand}): {len(d)} samples, distance p10 {q[0]:.2f}  median {q[1]:.2f}  "
              f"p90 {q[2]:.2f} m, within 0.3 m {(d < 0.3).float().mean().item():.0%}")
