"""Run a trained backflip policy and print how each flip went.

    scripts/py playground/quadruped/backflip/play.py logs/playground/quadruped/<run>
    scripts/py playground/quadruped/backflip/play.py <run> --livestream 2 --seconds 60

Every episode is one flip attempt (2 s by default), so a 60 s run shows ~30 attempts back to back.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, so this runs with plain python too

from playground.quadruped.common import add_play_args, close_view_cfg, load_policy, make_env

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_play_args(parser)
add_launcher_args(parser)
args = parser.parse_args()

import torch
import yaml

import isaaclab.sim as sim_utils


def main():
    run = yaml.safe_load((args.run / "run.yaml").read_text())
    overrides = run["overrides"]

    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from playground.quadruped.backflip.env import BackflipEnv, BackflipEnvCfg

        overrides["viewer"] = close_view_cfg()
        raw_env = make_env(BackflipEnv, BackflipEnvCfg(), overrides, args.num_envs, sim_cfg, args.suppress_warnings)
        env, agent = load_policy(raw_env, run, args.run, args.checkpoint)

        steps = int(args.seconds / raw_env.step_dt)
        attempts, flips = 0, 0
        obs, _ = env.reset()
        for t in range(steps):
            with torch.inference_mode():
                actions, outputs = agent.act(obs, None, timestep=t, timesteps=steps)
                obs, _, terminated, truncated, _ = env.step(outputs.get("mean_actions", actions))
            if terminated[0] or truncated[0]:  # env 0 just finished an attempt
                angle, fell = raw_env.last_flip_angle[0].item(), raw_env.last_fell[0].item()
                ok = angle > raw_env.cfg.success_angle and not fell
                attempts, flips = attempts + 1, flips + ok
                print(f"[play] attempt {attempts:3d}: {angle / (2 * math.pi):5.2f} rotations, "
                      f"{'fell' if fell else 'landed'}{'  FLIP' if ok else ''}")

        print(f"\n[play] {flips}/{attempts} clean flips")
        env.close()


if __name__ == "__main__":
    main()
