"""Train the Go2 to backflip with PPO (or SAC).

    scripts/py playground/quadruped/backflip/train.py
    scripts/py playground/quadruped/backflip/train.py --set rew_height=20 action_scale=0.7

Runs land in logs/playground/quadruped/<time>_<algo>_backflip/ (TensorBoard events + checkpoints/).
--set overrides any BackflipEnvCfg field (reward weights, timeline, action_scale, ...); see backflip/env.py.
Watch Episode_Flip/rotations (1.0 = one full flip) and Episode_Flip/success in TensorBoard.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, so this runs with plain python too

from playground.quadruped.common import add_train_args, make_env, new_run, train

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_train_args(parser, default_timesteps=30000)  # harder than walking: twice the walk's PPO budget
add_launcher_args(parser)
args = parser.parse_args()
args.headless = True

import isaaclab.sim as sim_utils


def main():
    run_dir, run, agent_cfg = new_run(args, task="backflip")
    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from playground.quadruped.backflip.env import BackflipEnv, BackflipEnvCfg

        env = make_env(BackflipEnv, BackflipEnvCfg(), run["overrides"], run["num_envs"], sim_cfg, args.suppress_warnings)
        train(env, agent_cfg, run_dir, run, args.seed)


if __name__ == "__main__":
    main()
