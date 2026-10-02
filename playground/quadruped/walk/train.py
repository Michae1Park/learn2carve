"""Train the Go2 to follow velocity commands with PPO or SAC.

    scripts/py playground/quadruped/walk/train.py --algo ppo --gait free
    scripts/py playground/quadruped/walk/train.py --algo ppo --gait pace --set rew_gait=2.0 "cmd_vx=[0.3,1.0]"
    scripts/py playground/quadruped/walk/train.py --algo sac --gait trot --timesteps 5000

Runs land in logs/playground/quadruped/<time>_<algo>_<gait>/ (TensorBoard events + checkpoints/).
--set overrides any Go2EnvCfg field (reward weights, command ranges, ...); see walk/env.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, so this runs with plain python too

from playground.quadruped.common import add_train_args, make_env, new_run, train

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gait", choices=["free", "walk", "trot", "pace"], default="free")
add_train_args(parser)
add_launcher_args(parser)
args = parser.parse_args()
args.headless = True

import isaaclab.sim as sim_utils


def main():
    run_dir, run, agent_cfg = new_run(args, task=args.gait, gait=args.gait)
    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from playground.quadruped.walk.env import Go2Env, Go2EnvCfg

        env = make_env(Go2Env, Go2EnvCfg(gait=args.gait), run["overrides"], run["num_envs"], sim_cfg, args.suppress_warnings)
        train(env, agent_cfg, run_dir, run, args.seed)


if __name__ == "__main__":
    main()
