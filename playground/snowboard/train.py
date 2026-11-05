"""Train the snowboarder to side slip, skid or carve with PPO.

    scripts/py playground/snowboard/train.py --task sideslip
    scripts/py playground/snowboard/train.py --task skid --set slope_deg=10
    scripts/py playground/snowboard/train.py --task carve --init_from logs/playground/snowboard/<run>   # warm start

--livestream 2 streams env 0 to the Isaac Sim WebRTC client. --set overrides any SnowboardEnvCfg field.
Runs land in logs/playground/snowboard/<time>_ppo_<task>/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.common import LOG_ROOT, add_train_args, make_env, new_run, train

from isaaclab.app import add_launcher_args, launch_simulation

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", choices=["sideslip", "skid", "carve"], default="sideslip")
add_train_args(parser)
add_launcher_args(parser)
args = parser.parse_args()
args.headless = not (getattr(args, "livestream", None) or 0) > 0
if args.algo != "ppo":
    parser.error("only ppo is set up for the snowboarder (playground/snowboard/ppo.yaml)")

import isaaclab.sim as sim_utils


def main():
    run_dir, run, agent_cfg = new_run(args, task=args.task, cfg_dir=HERE, log_root=LOG_ROOT, seed=args.seed)
    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

        env = make_env(SnowboardEnv, SnowboardEnvCfg(task=args.task), run["overrides"], run["num_envs"], sim_cfg,
                       args.suppress_warnings)
        train(env, agent_cfg, run_dir, run, args.seed, args.init_from)


if __name__ == "__main__":
    main()
