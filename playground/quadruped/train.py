"""Train the Go2 to follow velocity commands with PPO or SAC.

    scripts/py playground/quadruped/train.py --algo ppo --gait free
    scripts/py playground/quadruped/train.py --algo ppo --gait pace --set rew_gait=2.0 "cmd_vx=[0.3,1.0]"
    scripts/py playground/quadruped/train.py --algo sac --gait trot --timesteps 5000

Runs land in logs/playground/quadruped/<time>_<algo>_<gait>/ (TensorBoard events + checkpoints/).
--set overrides any Go2EnvCfg field (reward weights, command ranges, ...); see env.py.
"""

import argparse
from datetime import datetime

from common import LOG_ROOT, load_agent_cfg, make_env, parse_overrides

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
parser.add_argument("--gait", choices=["free", "walk", "trot", "pace"], default="free")
parser.add_argument("--num_envs", type=int, default=None, help="default: from <algo>.yaml")
parser.add_argument("--timesteps", type=int, default=None, help="env steps per env; default: from <algo>.yaml")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Go2EnvCfg overrides")
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
args = parser.parse_args()
args.headless = True

import yaml

import isaaclab.sim as sim_utils


def main():
    agent_cfg = load_agent_cfg(args.algo)
    num_envs = args.num_envs or agent_cfg["num_envs"]
    timesteps = args.timesteps or agent_cfg["timesteps"]
    overrides = parse_overrides(args.set)

    run_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.algo}_{args.gait}"
    run_dir = LOG_ROOT / run_name
    run_dir.mkdir(parents=True)
    run = {"algo": args.algo, "gait": args.gait, "overrides": overrides, "num_envs": num_envs, "timesteps": timesteps}
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False))

    agent_cfg["seed"] = args.seed
    agent_cfg["agent"]["experiment"] = {"directory": str(LOG_ROOT), "experiment_name": run_name}
    agent_cfg["trainer"]["timesteps"] = timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from skrl.utils import set_seed
        from skrl.utils.runner.torch import Runner

        set_seed(args.seed)
        env = SkrlVecEnvWrapper(make_env(args.gait, overrides, num_envs, sim_cfg))
        runner = Runner(env, agent_cfg)
        print(f"[train] {args.algo} / {args.gait} / {num_envs} envs / {timesteps} steps -> {run_dir}")
        try:
            runner.run()
        except KeyboardInterrupt:
            print("[train] interrupted")
        runner.agent.write_checkpoint(timestep=timesteps, timesteps=timesteps)
        env.close()


if __name__ == "__main__":
    main()
