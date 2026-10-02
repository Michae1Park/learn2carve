"""Run a trained policy with a fixed velocity command and print each foot's contact pattern.

    scripts/py playground/quadruped/play.py logs/playground/quadruped/<run>
    scripts/py playground/quadruped/play.py <run> --cmd 1.0 0 0 --livestream 2 --seconds 60

Prints a footfall diagram of env 0 (█ = foot on the ground) so you can tell walk / trot / pace apart headless:

    trot: FL █████_____█████_____    pace: FL █████_____█████_____
          FR _____█████_____█████          FR _____█████_____█████
          RL _____█████_____█████          RL █████_____█████_____
          RR █████_____█████_____          RR _____█████_____█████
"""

import argparse
from pathlib import Path

from common import make_env

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, help="run directory from train.py")
parser.add_argument("--checkpoint", default="best_agent.pt", help="file in <run>/checkpoints/")
parser.add_argument("--cmd", type=float, nargs=3, default=(0.8, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=10.0)
add_launcher_args(parser)
args = parser.parse_args()

import torch
import yaml

import isaaclab.sim as sim_utils


def main():
    run = yaml.safe_load((args.run / "run.yaml").read_text())
    vx, vy, yaw = args.cmd
    overrides = run["overrides"] | {"cmd_vx": (vx, vx), "cmd_vy": (vy, vy), "cmd_yaw": (yaw, yaw)}

    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from skrl.utils.runner.torch import Runner

        from common import load_agent_cfg
        from playground.quadruped.env import FEET

        raw_env = make_env(run["gait"], overrides, args.num_envs, sim_cfg)
        env = SkrlVecEnvWrapper(raw_env)
        agent_cfg = load_agent_cfg(run["algo"])
        agent_cfg["agent"]["experiment"] = {"write_interval": 0, "checkpoint_interval": 0}
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent = Runner(env, agent_cfg).agent
        agent.load(str(args.run / "checkpoints" / args.checkpoint))
        agent.enable_training_mode(False, apply_to_models=True)

        steps = int(args.seconds / raw_env.step_dt)
        contacts, speeds = [], []
        obs, _ = env.reset()
        for t in range(steps):
            with torch.inference_mode():
                actions, outputs = agent.act(obs, None, timestep=t, timesteps=steps)
                obs, *_ = env.step(outputs.get("mean_actions", actions))
            forces = raw_env.contact.data.net_normal_forces_w.torch[0, raw_env.feet].norm(dim=-1)
            contacts.append((forces > 1.0).tolist())
            speeds.append(raw_env.robot.data.root_lin_vel_b.torch[:, 0].mean().item())

        window = contacts[-50:]  # last second at 50 Hz
        print(f"\n[play] gait={run['gait']} algo={run['algo']} cmd={args.cmd}  mean vx (last 2 s) = "
              f"{sum(speeds[-100:]) / len(speeds[-100:]):.2f} m/s")
        for i, foot in enumerate(FEET):
            print(f"  {foot[:2]} " + "".join("█" if c[i] else "_" for c in window))
        env.close()


if __name__ == "__main__":
    main()
