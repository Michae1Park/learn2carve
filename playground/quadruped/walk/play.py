"""Run a trained walking policy with a fixed velocity command and print each foot's contact pattern.

    scripts/py playground/quadruped/walk/play.py logs/playground/quadruped/<run>
    scripts/py playground/quadruped/walk/play.py <run> --cmd 1.0 0 0 --livestream 2 --seconds 60

Prints a footfall diagram of env 0 (█ = foot on the ground) so you can tell walk / trot / pace apart headless:

    trot: FL █████_____█████_____    pace: FL █████_____█████_____
          FR _____█████_____█████          FR _____█████_____█████
          RL _____█████_____█████          RL █████_____█████_____
          RR █████_____█████_____          RR _____█████_____█████
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, so this runs with plain python too

from playground.quadruped.common import add_play_args, close_view_cfg, load_policy, make_env

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_play_args(parser)
parser.add_argument("--cmd", type=float, nargs=3, default=(0.8, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
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
        from playground.quadruped.go2 import FEET
        from playground.quadruped.walk.env import Go2Env, Go2EnvCfg

        overrides["viewer"] = close_view_cfg()
        raw_env = make_env(Go2Env, Go2EnvCfg(gait=run["gait"]), overrides, args.num_envs, sim_cfg, args.suppress_warnings)
        env, agent = load_policy(raw_env, run, args.run, args.checkpoint)

        steps = int(args.seconds / raw_env.step_dt)
        contacts, speeds = [], []
        obs, _ = env.reset()
        for t in range(steps):
            with torch.inference_mode():
                actions, outputs = agent.act(obs, None, timestep=t, timesteps=steps)
                obs, *_ = env.step(outputs.get("mean_actions", actions))
            contacts.append(raw_env.feet_in_contact()[0].tolist())
            speeds.append(raw_env.robot.data.root_lin_vel_b.torch[:, 0].mean().item())

        window = contacts[-50:]  # last second at 50 Hz
        print(f"\n[play] gait={run['gait']} algo={run['algo']} cmd={args.cmd}  mean vx (last 2 s) = "
              f"{sum(speeds[-100:]) / len(speeds[-100:]):.2f} m/s")
        for i, foot in enumerate(FEET):
            print(f"  {foot[:2]} " + "".join("█" if c[i] else "_" for c in window))
        env.close()


if __name__ == "__main__":
    main()
