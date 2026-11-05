"""Rider model check: hold the stance (zero action) on flat snow and measure sag, stability and speed.

    scripts/py scripts/rider_check.py                                       # baseline rider, 120 Hz, 32/4
    scripts/py scripts/rider_check.py --set physics_hz=240 "solver_iterations=[8,4]"
    scripts/py scripts/rider_check.py --set thoracic=true pelvis_mass_frac=0.2

Prints one line: DoF, worst and mean joint error from the stance after --seconds [rad], whether anyone fell,
centre of mass height above the board, and wall time per policy step (--num_envs envs).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.common import parse_overrides, quiet_physx_tensors

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="CarveEnvCfg overrides")
add_launcher_args(parser)
args = parser.parse_args()

import torch

import isaaclab.sim as sim_utils


def main():
    with launch_simulation(sim_utils.SimulationCfg(), args):
        from envs.carve_env import CarveEnv, CarveEnvCfg

        quiet_physx_tensors()
        cfg = CarveEnvCfg()
        cfg.slope_deg, cfg.start_speed, cfg.episode_length_s = 0.0, (0.0, 0.0), 60.0
        for key, value in parse_overrides(args.set).items():
            setattr(cfg, key, value)
        cfg.scene.num_envs = args.num_envs
        env = CarveEnv(cfg)
        env.reset()
        action = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
        steps = int(args.seconds / env.step_dt)
        fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for i in range(steps):
            if i == 5:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            _, _, terminated, _, _ = env.step(action)
            fell |= terminated
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / (steps - 5) * 1000

        err = (env.robot.data.joint_pos.torch - env.stance_pos).abs()
        mass = env.robot.data.body_mass.torch[0]
        pos = env.body_pos(slice(None))[0]
        com = (mass[:, None] * pos).sum(0) / mass.sum()
        com_h = (com[2] - pos[env.board_id, 2]).item()
        worst = err.max(0).values
        names = env.robot.joint_names
        top = sorted(zip(worst.tolist(), names), reverse=True)[:3]
        print(f"[rider] {' '.join(args.set) or 'baseline'}")
        print(f"[rider] dof {len(names)}  dt 1/{round(1 / env.physics_dt)}  sag max {err.max():.3f} mean {err.mean():.4f} rad  "
              f"fell {fell.float().mean():.0%}  com {com_h:.3f} m  pelvis {mass[env.pelvis_id]:.1f} kg  "
              f"step {ms:.1f} ms ({env.num_envs} envs)")
        print(f"[rider] worst joints: " + ", ".join(f"{n} {v:.3f}" for v, n in top))
        info = env.snow_info
        if "camber_engaged" in info:
            twist = env.robot.data.joint_pos.torch[:, names.index("board_flex:0")]
            print(f"[rider] board: camber flattened {info['camber_engaged'].mean():.2f}  load on nose "
                  f"{info['load_nose'].mean():.2f}  N {info['normal_force'].mean():.0f} N  "
                  f"twist {twist.abs().mean().rad2deg():.2f}°")
        env.close()


if __name__ == "__main__":
    main()
