"""Holding targets for a stance: the joint targets at which the real drives hold the stance under gravity.

    scripts/py scripts/feed_forward.py --robot g1 --tag g1                 # envs/physical_g1.yaml's bindings

Soft drives (the G1 at Unitree's gains) sag far off the stance when their targets are the stance itself, and the
policy's small residual can't make up for it. Here the board is bolted to the ground, gravity is on, and the drives
are the robot's own. Each round restarts from the stance, holds the current targets for --hold seconds and moves each
target by the joint's sag (target -= q - stance), until the robot stays put. The result is written into the stance
file as joint_target (the env's zero-action targets). Joints at their torque limit are reported: the stance asks too
much of them.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot", choices=["g1"], default="g1")
parser.add_argument("--angles", type=float, nargs=2, default=None, metavar=("FRONT", "BACK"))
parser.add_argument("--tag", default="g1")
parser.add_argument("--rounds", type=int, default=40)
parser.add_argument("--hold", type=float, default=0.15, help="seconds held per round [s]")
add_launcher_args(parser)
args = parser.parse_args()

import torch
import yaml

import isaaclab.sim as sim_utils

from envs import physical

physical.select(args.robot)
GRAVITY = (0.0, 0.0, -9.81)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args.device, gravity=GRAVITY)
    with launch_simulation(sim_cfg, args):
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import quat_apply

        from envs import rider_g1
        from envs.carve_env import load_stance, stance_file

        angles = tuple(args.angles) if args.angles else None
        stance = load_stance(angles, args.tag)
        board_z = 0.5
        pelvis_pos = tuple(p + (board_z if i == 2 else 0.0) for i, p in enumerate(stance["pelvis_pos"]))
        sim = SimulationContext(sim_cfg)
        sim_utils.create_prim("/World/Origin", "Xform")
        cfg = rider_g1.rider_cfg(stance=stance, world_pins={"board": ((0.0, 0.0, board_z), (1.0, 0.0, 0.0, 0.0))},
                                 binding_angles=angles)
        cfg = cfg.replace(prim_path="/World/Origin/Robot",
                          init_state=cfg.init_state.replace(pos=pelvis_pos, rot=tuple(stance["pelvis_quat"])))
        robot = Articulation(cfg)
        sim.reset()
        names = robot.data.joint_names
        q0 = robot.data.default_joint_pos.torch.clone()
        root0 = robot.data.default_root_pose.torch.clone()
        target = q0.clone()
        body = torch.tensor([("_hand_" not in n) for n in names], device=q0.device)
        steps = max(1, int(args.hold / sim.get_physics_dt()))
        dt = sim.get_physics_dt()

        def hold(t):
            robot.write_root_pose_to_sim_index(root_pose=root0)
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=q0.device))
            robot.write_joint_position_to_sim_index(position=q0)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
            robot.actuators.target_command.set_position_index(value=t)
            for _ in range(steps):
                robot.write_data_to_sim()
                sim.step()
                robot.update(dt)
            return robot.data.joint_pos.torch - q0

        lo, hi = robot.data.soft_joint_pos_limits.torch[0, :, 0], robot.data.soft_joint_pos_limits.torch[0, :, 1]
        err = hold(target)
        print(f"[ff] round 0: worst sag {err[0, body].abs().max():.3f} rad")
        for k in range(args.rounds):
            target = torch.where(body, (target - err).clamp(lo, hi), target)
            err = hold(target)
        worst = err[0, body].abs()
        print(f"[ff] round {args.rounds}: worst sag {worst.max():.3f} rad, mean {worst.mean():.4f}")
        for i in worst.argsort(descending=True)[:3].tolist():
            print(f"[ff]   {names[body.nonzero().flatten()[i]]} {worst[i]:.3f}")

        effort = robot.data.applied_torque.torch[0]
        limit = robot.data.joint_effort_limits.torch[0]
        for i, n in enumerate(names):
            if body[i] and effort[i].abs() > 0.8 * limit[i]:
                print(f"[ff] near torque limit: {n} {effort[i]:+.1f} / {limit[i]:.0f} N·m")
        offset = (target - q0)[0]
        print("[ff] offsets [rad]: " + ", ".join(f"{n} {offset[i]:+.2f}" for i, n in enumerate(names)
                                                  if body[i] and abs(offset[i]) > 0.02))

        stance["joint_target"] = {n: round(float(v), 5) for n, v in zip(names, target[0])}
        stance["joint_target_gains"] = "unitree"  # solved for these drives; the env ignores them under other gains
        out = stance_file(angles, args.tag)
        text = out.read_text().split("\n", 1)[0] + "\n" + yaml.safe_dump(stance, sort_keys=False)
        out.write_text(text)
        print(f"[ff] wrote joint_target to {out}")


if __name__ == "__main__":
    main()
