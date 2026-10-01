"""Build the rider's snowboard stance (Stage 1). Run once, or whenever rider/board specs change.

    scripts/py scripts/make_stance.py

The pelvis and the board are welded to the world and held at the poses we want. The binding
constraints then pull the feet onto the board, the joint drives settle the legs, and the result is a joint
configuration that closes both binding loops exactly. It's saved to envs/assets/stance.yaml:
  joint_pos      default joint targets
  pelvis_pos/quat  pelvis pose in the board frame (quat x, y, z, w)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `envs` without scripts/py

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hip_height", type=float, default=0.80, help="pelvis height above the board top [m]")
parser.add_argument("--open_deg", type=float, default=20.0, help="pelvis rotation from facing the toe edge toward the nose")
parser.add_argument("--seconds", type=float, default=6.0)
add_launcher_args(parser)
args = parser.parse_args()

import math

import torch
import yaml

import isaaclab.sim as sim_utils

from envs import config

# initial guess: bent knees, flexed hips, feet turned toward the nose
GUESS = {".*_knee": 0.6, ".*_hip_y": -0.35, ".*_ankle_y": -0.25, "left_hip_z": 0.38, "right_hip_z": 0.12}


def main():
    cfg = config.load()
    board = cfg["board"]
    board_top = board["thickness"] / 2
    board_z = 0.5  # pin the board in the air; gravity is off
    board_pose = ((0.0, 0.0, board_z), (1.0, 0.0, 0.0, 0.0))  # wxyz for USD
    pelvis_yaw = math.radians(-90.0 + args.open_deg)
    pelvis_pos = (board["setback"], 0.0, board_z + board_top + args.hip_height)
    pelvis_quat = (0.0, 0.0, math.sin(pelvis_yaw / 2), math.cos(pelvis_yaw / 2))  # xyzw

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args.device, gravity=(0.0, 0.0, 0.0))
    with launch_simulation(sim_cfg, args):
        # USD-dependent imports only after Kit is up, or the kit-less pxr from site-packages wins
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

        from envs.rider import FOOT_SOLE, binding_frames, rider_cfg

        sim = SimulationContext(sim_cfg)
        sim_utils.create_prim("/World/Origin", "Xform")
        pelvis_pose = (pelvis_pos, (pelvis_quat[3], *pelvis_quat[:3]))  # wxyz for USD
        robot_cfg = rider_cfg(world_pins={"board": board_pose, "pelvis": pelvis_pose})
        robot_cfg = robot_cfg.replace(
            prim_path="/World/Origin/Robot",
            init_state=robot_cfg.init_state.replace(pos=pelvis_pos, rot=pelvis_quat, joint_pos=GUESS),
        )
        # soft drives while settling, so the joints yield to the binding constraints instead of fighting them
        robot_cfg.actuators["body"] = robot_cfg.actuators["body"].replace(stiffness=30.0, damping=10.0)
        robot = Articulation(robot_cfg)
        sim.reset()
        print("[stance] bodies:", robot.data.body_names)
        print("[stance] joints:", robot.data.joint_names)
        print("[stance] total mass: %.2f kg" % robot.data.body_mass.torch[0].sum().item())

        target = robot.data.default_joint_pos.torch.clone()
        robot.actuators.target_command.set_position_index(value=target)
        dt = sim.get_physics_dt()
        for _ in range(int(args.seconds / dt)):
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

        names = robot.data.body_names
        pos = robot.data.body_pos_w.torch[0]
        quat = robot.data.body_quat_w.torch[0]
        b = names.index("board")
        bp, bq = pos[b], quat[b]

        # binding residual: foot sole frame vs. its binding frame on the board
        frames = binding_frames(board, cfg["rider"])
        sole = torch.tensor(FOOT_SOLE, device=pos.device)
        for foot, (bind_pos, bind_rot) in frames.items():
            f = names.index(foot)
            sole_w = pos[f] + quat_apply(quat[f], sole)
            bind_w = bp + quat_apply(bq, torch.tensor(bind_pos, device=pos.device))
            w, x, y, z = bind_rot  # wxyz -> xyzw
            bind_q = quat_mul(bq, torch.tensor([x, y, z, w], device=pos.device))
            dq = quat_mul(quat_conjugate(bind_q), quat[f])
            ang = math.degrees(2 * math.acos(min(1.0, abs(dq[3].item()))))
            print(f"[stance] {foot} binding residual: {1000 * (sole_w - bind_w).norm().item():.2f} mm, {ang:.2f} deg")

        p = names.index("pelvis")
        rel_pos = quat_apply_inverse(bq, pos[p] - bp)
        rel_quat = quat_mul(quat_conjugate(bq), quat[p])
        joint_pos = robot.data.joint_pos.torch[0]
        stance = {
            "joint_pos": {n: round(float(v), 5) for n, v in zip(robot.data.joint_names, joint_pos)},
            "pelvis_pos": [round(float(v), 5) for v in rel_pos],
            "pelvis_quat": [round(float(v), 6) for v in rel_quat],
            "hip_height": args.hip_height,
            "open_deg": args.open_deg,
        }
        out = config.ROOT / "envs/assets/stance.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Generated by scripts/make_stance.py: pelvis pose in the board frame (quat x, y, z, w)\n"
                       + yaml.safe_dump(stance, sort_keys=False))
        print(f"[stance] pelvis in board frame: pos {stance['pelvis_pos']}")
        print(f"[stance] wrote {out}")


if __name__ == "__main__":
    main()
