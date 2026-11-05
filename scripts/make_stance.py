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
parser.add_argument("--angles", type=float, nargs=2, default=None, metavar=("FRONT", "BACK"),
                    help="binding angles [deg]; default config.yaml. Others write envs/assets/stance_<front>_<back>.yaml")
parser.add_argument("--knee", type=float, default=0.6, help="initial guess: knee flexion [rad]")
parser.add_argument("--hip_y", type=float, default=-0.35, help="initial guess: hip pitch [rad], negative = flexed")
parser.add_argument("--ankle_y", type=float, default=-0.25, help="initial guess: ankle pitch [rad]")
parser.add_argument("--guess_from", type=Path, default=None,
                    help="start the settle from another stance file's joint angles (lower the hip step by step)")
parser.add_argument("--back", type=float, default=0.0, help="hips pushed back [m] along the board's width (+/-: try both)")
parser.add_argument("--tail", type=float, default=0.0, help="hips shifted toward the tail [m] along the board (loads the back foot)")
parser.add_argument("--pitch", type=float, default=0.0, help="pelvis hinged forward [deg] about its own y axis (+/-)")
parser.add_argument("--twist", type=float, default=0.0, help="torso twist abdomen_z [rad]: opens the chest toward the nose")
parser.add_argument("--to_height", action="store_true", help="scale the humanoid to config.yaml's rider height")
parser.add_argument("--stance_width", type=float, default=None, help="binding centre to centre [m]; default config.yaml")
parser.add_argument("--tag", default="", help="write a stance variant, e.g. squat: stance[_<front>_<back>]_<tag>.yaml")
add_launcher_args(parser)
args = parser.parse_args()

import math

import torch
import yaml

import isaaclab.sim as sim_utils

from envs import config

# initial guess: bent knees, flexed hips, feet turned toward the nose
GUESS = {".*_knee": args.knee, ".*_hip_y": args.hip_y, ".*_ankle_y": args.ankle_y, "left_hip_z": 0.38,
         "right_hip_z": 0.12, "abdomen_z": args.twist}
if args.guess_from is not None:
    import yaml as _yaml

    GUESS = _yaml.safe_load(args.guess_from.read_text())["joint_pos"]


def main():
    cfg = config.load()
    angles = tuple(args.angles) if args.angles else None
    board = cfg["board"]
    board_top = board["thickness"] / 2
    board_z = 0.5  # pin the board in the air; gravity is off
    board_pose = ((0.0, 0.0, board_z), (1.0, 0.0, 0.0, 0.0))  # wxyz for USD
    pelvis_yaw = math.radians(-90.0 + args.open_deg)
    pelvis_pos = (board["setback"] - args.tail, args.back, board_z + board_top + args.hip_height)
    sz, cz = math.sin(pelvis_yaw / 2), math.cos(pelvis_yaw / 2)
    sy, cy = math.sin(math.radians(args.pitch) / 2), math.cos(math.radians(args.pitch) / 2)
    # yaw about z, then hinge about the pelvis's own y: q = q_yaw * q_pitch (xyzw)
    pelvis_quat = (-sz * sy, cz * sy, sz * cy, cz * cy)

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args.device, gravity=(0.0, 0.0, 0.0))
    with launch_simulation(sim_cfg, args):
        # USD-dependent imports only after Kit is up, or the kit-less pxr from site-packages wins
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

        from envs.rider import binding_frames, foot_sole, rider_cfg, rider_scale

        scale = rider_scale() if args.to_height else 1.0
        width = {"stance_width": args.stance_width} if args.stance_width else {}

        sim = SimulationContext(sim_cfg)
        sim_utils.create_prim("/World/Origin", "Xform")
        pelvis_pose = (pelvis_pos, (pelvis_quat[3], *pelvis_quat[:3]))  # wxyz for USD
        robot_cfg = rider_cfg(world_pins={"board": board_pose, "pelvis": pelvis_pose}, binding_angles=angles,
                              stance_width=args.stance_width, scale_factor=scale)
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
        frames = binding_frames(board, cfg["rider"] | ({"binding_angles_deg": list(angles)} if angles else {}) | width)
        sole = torch.tensor(foot_sole(scale), device=pos.device)
        for foot, (bind_pos, bind_rot) in frames.items():
            f = names.index(foot)
            sole_w = pos[f] + quat_apply(quat[f], sole)
            bind_w = bp + quat_apply(bq, torch.tensor(bind_pos, device=pos.device))
            w, x, y, z = bind_rot  # wxyz -> xyzw
            bind_q = quat_mul(bq, torch.tensor([x, y, z, w], device=pos.device))
            dq = quat_mul(quat_conjugate(bind_q), quat[f])
            ang = math.degrees(2 * math.acos(min(1.0, abs(dq[3].item()))))
            print(f"[stance] {foot} binding residual: {1000 * (sole_w - bind_w).norm().item():.2f} mm, {ang:.2f} deg")

        # squat metrics. f = the way the pelvis faces, flat. knee_fwd: knee ahead of the ankle along its foot's direction (a foot reaches
        # ~0.1 m past its ankle, so more than that is a knee poking out past the toes); hip_back: hips behind the feet
        # (negative = back); lean: torso from vertical, + = forward; chest_yaw: chest vs the board's nose (0 = facing it)
        def flat(v):
            v = v.clone()
            v[2] = 0.0
            return v / v.norm().clamp_min(1e-6)

        pi_ = names.index("pelvis")
        fwd = flat(quat_apply(quat[pi_], torch.tensor([1.0, 0.0, 0.0], device=pos.device)))
        up = torch.tensor([0.0, 0.0, 1.0], device=pos.device)
        parts = []
        for side in ("right", "left"):
            foot_dir = flat(quat_apply(quat[names.index(f"{side}_foot")], torch.tensor([1.0, 0.0, 0.0], device=pos.device)))
            knee_fwd = ((pos[names.index(f"{side}_shin")] - pos[names.index(f"{side}_foot")]) * foot_dir).sum().item()
            parts.append(f"{side} knee_fwd {knee_fwd:+.2f} m")
        feet_mid = (pos[names.index("right_foot")] + pos[names.index("left_foot")]) / 2
        parts.append(f"hip_back {((pos[pi_] - feet_mid) * fwd).sum().item():+.2f} m")
        torso_v = pos[names.index("head")] - pos[pi_]
        parts.append(f"lean {math.degrees(math.atan2((torso_v * fwd).sum().item(), torso_v[2].item())):+.0f} deg")
        chest = flat(quat_apply(quat[names.index("torso")], torch.tensor([1.0, 0.0, 0.0], device=pos.device)))
        nose = flat(quat_apply(bq, torch.tensor([1.0, 0.0, 0.0], device=pos.device)))
        cyaw = math.degrees(math.atan2(torch.cross(nose, chest, dim=-1)[2].item(), (nose * chest).sum().item()))
        rf, lf = pos[names.index("right_foot")], pos[names.index("left_foot")]
        t_ = flat(rf - lf)
        mass = robot.data.body_mass.torch[0]
        keep = [i for i, n in enumerate(names) if not n.startswith("board")]
        com = (pos[keep] * mass[keep].view(-1, 1)).sum(0) / mass[keep].sum()
        span = ((rf - lf) * t_).sum().item()
        parts.append(f"hip_tail {((pos[pi_] - (rf + lf) / 2) * t_).sum().item():+.2f} m (+ = toward the tail)")
        parts.append(f"back_load {((com - lf) * t_).sum().item() / span:.2f}")
        parts.append(f"chest_yaw {cyaw:+.0f} deg")
        print("[stance] squat: " + ", ".join(parts))

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
        from envs.carve_env import stance_file

        out = stance_file(angles, args.tag)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Generated by scripts/make_stance.py: pelvis pose in the board frame (quat x, y, z, w)\n"
                       + yaml.safe_dump(stance, sort_keys=False))
        print(f"[stance] pelvis in board frame: pos {stance['pelvis_pos']}")
        print(f"[stance] wrote {out}")


if __name__ == "__main__":
    main()
