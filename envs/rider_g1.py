"""Rider = the stock Unitree G1 (29 DoF + Dex3 hands) + a snowboard, as one articulation.

    pelvis ── ... ── left_ankle_roll_link  ══ front binding (in the articulation tree) ══ board
                     right_ankle_roll_link ══ back binding (loop-closing constraint)  ══╝

The robot is not modified: Isaac Lab's G1_29DOF_CFG masses, joint ranges, gains and motor torque limits. Only the
board, the stance and the binding angles (envs/physical_g1.yaml) are chosen to suit it. The ankle has pitch and roll
only, so the binding yaw comes from the hip yaw joints.
"""

from __future__ import annotations

import math

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import replace
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from . import physical
from .rider import RiderSpawnCfg, binding_frames

FEET = {"left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link"}
FOOT_SOLE = (0.05, 0.0, -0.036)  # centre of the sole, in the ankle roll link frame
TORSO = "torso_link"
HEAD = "torso_link"  # no head link; make_stance measures the lean to the shoulders instead
SHOULDERS = ("left_shoulder_pitch_link", "right_shoulder_pitch_link")
SHINS = {"left": "left_knee_link", "right": "right_knee_link"}
KEY_BODIES = ["right_hand_palm_link", "left_hand_palm_link", FEET["right_foot"], FEET["left_foot"]]
STOCK_JOINTS = 43  # 29 body + 14 hand
PASSIVE_GROUPS = {"_hand_": 14, "wrist": 6}  # DoF-name substring -> DoF count


def implicit_actuators() -> dict:
    """Unitree's deployment gains for the G1 29-DoF (unitree_rl_lab UNITREE_G1_29DOF_CFG), as PhysX implicit drives,
    except the waist roll/pitch (kp 200, see below).

    Isaac Lab's G1_29DOF_CFG drives the waist at kp 5000 and the arms at 3000 with 300 N·m: far stiffer and stronger
    than the real motors, and nearly undamped (the torso rang). These are the gains Unitree deploys with, so a policy
    trained here sees the controller it would run on. Torque limits are the motors' (waist roll/pitch and ankles: two
    5020 motors on a linkage, 50 N·m, as Isaac Lab and Unitree's mimic config).
    """
    arm = 0.01
    return {
        "hip_pitch_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint"], stiffness=100.0, damping=2.0,
            armature=arm, joint_effort_limit=88.0, joint_velocity_limit=32.0),
        "hip_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
            stiffness={".*_hip_roll_joint": 100.0, ".*_knee_joint": 150.0},
            damping={".*_hip_roll_joint": 2.0, ".*_knee_joint": 4.0},
            armature=arm, joint_effort_limit=139.0, joint_velocity_limit=20.0),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"], stiffness=200.0, damping=5.0, armature=arm,
            joint_effort_limit=88.0, joint_velocity_limit=32.0),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*_joint"], stiffness=40.0, damping=2.0, armature=arm,
            joint_effort_limit=50.0, joint_velocity_limit=37.0),
        # waist roll/pitch: Unitree runs kp 40, which caps the torque at ~20 N·m over the ±30° range; a carve needs the
        # motors' full 50 N·m to hold the torso against the lean. kp 200 / kd 5 as Unitree's waist yaw, same motors.
        "waist_roll_pitch": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"], stiffness=200.0, damping=5.0, armature=arm,
            joint_effort_limit=50.0, joint_velocity_limit=37.0),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"], stiffness=40.0,
            damping=1.0, armature=arm, joint_effort_limit=25.0, joint_velocity_limit=37.0),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"], stiffness=40.0, damping=1.0,
            armature=arm, joint_effort_limit=5.0, joint_velocity_limit=22.0),
        "hands": G1_29DOF_CFG.actuators["hands"],
    }


def isaac_actuators() -> dict:
    """Isaac Lab's G1_29DOF_CFG drives as implicit drives (its explicit DC-motor path fails CUDA-graph capture here).

    Stiffer and stronger than the real robot (waist kp 5000, arms 3000 at 300 N·m): the first 60/50 runs trained on
    these. Kept so their checkpoints can be continued; new work uses implicit_actuators (Unitree's gains).
    """
    out = {}
    for name, act in G1_29DOF_CFG.actuators.items():
        if isinstance(act, DCMotorCfg):
            out[name] = ImplicitActuatorCfg(
                joint_names_expr=act.joint_names_expr, stiffness=act.stiffness, damping=act.damping,
                armature=act.armature, joint_effort_limit=act.actuator_effort_limit,
                joint_velocity_limit=act.actuator_velocity_limit,
            )
        else:
            out[name] = act
    return out


def rider_cfg(stance: dict | None = None, world_pins: dict | None = None, solver_iterations: tuple[int, int] = (32, 4),
              binding_angles: tuple[float, float] | None = None, stance_width: float | None = None,
              board_flex: bool = False, gains: str = "unitree", arm_gains: tuple[float, float] | None = None,
              **_amp_only) -> ArticulationCfg:
    """Same signature as envs.rider.rider_cfg; AMP-only options are ignored. gains: unitree (deployment) or isaac.
    arm_gains: (kp, kd) for the arm group only (the stock arms, kp 3000 / kd 10, are underdamped and wobble)."""
    cfg = physical.load()
    board, rider = dict(cfg["board"]), dict(cfg["rider"])
    if binding_angles:
        rider["binding_angles_deg"] = list(binding_angles)
    if stance_width:
        rider["stance_width"] = stance_width
    bindings = {FEET[foot]: pose for foot, pose in binding_frames(board, rider).items()}
    stock = G1_29DOF_CFG.spawn
    spawn = RiderSpawnCfg(
        robot="g1",
        usd_path=stock.usd_path,
        rigid_props=stock.rigid_props,
        articulation_props=[
            replace(a, solver_position_iteration_count=solver_iterations[0],
                    solver_velocity_iteration_count=solver_iterations[1])
            if hasattr(a, "solver_position_iteration_count") else a
            for a in stock.articulation_props
        ],
        copy_from_source=False,
        activate_contact_sensors=True,
        board_size=(board["length"], board["waist_width"], board["thickness"]),
        board_mass=board["mass"],
        bindings=bindings,
        foot_sole=FOOT_SOLE,
        world_pins=world_pins or {},
        board_flex=board_flex,
        effective_edge=board["effective_edge"],
    )
    if stance is None:
        init = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8), joint_pos={".*": 0.0})
    else:
        joints = dict(stance["joint_pos"])
        if board_flex:
            from .rider import camber_angle

            joints |= {"board_flex:0": 0.0, "board_flex:1": -camber_angle(board["effective_edge"])}
        init = ArticulationCfg.InitialStateCfg(
            pos=tuple(stance["pelvis_pos"]), rot=tuple(stance["pelvis_quat"]), joint_pos=joints,
        )
    actuators = isaac_actuators() if gains == "isaac" else implicit_actuators()
    if arm_gains:
        actuators["arms"] = actuators["arms"].replace(stiffness=arm_gains[0], damping=arm_gains[1])
    return replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot", spawn=spawn, init_state=init, actuators=actuators)


def stance_guess(knee: float, hip_y: float, ankle_y: float, twist: float, angles, open_deg: float) -> dict:
    """make_stance's first guess: a squat, each foot yawed by the hip from the pelvis to its binding angle."""
    front, back = angles if angles else physical.load()["rider"]["binding_angles_deg"]
    return {".*_knee_joint": knee, ".*_hip_pitch_joint": hip_y, ".*_ankle_pitch_joint": ankle_y,
            "left_hip_yaw_joint": math.radians(front - open_deg), "right_hip_yaw_joint": math.radians(back - open_deg),
            "waist_yaw_joint": twist}
