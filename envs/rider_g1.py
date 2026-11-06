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
    """G1_29DOF_CFG's drives as PhysX implicit drives: the same stiffness, damping, armature, torque and speed limits.

    Isaac Lab's explicit DC-motor path fails CUDA-graph capture next to the binding loop, so the legs and feet run
    implicit too. Lost: the DC motor's torque-speed curve (saturation 180 / 80 N·m, above the 88-139 / 50 N·m limits
    except at high joint speed).
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
              board_flex: bool = False, **_amp_only) -> ArticulationCfg:
    """Same signature as envs.rider.rider_cfg; AMP-only options are ignored."""
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
    return replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot", spawn=spawn, init_state=init,
                   actuators=implicit_actuators())


def stance_guess(knee: float, hip_y: float, ankle_y: float, twist: float, angles, open_deg: float) -> dict:
    """make_stance's first guess: a squat, each foot yawed by the hip from the pelvis to its binding angle."""
    front, back = angles if angles else physical.load()["rider"]["binding_angles_deg"]
    return {".*_knee_joint": knee, ".*_hip_pitch_joint": hip_y, ".*_ankle_pitch_joint": ankle_y,
            "left_hip_yaw_joint": math.radians(front - open_deg), "right_hip_yaw_joint": math.radians(back - open_deg),
            "waist_yaw_joint": twist}
