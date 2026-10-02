"""Rider = Isaac Lab's 28-DoF AMP humanoid + a snowboard, as one articulation (Stage 1).

The board is an extra link authored onto the humanoid at spawn time:

    pelvis ── ... ── left_foot ══(front binding, fixed, in the articulation tree)══ board
                     right_foot ══(back binding, fixed, loop-closing constraint)══╝

Regular stance only: left foot toward the nose, rider facing the -y (toe) edge of the board.

Board frame: x toward the nose, y toward the heel edge, z up out of the base, origin at the board centre.
Binding angle α (deg, + toward the nose) puts the foot's toe direction at yaw (α - 90°) in the board frame.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils.prims import clone
from isaaclab.utils import configclass, replace
from isaaclab_assets import HUMANOID_28_CFG

from . import config

# Outfit, by body name prefix: bright jacket and blue board stand out on white snow and against the sky
OUTFIT = {
    "jacket": ((0.85, 0.07, 0.05), ("pelvis", "torso", "right_upper_arm", "left_upper_arm", "right_lower_arm",
                                    "left_lower_arm")),
    "pants": ((0.03, 0.03, 0.035), ("right_thigh", "left_thigh", "right_shin", "left_shin")),
    "dark": ((0.03, 0.03, 0.03), ("head", "right_hand", "left_hand", "right_foot", "left_foot")),  # helmet, gloves, boots
}
BOARD_COLOR = (0.05, 0.30, 0.95)

# Sole centre of the humanoid's foot box, in the foot body frame (box centre (0.045, 0, -0.0225), half-height 0.0275)
FOOT_SOLE = (0.045, 0.0, -0.05)
HUMANOID_HEIGHT = 1.612  # m, sole to top of head in the zero pose


def _quat_wxyz_z(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def binding_frames(board: dict, rider: dict) -> dict[str, tuple[tuple[float, float, float], tuple[float, ...]]]:
    """Binding pose (position, quat wxyz) of each foot's sole frame in the board frame."""
    if rider["stance"] != "regular":
        raise NotImplementedError("Only a regular stance is modelled (mirror the clip for goofy riders).")
    half = rider["stance_width"] / 2
    front_deg, back_deg = rider["binding_angles_deg"]
    top = board["thickness"] / 2
    return {
        "left_foot": ((board["setback"] + half, 0.0, top), _quat_wxyz_z(math.radians(front_deg - 90.0))),
        "right_foot": ((board["setback"] - half, 0.0, top), _quat_wxyz_z(math.radians(back_deg - 90.0))),
    }


def _fix_hinge_drives(stage, root: str) -> None:
    """The USD authors its revolute (knee/elbow) drives as ``drive:X``, which PhysX ignores (revolute joints
    read ``drive:angular``), so those joints load with no motor and the knees fold. Copy X -> angular."""
    from pxr import Usd, UsdPhysics

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdPhysics.RevoluteJoint) or not prim.HasAPI(UsdPhysics.DriveAPI, "X"):
            continue
        src, dst = UsdPhysics.DriveAPI(prim, "X"), UsdPhysics.DriveAPI.Apply(prim, "angular")
        for attr in ("Type", "Stiffness", "Damping", "MaxForce", "TargetPosition", "TargetVelocity"):
            getattr(dst, f"Create{attr}Attr")(getattr(src, f"Get{attr}Attr")().Get())


def _author_board(stage, root: str, cfg: RiderSpawnCfg) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    board_path = f"{root}/board"
    board = UsdGeom.Xform.Define(stage, board_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(board)
    UsdPhysics.MassAPI.Apply(board).CreateMassAttr(cfg.board_mass)

    geom = UsdGeom.Cube.Define(stage, f"{board_path}/geometry")
    geom.CreateSizeAttr(1.0)
    geom.AddScaleOp().Set(Gf.Vec3f(*cfg.board_size))
    geom.CreateDisplayColorAttr([Gf.Vec3f(*BOARD_COLOR)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

    def fixed_joint(name: str, body0: str | None, body1: str, pos0, rot0, pos1, rot1, in_tree: bool) -> None:
        joint = UsdPhysics.FixedJoint.Define(stage, f"{root}/joints/{name}")
        if body0 is not None:
            joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
        joint.CreateLocalRot0Attr(Gf.Quatf(*rot0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
        joint.CreateLocalRot1Attr(Gf.Quatf(*rot1))
        joint.CreateExcludeFromArticulationAttr(not in_tree)

    identity = (1.0, 0.0, 0.0, 0.0)
    for i, (foot, (pos, rot)) in enumerate(cfg.bindings.items()):
        fixed_joint(f"{foot}_binding", f"{root}/{foot}", board_path, FOOT_SOLE, identity, pos, rot, in_tree=(i == 0))

    for body, (pos, rot) in cfg.world_pins.items():  # stance building only: hold bodies still in the world
        fixed_joint(f"{body}_pin", None, f"{root}/{body}", pos, rot, (0.0, 0.0, 0.0), identity, in_tree=False)


def _dress(stage, root: str) -> None:
    """Recolour the humanoid (it ships light grey, hard to see on snow). Its visuals are instanced with their own
    materials, so make them editable and bind ours over them."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils.prims import bind_visual_material

    for name, (color, bodies) in OUTFIT.items():
        material = f"{root}/Looks/{name}"
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)
        look.func(material, look)
        for body in bodies:
            visuals = stage.GetPrimAtPath(f"{root}/{body}/visuals")
            if not visuals.IsValid():
                continue
            visuals.SetInstanceable(False)
            bind_visual_material(str(visuals.GetPath()), material, stage=stage, stronger_than_descendants=True)


@clone
def spawn_rider(prim_path: str, cfg: RiderSpawnCfg, translation=None, orientation=None, **kwargs):
    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    _fix_hinge_drives(prim.GetStage(), prim_path)
    _author_board(prim.GetStage(), prim_path, cfg)
    _dress(prim.GetStage(), prim_path)
    return prim


@configclass
class RiderSpawnCfg(UsdFileCfg):
    func: Callable | str = spawn_rider
    board_size: tuple[float, float, float] = (1.52, 0.246, 0.008)
    board_mass: float = 2.7
    bindings: dict = {}
    world_pins: dict = {}
    """{body name: (position, quat wxyz)} to weld to the world. Only used to build the stance."""


def rider_cfg(stance: dict | None = None, world_pins: dict | None = None) -> ArticulationCfg:
    """ArticulationCfg for the rider. ``stance`` (from envs/assets/stance.yaml) sets the default joint positions."""
    cfg = config.load()
    board, rider = cfg["board"], cfg["rider"]
    spawn = RiderSpawnCfg(
        usd_path=HUMANOID_28_CFG.spawn.usd_path,
        rigid_props=HUMANOID_28_CFG.spawn.rigid_props,
        # The two-binding loop needs a converged solver. 4/0 (upstream): the rider folds at the ankles and tips
        # over standing still. 16/4: hips/ankles still sag ~0.05 rad. 32/4: within 0.02 rad, ~22% slower than 16/4.
        articulation_props=[
            replace(a, solver_position_iteration_count=32, solver_velocity_iteration_count=4)
            if hasattr(a, "solver_position_iteration_count") else a
            for a in HUMANOID_28_CFG.spawn.articulation_props
        ],
        copy_from_source=False,
        activate_contact_sensors=True,
        board_size=(board["length"], board["waist_width"], board["thickness"]),
        board_mass=board["mass"],
        bindings=binding_frames(board, rider),
        world_pins=world_pins or {},
    )
    init = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos={".*": 0.0})
    if stance is not None:
        init = ArticulationCfg.InitialStateCfg(
            pos=tuple(stance["pelvis_pos"]), rot=tuple(stance["pelvis_quat"]), joint_pos=dict(stance["joint_pos"])
        )
    return replace(
        HUMANOID_28_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn,
        init_state=init,
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=None, damping=None, joint_velocity_limit={".*": 100.0}
            ),
        },
    )
