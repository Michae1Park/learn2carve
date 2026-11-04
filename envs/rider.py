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
    "jacket": ((0.85, 0.07, 0.05), ("pelvis", "torso", "chest", "right_upper_arm", "left_upper_arm", "right_lower_arm",
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

    board_path = f"{root}/board"  # the whole board, or its nose half with board_flex
    tail_path = f"{root}/board_tail"
    length, width, thickness = cfg.board_size

    def board_body(path: str, mass: float, size, offset_x: float) -> None:
        body = UsdGeom.Xform.Define(stage, path).GetPrim()  # body frame at the board centre either way
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass)
        geom = UsdGeom.Cube.Define(stage, f"{path}/geometry")
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(offset_x, 0.0, 0.0))
        geom.AddScaleOp().Set(Gf.Vec3f(*size))
        geom.CreateDisplayColorAttr([Gf.Vec3f(*BOARD_COLOR)])
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

    if not cfg.board_flex:
        board_body(board_path, cfg.board_mass, cfg.board_size, 0.0)
    else:
        # two halves, nose (+x) and tail (-x), each a box of half the length; the joint between them at the centre
        half = (length / 2, width, thickness)
        board_body(board_path, cfg.board_mass / 2, half, +length / 4)
        board_body(tail_path, cfg.board_mass / 2, half, -length / 4)
        _author_board_spring(stage, root, board_path, tail_path, cfg)

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
        # front foot -> nose half (in the tree), back foot -> tail half (loop-closing); one rigid board without flex
        target = tail_path if cfg.board_flex and pos[0] < 0 else board_path
        fixed_joint(f"{foot}_binding", f"{root}/{foot}", target, FOOT_SOLE, identity, pos, rot, in_tree=(i == 0))

    for body, (pos, rot) in cfg.world_pins.items():  # stance building only: hold bodies still in the world
        fixed_joint(f"{body}_pin", None, f"{root}/{body}", pos, rot, (0.0, 0.0, 0.0), identity, in_tree=False)


CAMBER_HEIGHT = 0.005  # m, centre lift of the unloaded board (typical camber: 4–6 mm)


def camber_angle(effective_edge: float, height: float = CAMBER_HEIGHT) -> float:
    """Rest angle between the halves [rad] that lifts the board centre by `height` with both tips down."""
    return 4.0 * height / effective_edge


def _author_board_spring(stage, root: str, nose: str, tail: str, cfg: RiderSpawnCfg) -> None:
    """D6 joint at the board centre: rotX = torsion, rotY = camber/bend (rest: camber), everything else locked.

    Bend stiffness is set so camber_flat_load at the bindings presses the camber flat: k φ₀ = F L / 4.
    """
    from pxr import Gf, Sdf, UsdPhysics

    prim = UsdPhysics.Joint.Define(stage, f"{root}/joints/board_flex").GetPrim()  # DoFs: board_flex:0 / :1
    joint = UsdPhysics.Joint(prim)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(nose)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tail)])
    for attr in (joint.CreateLocalPos0Attr, joint.CreateLocalPos1Attr):
        attr(Gf.Vec3f(0.0, 0.0, 0.0))
    for attr in (joint.CreateLocalRot0Attr, joint.CreateLocalRot1Attr):
        attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(False)
    for axis in ("transX", "transY", "transZ", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)
    phi0 = camber_angle(cfg.effective_edge, cfg.camber_height)
    k_bend = cfg.camber_flat_load * cfg.effective_edge / (4.0 * phi0)  # N·m/rad
    k_twist = math.degrees(cfg.torsion_stiffness)  # N·m/deg -> N·m/rad
    # USD drives are per degree. Camber: the tail tip down relative to the nose = negative rotY.
    for axis, k, target, (lo, hi) in (("rotX", k_twist, 0.0, (-15.0, 15.0)),
                                      ("rotY", k_bend, -math.degrees(phi0), (-3.0, 3.0))):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr(lo)
        limit.CreateHighAttr(hi)
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(math.radians(k))
        drive.CreateDampingAttr(math.radians(k) * 0.02)
        drive.CreateMaxForceAttr(3.4028234663852886e38)
        drive.CreateTargetPositionAttr(target)
    prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.01)


SPLIT_Z = 0.10  # m above the abdomen joint: where the thoracic joint splits the torso into lumbar + chest
CHEST_CHILDREN = ("head", "right_upper_arm", "left_upper_arm")  # joints whose parent moves from torso to chest


def _split_torso(stage, root: str, cfg: RiderSpawnCfg) -> None:
    """Split the humanoid's one-piece torso into a lumbar segment and a chest, joined by a 2-DoF thoracic joint.

        pelvis ─abdomen (3)─ torso (now lumbar) ─thorax x/y (2)─ chest ─ neck, shoulders

    The chest takes the torso's chest sphere and clavicles; the lumbar segment gets a smaller sphere of its own.
    Neck and shoulder joints are re-parented to the chest at the same world position, so the stance is unchanged
    with the thorax at zero.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    torso, chest = f"{root}/torso", f"{root}/chest"
    torso_xf = UsdGeom.Xformable(stage.GetPrimAtPath(torso)).GetLocalTransformation()
    chest_prim = UsdGeom.Xform.Define(stage, chest).GetPrim()
    UsdGeom.Xformable(chest_prim).AddTransformOp().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, SPLIT_Z)) * torso_xf)
    UsdPhysics.RigidBodyAPI.Apply(chest_prim)
    UsdPhysics.MassAPI.Apply(chest_prim)
    # the chest sphere overlaps the pelvis's upper-waist sphere by ~3 cm; they were neighbours (no collision) before
    # the lumbar segment came between them, so filter the pair or PhysX pushes them apart every step
    UsdPhysics.FilteredPairsAPI.Apply(chest_prim).CreateFilteredPairsRel().SetTargets([Sdf.Path(f"{root}/pelvis")])

    # chest geometry: the torso's own shapes, shifted down by the split height (positions from humanoid_28.usd)
    clav_rot = Gf.Matrix3d(-0.12950, -0.98594, 0.10557, 0.98594, -0.11669, 0.11957, -0.10557, 0.11957, 0.98720)
    shapes = {
        "chest": ("sphere", 0.11, None, Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 0.12 - SPLIT_Z))),
        "right_clavicle": ("capsule", 0.045, 0.0836, Gf.Matrix4d(clav_rot, Gf.Vec3d(-0.01142, -0.08698, 0.23321 - SPLIT_Z))),
        "left_clavicle": ("capsule", 0.045, 0.0836, Gf.Matrix4d(
            Gf.Matrix3d(-0.12950, 0.98594, 0.10557, -0.98594, -0.11669, -0.11957, -0.10557, -0.11957, 0.98720),
            Gf.Vec3d(-0.01142, 0.08698, 0.23321 - SPLIT_Z))),
    }
    lumbar = {"lumbar": ("sphere", 0.085, None, Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 0.03)))}

    def add_shapes(body: str, specs: dict) -> None:
        for kind in ("collisions", "visuals"):
            for name, (shape, radius, height, xf) in specs.items():
                path = f"{body}/{kind}_split/{name}"
                geom = UsdGeom.Sphere.Define(stage, path) if shape == "sphere" else UsdGeom.Capsule.Define(stage, path)
                geom.CreateRadiusAttr(radius)
                if shape == "capsule":
                    geom.CreateHeightAttr(height)
                    geom.CreateAxisAttr("X")
                geom.AddTransformOp().Set(xf)
                if kind == "collisions":
                    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
                    UsdGeom.Imageable(geom).CreatePurposeAttr("guide")  # collision only, not drawn
                else:
                    geom.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.07, 0.05)])

    # the torso's original shapes now belong to the chest: hide and disable them on the torso
    for kind in ("collisions", "visuals"):
        stage.GetPrimAtPath(f"{torso}/{kind}").SetActive(False)
    add_shapes(chest, shapes)
    add_shapes(torso, lumbar)

    # re-parent neck and shoulders to the chest (same world anchor: shift their local position by the split)
    for child in CHEST_CHILDREN:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"{root}/joints/{child}"))
        joint.GetBody0Rel().SetTargets([Sdf.Path(chest)])
        p = joint.GetLocalPos0Attr().Get()
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2] - SPLIT_Z))

    # thoracic joint: D6 with only rotX (roll) and rotY (pitch) free, like the abdomen's spherical joint
    prim = UsdPhysics.Joint.Define(stage, f"{root}/joints/thorax").GetPrim()  # DoFs: thorax:0 roll, thorax:1 pitch
    joint = UsdPhysics.Joint(prim)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(torso)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(chest)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, SPLIT_Z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(False)
    for axis in ("transX", "transY", "transZ", "rotZ"):  # locked: low > high
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)
    for axis, (lo, hi) in (("rotX", cfg.thorax_roll_deg), ("rotY", cfg.thorax_pitch_deg)):
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr(lo)
        limit.CreateHighAttr(hi)
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(cfg.thorax_stiffness)
        drive.CreateDampingAttr(cfg.thorax_stiffness / 10.0)
        drive.CreateMaxForceAttr(3.4028234663852886e38)
        drive.CreateTargetPositionAttr(0.0)
    prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(cfg.thorax_armature)


# Compliant drives: stiffness × this per joint group (damping follows, so each joint keeps its damping ratio).
# Legs carry the body and stay stiff; the spine, arms and neck give, as a human's do. Stock = 1.0 everywhere.
COMPLIANT = {
    "hip": 1.0, "knee": 1.0, "ankle": 0.7,
    "abdomen": 0.35, "thorax": 0.35,  # a supple spine: it curls and twists under load
    "shoulder": 0.3, "elbow": 0.3,
    "neck": 0.4,
}
# USD joint prim -> group
JOINT_GROUP = {
    "right_thigh": "hip", "left_thigh": "hip", "right_knee": "knee", "left_knee": "knee",
    "right_foot": "ankle", "left_foot": "ankle", "torso": "abdomen", "thorax": "thorax",
    "right_upper_arm": "shoulder", "left_upper_arm": "shoulder", "right_elbow": "elbow", "left_elbow": "elbow",
    "head": "neck",
}


def _tune_drives(stage, root: str, cfg: RiderSpawnCfg) -> None:
    """Scale the USD's own PD gains per joint.

        stiffness × compliant group scale (if compliant_drives) × neck_stiffness_scale (neck only)
        damping   × the same, × spine_damping_scale (abdomen + thorax: anti-jerk)
    """
    from pxr import UsdPhysics

    for joint, group in JOINT_GROUP.items():
        prim = stage.GetPrimAtPath(f"{root}/joints/{joint}")
        if not prim.IsValid():
            continue
        k_scale = COMPLIANT[group] if cfg.compliant_drives else 1.0
        if group == "neck":
            k_scale *= cfg.neck_stiffness_scale
        d_scale = k_scale * (cfg.spine_damping_scale if group in ("abdomen", "thorax") else 1.0)
        for axis in ("rotX", "rotY", "rotZ", "angular"):
            if not prim.HasAPI(UsdPhysics.DriveAPI, axis):
                continue
            drive = UsdPhysics.DriveAPI(prim, axis)
            drive.GetStiffnessAttr().Set(drive.GetStiffnessAttr().Get() * k_scale)
            drive.GetDampingAttr().Set(drive.GetDampingAttr().Get() * d_scale)


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
            for kind in ("visuals", "visuals_split"):  # _split_torso's chest / lumbar shapes are "visuals_split"
                visuals = stage.GetPrimAtPath(f"{root}/{body}/{kind}")
                if not visuals.IsValid() or not visuals.IsActive():
                    continue
                visuals.SetInstanceable(False)
                bind_visual_material(str(visuals.GetPath()), material, stage=stage, stronger_than_descendants=True)


@clone
def spawn_rider(prim_path: str, cfg: RiderSpawnCfg, translation=None, orientation=None, **kwargs):
    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    stage = prim.GetStage()
    _fix_hinge_drives(stage, prim_path)
    if cfg.thoracic:
        _split_torso(stage, prim_path, cfg)
    _tune_drives(stage, prim_path, cfg)
    _author_board(stage, prim_path, cfg)
    _dress(stage, prim_path)
    return prim


@configclass
class RiderSpawnCfg(UsdFileCfg):
    func: Callable | str = spawn_rider
    board_size: tuple[float, float, float] = (1.52, 0.246, 0.008)
    board_mass: float = 2.7
    bindings: dict = {}
    world_pins: dict = {}
    """{body name: (position, quat wxyz)} to weld to the world. Only used to build the stance."""
    board_flex: bool = False  # two-half board: torsion + camber spring at the centre (see _author_board_spring)
    camber_height: float = CAMBER_HEIGHT
    camber_flat_load: float = 330.0  # N at the bindings that presses the camber flat
    torsion_stiffness: float = 5.0  # N·m per degree of twist between nose and tail halves
    effective_edge: float = 1.14  # m
    thoracic: bool = False  # split the torso: + 2-DoF thoracic joint (roll, pitch) between lumbar and chest
    thorax_roll_deg: tuple[float, float] = (-40.0, 40.0)
    thorax_pitch_deg: tuple[float, float] = (-40.0, 60.0)
    thorax_stiffness: float = 400.0  # USD units (per degree), like the abdomen's 600
    thorax_armature: float = 0.025  # kg·m², rotor inertia added to the thorax DoFs (the USD uses 0.025)
    compliant_drives: bool = False  # per-group stiffness (COMPLIANT) instead of the stock uniform, very stiff drives
    neck_stiffness_scale: float = 1.0  # neck PD gains × this
    spine_damping_scale: float = 1.0  # abdomen + thorax damping × this (anti-jerk)


def rider_cfg(stance: dict | None = None, world_pins: dict | None = None, solver_iterations: tuple[int, int] = (32, 4),
              **options) -> ArticulationCfg:
    """ArticulationCfg for the rider. ``stance`` (from envs/assets/stance.yaml) sets the default joint positions.

    solver_iterations: PhysX (position, velocity) iterations. options: RiderSpawnCfg fields (thoracic, neck / spine
    gains). Joints the stance file doesn't list (e.g. the thorax) start at 0.
    """
    cfg = config.load()
    board, rider = cfg["board"], cfg["rider"]
    spawn = RiderSpawnCfg(
        usd_path=HUMANOID_28_CFG.spawn.usd_path,
        rigid_props=HUMANOID_28_CFG.spawn.rigid_props,
        # The two-binding loop needs a converged solver. 4/0 (upstream): the rider folds at the ankles and tips
        # over standing still. 16/4: hips/ankles still sag ~0.05 rad. 32/4: within 0.02 rad, ~22% slower than 16/4.
        articulation_props=[
            replace(a, solver_position_iteration_count=solver_iterations[0],
                    solver_velocity_iteration_count=solver_iterations[1])
            if hasattr(a, "solver_position_iteration_count") else a
            for a in HUMANOID_28_CFG.spawn.articulation_props
        ],
        copy_from_source=False,
        activate_contact_sensors=True,
        board_size=(board["length"], board["waist_width"], board["thickness"]),
        board_mass=board["mass"],
        bindings=binding_frames(board, rider),
        world_pins=world_pins or {},
        **options,
    )
    init = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos={".*": 0.0})
    if stance is not None:
        init = ArticulationCfg.InitialStateCfg(
            pos=tuple(stance["pelvis_pos"]), rot=tuple(stance["pelvis_quat"]),
            joint_pos=dict(stance["joint_pos"]) | ({"thorax:.*": 0.0} if options.get("thoracic") else {})
            | ({"board_flex:0": 0.0, "board_flex:1": -camber_angle(board["effective_edge"])}
               if options.get("board_flex") else {}),
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
