"""Rider = Isaac Lab's 28-DoF AMP humanoid + a snowboard, as one articulation.

    pelvis ── ... ── left_foot  ══ front binding (in the articulation tree) ══ board
                     right_foot ══ back binding (loop-closing constraint)  ══╝

Regular stance only, rider facing the -y (toe) edge. Board frame: x nose, y heel edge, z up out of the base.
A binding angle α (+ toward the nose) yaws the foot's toes to α - 90° in the board frame.
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

from . import physical

JACKET = (0.85, 0.07, 0.05)
OUTFIT = {
    "jacket": (JACKET, ("pelvis", "torso", "chest", "right_upper_arm", "left_upper_arm", "right_lower_arm",
                        "left_lower_arm")),
    "pants": ((0.03, 0.03, 0.035), ("right_thigh", "left_thigh", "right_shin", "left_shin")),
    "dark": ((0.03, 0.03, 0.03), ("head", "right_hand", "left_hand", "right_foot", "left_foot")),
}
BOARD_NAVY = (0.015, 0.03, 0.14)
BOARD_CYAN = (0.0, 0.62, 0.85)
BOARD_WHITE = (0.9, 0.93, 0.97)

FOOT_SOLE = (0.045, 0.0, -0.05)  # in the foot body frame
HUMANOID_HEIGHT = 1.612  # m, sole to top of head
CAMBER_HEIGHT = 0.005  # m
IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)
UNLIMITED_FORCE = 3.4028234663852886e38

TORSO_SPLIT_HEIGHT = 0.10  # m above the abdomen joint
CHEST_CHILDREN = ("head", "right_upper_arm", "left_upper_arm")

COMPLIANCE = {
    "hip": 1.0, "knee": 1.0, "ankle": 0.7,
    "abdomen": 0.35, "thorax": 0.35,
    "shoulder": 0.3, "elbow": 0.3,
    "neck": 0.4,
}
JOINT_GROUP = {
    "right_thigh": "hip", "left_thigh": "hip", "right_knee": "knee", "left_knee": "knee",
    "right_foot": "ankle", "left_foot": "ankle", "torso": "abdomen", "thorax": "thorax",
    "right_upper_arm": "shoulder", "left_upper_arm": "shoulder", "right_elbow": "elbow", "left_elbow": "elbow",
    "head": "neck",
}

KNEE_TWIST_DEG = 20.0
KNEE_TWIST_GAIN = 0.3

BOOT_RANGE_DEG = {"x": 5.0, "y": 10.0, "z": 5.0}  # roll, forward flex, twist about the stance angle
BOOT_GAIN = 3.0
ANKLE_DOF_PREFIX = {"right_foot": "right_ankle", "left_foot": "left_ankle"}


def rider_scale() -> float:
    return physical.load()["rider"]["height"] / HUMANOID_HEIGHT


def foot_sole(scale: float | None = None) -> tuple[float, float, float]:
    k = rider_scale() if scale is None else scale
    return tuple(k * v for v in FOOT_SOLE)


def camber_angle(effective_edge: float, height: float = CAMBER_HEIGHT) -> float:
    """Rest angle between the board halves that lifts the centre by `height` with both tips down."""
    return 4.0 * height / effective_edge


def yaw_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def binding_frames(board: dict, rider: dict) -> dict[str, tuple[tuple[float, float, float], tuple[float, ...]]]:
    """Each foot's sole pose (position, quat wxyz) in the board frame."""
    if rider["stance"] != "regular":
        raise NotImplementedError("Only a regular stance is modelled (mirror the clip for goofy riders).")
    half_width = rider["stance_width"] / 2
    front_deg, back_deg = rider["binding_angles_deg"]
    deck = board["thickness"] / 2
    return {
        "left_foot": ((board["setback"] + half_width, 0.0, deck), yaw_quat_wxyz(math.radians(front_deg - 90.0))),
        "right_foot": ((board["setback"] - half_width, 0.0, deck), yaw_quat_wxyz(math.radians(back_deg - 90.0))),
    }


# -- USD joint helpers --------------------------------------------------------------------------------------------

def lock_axes(prim, axes) -> None:
    from pxr import UsdPhysics

    for axis in axes:
        limit = UsdPhysics.LimitAPI.Apply(prim, axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)


def add_limited_drive(prim, axis: str, low: float, high: float, stiffness: float, damping: float,
                      target: float = 0.0) -> None:
    from pxr import UsdPhysics

    limit = UsdPhysics.LimitAPI.Apply(prim, axis)
    limit.CreateLowAttr(low)
    limit.CreateHighAttr(high)
    drive = UsdPhysics.DriveAPI.Apply(prim, axis)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(UNLIMITED_FORCE)
    drive.CreateTargetPositionAttr(target)


def define_d6_joint(stage, path: str, body0: str, body1: str, pos0=(0.0, 0.0, 0.0)):
    from pxr import Gf, Sdf, UsdPhysics

    prim = UsdPhysics.Joint.Define(stage, path).GetPrim()
    joint = UsdPhysics.Joint(prim)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(*IDENTITY_WXYZ))
    joint.CreateLocalRot1Attr(Gf.Quatf(*IDENTITY_WXYZ))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(False)
    return prim


def set_armature(prim, armature: float) -> None:
    from pxr import Sdf

    prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(armature)


def scale_drive_gains(drive, stiffness_scale: float, damping_scale: float) -> None:
    drive.GetStiffnessAttr().Set(drive.GetStiffnessAttr().Get() * stiffness_scale)
    drive.GetDampingAttr().Set(drive.GetDampingAttr().Get() * damping_scale)


def set_limit(prim, axis: str, low: float, high: float) -> None:
    from pxr import UsdPhysics

    limit = UsdPhysics.LimitAPI(prim, axis)
    limit.GetLowAttr().Set(low)
    limit.GetHighAttr().Set(high)


# -- humanoid edits ----------------------------------------------------------------------------------------------

def fix_hinge_drives(stage, root: str) -> None:
    """PhysX reads revolute drives from ``drive:angular``; the USD authors them as ``drive:X``, so knees fold."""
    from pxr import Usd, UsdPhysics

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdPhysics.RevoluteJoint) or not prim.HasAPI(UsdPhysics.DriveAPI, "X"):
            continue
        src, dst = UsdPhysics.DriveAPI(prim, "X"), UsdPhysics.DriveAPI.Apply(prim, "angular")
        for attr in ("Type", "Stiffness", "Damping", "MaxForce", "TargetPosition", "TargetVelocity"):
            getattr(dst, f"Create{attr}Attr")(getattr(src, f"Get{attr}Attr")().Get())


def scale_humanoid(stage, root: str, k: float) -> None:
    """Uniform size × k: link positions, joint anchors and every link's shapes. Masses are set later."""
    from pxr import Gf, UsdGeom, UsdPhysics

    for link in stage.GetPrimAtPath(root).GetChildren():
        if not link.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        for op in UsdGeom.Xformable(link).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                m = Gf.Matrix4d(op.Get())
                m.SetTranslateOnly(m.ExtractTranslation() * k)
                op.Set(m)
            elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(op.Get() * k)
        for part in link.GetChildren():
            if not part.IsA(UsdGeom.Xformable):
                continue
            xformable = UsdGeom.Xformable(part)
            order = xformable.GetOrderedXformOps()
            scale_op = xformable.AddScaleOp(opSuffix="rider")
            scale_op.Set(Gf.Vec3f(k, k, k))
            xformable.SetXformOpOrder([scale_op] + order)
    for joint in stage.GetPrimAtPath(f"{root}/joints").GetChildren():
        j = UsdPhysics.Joint(joint)
        for attr in (j.GetLocalPos0Attr(), j.GetLocalPos1Attr()):
            if attr.HasAuthoredValue():
                attr.Set(Gf.Vec3f(attr.Get()) * k)


def twist_knees(stage, root: str) -> None:
    """Hinge → D6 with rotX = flexion and rotZ = shin twist. PhysX can't couple twist to flexion, so it's always free."""
    from pxr import UsdPhysics

    for side in ("right", "left"):
        prim = stage.GetPrimAtPath(f"{root}/joints/{side}_knee")
        hinge = UsdPhysics.RevoluteJoint(prim)
        flex_low, flex_high = hinge.GetLowerLimitAttr().Get(), hinge.GetUpperLimitAttr().Get()
        old_drive = UsdPhysics.DriveAPI(prim, "angular")
        k, d = old_drive.GetStiffnessAttr().Get(), old_drive.GetDampingAttr().Get()
        prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
        prim.RemoveAPI(UsdPhysics.DriveAPI, "X")
        prim.SetTypeName("PhysicsJoint")
        lock_axes(prim, ("transX", "transY", "transZ", "rotY"))
        add_limited_drive(prim, "rotX", flex_low, flex_high, k, d)
        add_limited_drive(prim, "rotZ", -KNEE_TWIST_DEG, KNEE_TWIST_DEG, k * KNEE_TWIST_GAIN, d * KNEE_TWIST_GAIN)


def split_torso(stage, root: str, cfg: RiderSpawnCfg) -> None:
    """pelvis ─abdomen─ torso (now lumbar) ─thorax (roll, pitch)─ chest ─ neck, shoulders.

    The chest takes the torso's chest sphere and clavicles; neck and shoulders re-parent to it at the same anchors.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    split = TORSO_SPLIT_HEIGHT
    torso, chest = f"{root}/torso", f"{root}/chest"
    torso_xform = UsdGeom.Xformable(stage.GetPrimAtPath(torso)).GetLocalTransformation()
    chest_prim = UsdGeom.Xform.Define(stage, chest).GetPrim()
    UsdGeom.Xformable(chest_prim).AddTransformOp().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, split)) * torso_xform)
    UsdPhysics.RigidBodyAPI.Apply(chest_prim)
    UsdPhysics.MassAPI.Apply(chest_prim)
    # the chest sphere overlaps the pelvis's waist sphere; once neighbours, they now need an explicit filter
    UsdPhysics.FilteredPairsAPI.Apply(chest_prim).CreateFilteredPairsRel().SetTargets([Sdf.Path(f"{root}/pelvis")])

    def clavicle(side: float):
        rotation = Gf.Matrix3d(-0.12950, side * 0.98594, 0.10557, -side * 0.98594, -0.11669, -side * 0.11957,
                               -0.10557, -side * 0.11957, 0.98720)
        return ("capsule", 0.045, 0.0836, Gf.Matrix4d(rotation, Gf.Vec3d(-0.01142, side * 0.08698, 0.23321 - split)))

    chest_shapes = {
        "chest": ("sphere", 0.11, None, Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 0.12 - split))),
        "right_clavicle": clavicle(side=-1.0),
        "left_clavicle": clavicle(side=1.0),
    }
    lumbar_shapes = {"lumbar": ("sphere", 0.085, None, Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 0.03)))}

    def add_shapes(body: str, specs: dict) -> None:
        for kind in ("collisions", "visuals"):
            for name, (shape, radius, height, xform) in specs.items():
                path = f"{body}/{kind}_split/{name}"
                geom = UsdGeom.Sphere.Define(stage, path) if shape == "sphere" else UsdGeom.Capsule.Define(stage, path)
                geom.CreateRadiusAttr(radius)
                if shape == "capsule":
                    geom.CreateHeightAttr(height)
                    geom.CreateAxisAttr("X")
                geom.AddTransformOp().Set(xform)
                if kind == "collisions":
                    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
                    UsdGeom.Imageable(geom).CreatePurposeAttr("guide")
                else:
                    geom.CreateDisplayColorAttr([Gf.Vec3f(*JACKET)])

    for kind in ("collisions", "visuals"):
        stage.GetPrimAtPath(f"{torso}/{kind}").SetActive(False)
    add_shapes(chest, chest_shapes)
    add_shapes(torso, lumbar_shapes)

    for child in CHEST_CHILDREN:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"{root}/joints/{child}"))
        joint.GetBody0Rel().SetTargets([Sdf.Path(chest)])
        p = joint.GetLocalPos0Attr().Get()
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2] - split))

    thorax = define_d6_joint(stage, f"{root}/joints/thorax", torso, chest, pos0=(0, 0, split))
    lock_axes(thorax, ("transX", "transY", "transZ", "rotZ"))
    for axis, (low, high) in (("rotX", cfg.thorax_roll_deg), ("rotY", cfg.thorax_pitch_deg)):
        add_limited_drive(thorax, axis, low, high, cfg.thorax_stiffness, cfg.thorax_stiffness / 10.0)
    set_armature(thorax, cfg.thorax_armature)


def tune_drives(stage, root: str, cfg: RiderSpawnCfg) -> None:
    from pxr import UsdPhysics

    for joint, group in JOINT_GROUP.items():
        prim = stage.GetPrimAtPath(f"{root}/joints/{joint}")
        if not prim.IsValid():
            continue
        stiffness_scale = COMPLIANCE[group] if cfg.compliant_drives else 1.0
        if group == "neck":
            stiffness_scale *= cfg.neck_stiffness_scale
        if group in ("shoulder", "elbow"):
            stiffness_scale *= cfg.arm_stiffness_scale
        damping_scale = stiffness_scale * (cfg.spine_damping_scale if group in ("abdomen", "thorax") else 1.0)
        for axis in ("rotX", "rotY", "rotZ", "angular"):
            if prim.HasAPI(UsdPhysics.DriveAPI, axis):
                scale_drive_gains(UsdPhysics.DriveAPI(prim, axis), stiffness_scale, damping_scale)


def widen_ankles(stage, root: str, cfg: RiderSpawnCfg) -> None:
    for joint in ANKLE_DOF_PREFIX:
        prim = stage.GetPrimAtPath(f"{root}/joints/{joint}")
        for axis, half_range in cfg.ankle_range_deg.items():
            set_limit(prim, f"rot{axis.upper()}", -half_range, half_range)


def widen_torso_twist(stage, root: str, cfg: RiderSpawnCfg) -> None:
    set_limit(stage.GetPrimAtPath(f"{root}/joints/torso"), "rotZ", -cfg.torso_twist_deg, cfg.torso_twist_deg)


def fit_boots(stage, root: str, cfg: RiderSpawnCfg) -> None:
    from pxr import UsdPhysics

    for joint, dof in ANKLE_DOF_PREFIX.items():
        prim = stage.GetPrimAtPath(f"{root}/joints/{joint}")
        for axis, half_range in BOOT_RANGE_DEG.items():
            usd_axis = f"rot{axis.upper()}"
            stance_angle = cfg.boot_center_deg.get(f"{dof}_{axis}", 0.0)
            below, above = cfg.boot_range_deg.get(axis, (-half_range, half_range))
            set_limit(prim, usd_axis, stance_angle + below, stance_angle + above)
            drive = UsdPhysics.DriveAPI(prim, usd_axis)
            scale_drive_gains(drive, BOOT_GAIN, BOOT_GAIN)
            drive.GetTargetPositionAttr().Set(stance_angle)


def dress(stage, root: str) -> None:
    """The humanoid's visuals are instanced with their own materials: un-instance them and bind ours on top."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils.prims import bind_visual_material

    for name, (color, bodies) in OUTFIT.items():
        material = f"{root}/Looks/{name}"
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)
        look.func(material, look)
        for body in bodies:
            for kind in ("visuals", "visuals_split"):
                visuals = stage.GetPrimAtPath(f"{root}/{body}/{kind}")
                if not visuals.IsValid() or not visuals.IsActive():
                    continue
                visuals.SetInstanceable(False)
                bind_visual_material(str(visuals.GetPath()), material, stage=stage, stronger_than_descendants=True)


# -- board --------------------------------------------------------------------------------------------------------

ROBOT_FINISH = (0.045, 0.045, 0.05)  # near-black, satin


def darken(stage, root: str) -> None:
    """One dark material over every link's visuals (instanced meshes take it from their instance root)."""
    from pxr import Sdf, Usd, UsdShade

    material = UsdShade.Material.Define(stage, f"{root}/Looks/robot_finish")
    shader = UsdShade.Shader.Define(stage, f"{root}/Looks/robot_finish/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(ROBOT_FINISH)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.2)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for part in Usd.PrimRange(stage.GetPrimAtPath(root)):  # instance proxies are skipped; their roots are bound
        if part.GetName() == "visuals":
            UsdShade.MaterialBindingAPI.Apply(part).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def add_board_graphics(stage, body_at, length: float, width: float, thickness: float) -> None:
    """Stadium-shaped deck with stripes, tip rings and chevrons; no mass, no collision. body_at(x) → body path."""
    from pxr import Gf, UsdGeom

    count = [0]

    def next_path(x: float) -> str:
        count[0] += 1
        return f"{body_at(x)}/vis_{count[0]}"

    def box(x, y, z, sx, sy, sz, color, yaw_deg=0.0):
        geom = UsdGeom.Cube.Define(stage, next_path(x))
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        geom.AddRotateZOp().Set(yaw_deg)
        geom.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    def disc(x, y, z, radius, height, color):
        geom = UsdGeom.Cylinder.Define(stage, next_path(x))
        geom.CreateAxisAttr("Z")
        geom.CreateRadiusAttr(radius)
        geom.CreateHeightAttr(height)
        geom.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    tip_radius = width / 2
    straight = length - width
    tips = (straight / 2, -straight / 2)
    top = thickness / 2 + 0.0006
    base = -thickness / 2 - 0.0006
    paint = 0.0012

    box(0.0, 0.0, 0.0, straight, width, thickness, BOARD_NAVY)
    for tip in tips:
        disc(tip, 0.0, 0.0, tip_radius, thickness, BOARD_NAVY)

    for side in (1, -1):
        box(0.0, side * (tip_radius - 0.018), top, straight, 0.012, paint, BOARD_CYAN)
    box(0.0, 0.0, top, straight, 0.008, paint, BOARD_WHITE)

    box(0.0, 0.0, base, straight, 0.07, paint, BOARD_CYAN)
    for tip in tips:
        disc(tip, 0.0, base, tip_radius - 0.03, paint, BOARD_WHITE)

    for sign, tip in zip((1, -1), tips):
        disc(tip, 0.0, top, tip_radius - 0.02, paint, BOARD_CYAN)
        disc(tip, 0.0, top + 0.0004, tip_radius - 0.036, paint, BOARD_NAVY)
        disc(tip, 0.0, top + 0.0008, 0.018, paint, BOARD_WHITE)
        for k, inset in enumerate((0.20, 0.30, 0.40)):
            x = sign * (straight / 2 - inset - 0.05)
            color = BOARD_CYAN if k < 2 else BOARD_WHITE
            for side in (1, -1):
                box(x - sign * 0.012, side * 0.045, top, 0.105, 0.012, paint, color, yaw_deg=sign * side * 32.0)


def add_board_body(stage, path: str, mass: float, size, offset_x: float) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    body = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass)
    collider = UsdGeom.Cube.Define(stage, f"{path}/geometry")
    collider.CreateSizeAttr(1.0)
    collider.AddTranslateOp().Set(Gf.Vec3d(offset_x, 0.0, 0.0))
    collider.AddScaleOp().Set(Gf.Vec3f(*size))
    collider.CreateDisplayColorAttr([Gf.Vec3f(*BOARD_NAVY)])
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    collider.MakeInvisible()


def add_board_flex_joint(stage, root: str, nose: str, tail: str, cfg: RiderSpawnCfg) -> None:
    """rotX = torsion, rotY = camber/bend (rest: camber). Bend stiffness: camber_flat_load presses it flat, k φ₀ = F L / 4."""
    prim = define_d6_joint(stage, f"{root}/joints/board_flex", nose, tail)
    lock_axes(prim, ("transX", "transY", "transZ", "rotZ"))
    camber = camber_angle(cfg.effective_edge, cfg.camber_height)
    bend_per_rad = cfg.camber_flat_load * cfg.effective_edge / (4.0 * camber)
    twist_per_rad = math.degrees(cfg.torsion_stiffness)
    for axis, per_rad, target_deg, (low, high) in (
        ("rotX", twist_per_rad, 0.0, (-15.0, 15.0)),
        ("rotY", bend_per_rad, -math.degrees(camber), (-3.0, 3.0)),
    ):
        per_deg = math.radians(per_rad)
        add_limited_drive(prim, axis, low, high, per_deg, per_deg * 0.02, target_deg)
    set_armature(prim, 0.01)


def add_fixed_joint(stage, path: str, body0: str | None, body1: str, pos0, rot0, pos1, rot1, in_tree: bool) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    joint = UsdPhysics.FixedJoint.Define(stage, path)
    if body0 is not None:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(*rot0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(*rot1))
    joint.CreateExcludeFromArticulationAttr(not in_tree)


def add_board(stage, root: str, cfg: RiderSpawnCfg) -> None:
    """One rigid board at ``board``, or with board_flex a nose half at ``board`` and a tail half at ``board_tail``."""
    nose_path, tail_path = f"{root}/board", f"{root}/board_tail"
    length, width, thickness = cfg.board_size

    if cfg.board_flex:
        half_size = (length / 2, width, thickness)
        add_board_body(stage, nose_path, cfg.board_mass / 2, half_size, +length / 4)
        add_board_body(stage, tail_path, cfg.board_mass / 2, half_size, -length / 4)
        add_board_flex_joint(stage, root, nose_path, tail_path, cfg)
        body_at = lambda x: nose_path if x >= 0 else tail_path
    else:
        add_board_body(stage, nose_path, cfg.board_mass, cfg.board_size, 0.0)
        body_at = lambda x: nose_path
    add_board_graphics(stage, body_at, length, width, thickness)

    for i, (foot, (pos, rot)) in enumerate(cfg.bindings.items()):
        is_front = i == 0
        add_fixed_joint(stage, f"{root}/joints/{foot}_binding", f"{root}/{foot}", body_at(pos[0]),
                        cfg.foot_sole or foot_sole(cfg.scale_factor), IDENTITY_WXYZ, pos, rot, in_tree=is_front)

    for body, (pos, rot) in cfg.world_pins.items():
        add_fixed_joint(stage, f"{root}/joints/{body}_pin", None, f"{root}/{body}", pos, rot, (0.0, 0.0, 0.0),
                        IDENTITY_WXYZ, in_tree=False)


@clone
def spawn_rider(prim_path: str, cfg: RiderSpawnCfg, translation=None, orientation=None, **kwargs):
    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    stage = prim.GetStage()
    if cfg.robot != "amp":  # a stock robot: only the board is added, and a darker finish
        add_board(stage, prim_path, cfg)
        darken(stage, prim_path)
        return prim
    if abs(cfg.scale_factor - 1.0) > 1e-6:
        assert not cfg.thoracic, "the thoracic split uses stock torso dimensions"
        scale_humanoid(stage, prim_path, cfg.scale_factor)
    fix_hinge_drives(stage, prim_path)
    if cfg.knee_rotation:
        twist_knees(stage, prim_path)
    if cfg.thoracic:
        split_torso(stage, prim_path, cfg)
    tune_drives(stage, prim_path, cfg)
    if cfg.ankle_range_deg:
        widen_ankles(stage, prim_path, cfg)
    if cfg.torso_twist_deg:
        widen_torso_twist(stage, prim_path, cfg)
    if cfg.stiff_boots:
        fit_boots(stage, prim_path, cfg)
    add_board(stage, prim_path, cfg)
    dress(stage, prim_path)
    return prim


@configclass
class RiderSpawnCfg(UsdFileCfg):
    func: Callable | str = spawn_rider
    robot: str = "amp"  # amp: edited AMP humanoid; g1: stock Unitree G1 (envs/rider_g1.py)
    foot_sole: tuple[float, float, float] | None = None  # in the foot body frame; None: the AMP foot, scaled
    scale_factor: float = 1.0  # humanoid only; the board keeps its size
    board_size: tuple[float, float, float] = (1.52, 0.246, 0.008)
    board_mass: float = 2.7
    bindings: dict = {}
    world_pins: dict = {}  # {body: (position, quat wxyz)} welded to the world, for building stances
    board_flex: bool = False
    camber_height: float = CAMBER_HEIGHT
    camber_flat_load: float = 330.0  # N
    torsion_stiffness: float = 5.0  # N·m/deg
    effective_edge: float = 1.14  # m
    thoracic: bool = False
    thorax_roll_deg: tuple[float, float] = (-40.0, 40.0)
    thorax_pitch_deg: tuple[float, float] = (-40.0, 60.0)
    thorax_stiffness: float = 400.0  # per degree, like the USD's abdomen (600)
    thorax_armature: float = 0.025  # kg·m²
    compliant_drives: bool = False
    knee_rotation: bool = False
    stiff_boots: bool = False
    boot_range_deg: dict = {}  # {axis: (below, above)} the stance angle
    boot_center_deg: dict = {}  # {ankle dof: stance angle}
    ankle_range_deg: dict = {}  # {axis: half range} about 0
    torso_twist_deg: float = 0.0  # 0 = stock ±50°
    neck_stiffness_scale: float = 1.0
    arm_stiffness_scale: float = 1.0
    spine_damping_scale: float = 1.0


def initial_joint_positions(stance: dict, options: dict, effective_edge: float) -> dict:
    joints = dict(stance["joint_pos"])
    if options.get("knee_rotation"):
        for side in ("right", "left"):
            joints[f"{side}_knee:0"] = joints.pop(f"{side}_knee")
            joints[f"{side}_knee:2"] = 0.0
    if options.get("thoracic"):
        joints["thorax:.*"] = 0.0
    if options.get("board_flex"):
        joints |= {"board_flex:0": 0.0, "board_flex:1": -camber_angle(effective_edge)}
    return joints


def rider_cfg(stance: dict | None = None, world_pins: dict | None = None, solver_iterations: tuple[int, int] = (32, 4),
              binding_angles: tuple[float, float] | None = None, stance_width: float | None = None,
              **options) -> ArticulationCfg:
    """``stance`` (envs/assets/stances/) sets the default joint positions; ``options`` are RiderSpawnCfg fields.

    The two-binding loop needs a converged solver: below 32/4 position/velocity iterations the joints sag.
    """
    cfg = physical.load()
    board, rider = dict(cfg["board"]), dict(cfg["rider"])
    if binding_angles:
        rider["binding_angles_deg"] = list(binding_angles)
    if stance_width:
        rider["stance_width"] = stance_width
    ankle_stance_deg = {k: math.degrees(v) for k, v in (stance or {}).get("joint_pos", {}).items() if "ankle" in k}
    spawn = RiderSpawnCfg(
        usd_path=HUMANOID_28_CFG.spawn.usd_path,
        rigid_props=HUMANOID_28_CFG.spawn.rigid_props,
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
        boot_center_deg=ankle_stance_deg,
        **options,
    )
    if stance is None:
        init = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos={".*": 0.0})
    else:
        init = ArticulationCfg.InitialStateCfg(
            pos=tuple(stance["pelvis_pos"]), rot=tuple(stance["pelvis_quat"]),
            joint_pos=initial_joint_positions(stance, options, board["effective_edge"]),
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
