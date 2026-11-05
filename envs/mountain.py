"""Visual-only mountain around the physics slope: terrain, fenced piste, forest, sky and sun. Nothing here collides.

Layout is in slope coordinates (x downhill from the start, y across, z along the snow normal). Import after Isaac
Sim launched.
"""

from __future__ import annotations

import math
import random

import numpy as np

import isaaclab.sim as sim_utils

ROOT = "/World/Mountain"
FENCE_HALF_WIDTH = 20.0  # m
FENCE_X = (-10.0, 400.0)
POLE_SPACING = 4.0
POLE_HEIGHT = 1.6
NET_ROW_HEIGHTS = [0.3 + 0.2 * i for i in range(6)]
NET_STRAND = 0.02
PISTE_HALF_WIDTH = FENCE_HALF_WIDTH + 8.0
HEADWALL_X = -600.0
TERRAIN_X = (-1500.0, 1600.0)
TERRAIN_Y = 1500.0
TERRAIN_STEP = 6.0
MAX_HEIGHT = 900.0
ROCK_STEEPNESS = 0.8  # |∇h|
TREE_LINE = 220.0
TREE_MAX_STEEPNESS = 0.6
SKY_DISTANCE, SKY_HEIGHT = 2500.0, 5000.0

SNOW = (0.96, 0.97, 1.0)
ROCK = (0.12, 0.12, 0.14)
PINE = (0.04, 0.16, 0.08)
BARK = (0.25, 0.16, 0.10)
NET = (1.0, 0.42, 0.18)
POLE = (0.75, 0.15, 0.10)
SKY = (0.42, 0.80, 0.92)


def slope_to_world(slope_deg: float, x, y, z=0.0):
    a = math.radians(slope_deg)
    return (x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a))


def material(color, roughness=0.8):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Terrain:
    """Height h(x, y) above the slope plane: flat on the piste, a valley with a headwall, ridged noise on top."""

    NUM_WAVES = 24

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = self.NUM_WAVES
        self.directions = rng.uniform(0, 2 * np.pi, n)
        self.wavenumbers = 2 * np.pi / np.geomspace(900.0, 60.0, n)
        self.phases = rng.uniform(0, 2 * np.pi, n)
        self.amplitudes = np.geomspace(1.0, 0.08, n)

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        beyond_piste = np.abs(y) - PISTE_HALF_WIDTH
        above_headwall = HEADWALL_X - x
        valley = np.where(beyond_piste > 0, 0.2 * beyond_piste + 4e-4 * beyond_piste**2, 0.0)
        headwall = np.where(above_headwall > 0, 0.12 * above_headwall + 1.2e-4 * above_headwall**2, 0.0)
        base = valley + headwall
        rough_mask = np.maximum(smoothstep(beyond_piste, 0.0, 60.0), smoothstep(above_headwall, 0.0, 60.0))
        h = base + rough_mask * self.ridges(x, y) * (0.5 * base + 20.0)
        return MAX_HEIGHT * np.tanh(h / MAX_HEIGHT)

    def ridges(self, x, y):
        total = np.zeros_like(x)
        for direction, k, phase, amplitude in zip(self.directions, self.wavenumbers, self.phases, self.amplitudes):
            total += amplitude * (1 - np.abs(np.sin(k * (x * np.cos(direction) + y * np.sin(direction)) + phase)))
        return total / self.amplitudes.sum() - 0.5

    def steepness(self, x, y, eps=2.0):
        dx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps)
        dy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps)
        return np.hypot(dx, dy)


def add_mountain(slope_deg: float, seed: int = 0):
    from isaaclab.sim.utils.prims import set_prim_visibility
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    ground_grid = stage.GetPrimAtPath("/World/ground/Environment")
    if ground_grid.IsValid():
        set_prim_visibility(ground_grid, False)

    terrain = Terrain(seed)
    add_terrain_mesh(stage, slope_deg, terrain)
    add_fence(slope_deg)
    add_forest(slope_deg, terrain, random.Random(seed))
    add_sky()
    sun = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0, color=(1.0, 0.97, 0.9))
    sun.func(f"{ROOT}/sun", sun, orientation=sun_orientation(elevation_deg=35.0, azimuth_deg=60.0))


def add_terrain_mesh(stage, slope_deg: float, terrain: Terrain):
    """Snow and rock as two meshes with one plain material each: per-vertex colours don't render in RTX real-time."""
    from pxr import UsdGeom, UsdShade, Vt

    xs = np.arange(TERRAIN_X[0], TERRAIN_X[1] + 1e-6, TERRAIN_STEP)
    ys = np.arange(-TERRAIN_Y, TERRAIN_Y + 1e-6, TERRAIN_STEP)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    H = terrain.height(X, Y)
    rows, cols = X.shape

    dHdx, dHdy = np.gradient(H, TERRAIN_STEP, TERRAIN_STEP)
    normals = np.stack([-dHdx, -dHdy, np.ones_like(H)], -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    def to_vec3_array(x, y, z):
        return Vt.Vec3fArray.FromNumpy(np.stack(slope_to_world(slope_deg, x, y, z), -1).reshape(-1, 3).astype(np.float32))

    points = to_vec3_array(X, Y, H)
    world_normals = to_vec3_array(normals[..., 0], normals[..., 1], normals[..., 2])

    corner = np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]
    quads = np.stack([corner, corner + 1, corner + cols + 1, corner + cols], -1).reshape(-1, 4)
    steepness = np.hypot(dHdx, dHdy)
    face_steepness = (steepness[:-1, :-1] + steepness[1:, :-1] + steepness[:-1, 1:] + steepness[1:, 1:]).reshape(-1) / 4
    is_rock = face_steepness > ROCK_STEEPNESS

    for name, faces, color, roughness in (("snow", quads[~is_rock], SNOW, 0.7), ("rock", quads[is_rock], ROCK, 0.95)):
        mesh = UsdGeom.Mesh.Define(stage, f"{ROOT}/terrain/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 4, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
        mesh.CreateNormalsAttr(world_normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        look_path = f"{ROOT}/Looks/{name}"
        look = material(color, roughness)
        look.func(look_path, look)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(look_path)))


def add_tree(path: str, base, height: float):
    trunk_height, crown_height = 0.25 * height, 0.85 * height
    bx, by, bz = base
    trunk = sim_utils.CylinderCfg(radius=0.06 * height, height=trunk_height, visual_material=material(BARK))
    trunk.func(f"{path}/trunk", trunk, translation=(bx, by, bz + trunk_height / 2))
    crown = sim_utils.ConeCfg(radius=0.22 * height, height=crown_height, visual_material=material(PINE))
    crown.func(f"{path}/crown", crown, translation=(bx, by, bz + trunk_height * 0.6 + crown_height / 2))


def add_forest(slope_deg: float, terrain: Terrain, rng: random.Random, count: int = 1400):
    placed = 0
    for _ in range(20 * count):
        if placed == count:
            break
        x = rng.uniform(-300.0, 1200.0)
        y = rng.choice((-1, 1)) * (FENCE_HALF_WIDTH + 6.0 + rng.expovariate(1 / 140.0))
        h = float(terrain.height(x, y))
        if h > TREE_LINE or float(terrain.steepness(x, y)) > TREE_MAX_STEEPNESS:
            continue
        add_tree(f"{ROOT}/trees/t{placed}", slope_to_world(slope_deg, x, y, h), rng.uniform(7.0, 14.0))
        placed += 1


def add_fence(slope_deg: float):
    """Net strands instead of a sheet: real-time rendering ignores opacity, and strands read as netting."""
    a = math.radians(slope_deg)
    along_slope = (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
    pole = sim_utils.CylinderCfg(radius=0.05, height=POLE_HEIGHT, visual_material=material(POLE, 0.5))
    net = material(NET, 0.9)
    x0, x1 = FENCE_X
    W = FENCE_HALF_WIDTH
    t = NET_STRAND
    mid_height = POLE_HEIGHT / 2

    def add_pole(path, x, y):
        bx, by, bz = slope_to_world(slope_deg, x, y)
        pole.func(path, pole, translation=(bx, by, bz + mid_height))

    def add_strand(path, center, size, orientation=None):
        cfg = sim_utils.CuboidCfg(size=size, visual_material=net)
        cfg.func(path, cfg, translation=center, orientation=orientation)

    def add_upright(path, x, y):
        bx, by, bz = slope_to_world(slope_deg, x, y)
        add_strand(path, (bx, by, bz + mid_height), (t, t, 1.0))

    for side in (-1, 1):
        y, tag = side * W, f"side{side + 1}"
        for k in range(int((x1 - x0) / POLE_SPACING) + 1):
            add_pole(f"{ROOT}/fence/{tag}_pole{k}", x0 + k * POLE_SPACING, y)
        for i, h in enumerate(NET_ROW_HEIGHTS):
            add_strand(f"{ROOT}/fence/{tag}_row{i}", slope_to_world(slope_deg, (x0 + x1) / 2, y, h), (x1 - x0, t, t),
                       along_slope)
        for k in range(int(x1 - x0)):
            add_upright(f"{ROOT}/fence/{tag}_col{k}", x0 + k, y)

    bx, by, bz = slope_to_world(slope_deg, x0, 0.0)
    for k in range(1, int(2 * W / POLE_SPACING)):
        add_pole(f"{ROOT}/fence/top_pole{k}", x0, -W + k * POLE_SPACING)
    for i, h in enumerate(NET_ROW_HEIGHTS):
        add_strand(f"{ROOT}/fence/top_row{i}", (bx, by, bz + h), (t, 2 * W, t))
    for k in range(int(2 * W) + 1):
        add_strand(f"{ROOT}/fence/top_col{k}", (bx, -W + k, bz + mid_height), (t, t, 1.0))


def add_sky():
    sky = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), emissive_color=SKY)
    d = SKY_DISTANCE
    for name, center, size in (
        ("up", (-d, 0.0), (1.0, 2 * d, SKY_HEIGHT)),
        ("down", (d, 0.0), (1.0, 2 * d, SKY_HEIGHT)),
        ("left", (0.0, -d), (2 * d, 1.0, SKY_HEIGHT)),
        ("right", (0.0, d), (2 * d, 1.0, SKY_HEIGHT)),
    ):
        wall = sim_utils.CuboidCfg(size=size, visual_material=sky)
        wall.func(f"{ROOT}/sky/{name}", wall, translation=(center[0], center[1], 0.0))


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def sun_orientation(elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float, float]:
    tilt = math.radians(90.0 - elevation_deg)
    azimuth = math.radians(azimuth_deg)
    about_z = (0.0, 0.0, math.sin(azimuth / 2), math.cos(azimuth / 2))
    about_x = (math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2))
    return quat_multiply(about_z, about_x)
