"""Mountain around the tilted slope (CarveEnvCfg.mountain = True). Visual only: nothing here collides.

The physics is one infinite tilted plane (the scene's ground, envs/carve_env.py), and the rider only ever uses the
fenced piste on it. This module builds what you see:

- terrain: one mesh, flat (= the physics plane) across the piste, rising outside it into a valley whose walls
  steepen into rocky ridges, closed above the start by a headwall: a cirque, like a ski run below a summit.
  Snow where it's gentle, rock where it's steep (two meshes, one material each). The ground plane's checker grid is hidden.
- a fenced piste: orange safety netting on poles along both sides and across the top. Visual only, but the
  episode ends if the board crosses it (playground/snowboard/env.py), so the policy learns to turn inside it.
- pine forest on the lower valley sides, outside the fence (trees stand vertical, not perpendicular to the snow)
- a cyan sky (emissive walls far out; the renderer's own background stays near-white) and a low sun

Layout is in slope coordinates (x = downhill along the snow from the start, y = across, z = along the snow
normal) and placed in the world with the slope rotation. Riders start around x = 0 and go toward +x.
Import after Isaac Sim launched.
"""

from __future__ import annotations

import math
import random

import numpy as np

import isaaclab.sim as sim_utils

ROOT = "/World/Mountain"
FENCE_HALF_WIDTH = 20.0  # m: a 40 m piste, room for long turns (~±10 m swing) plus correction
FENCE_X = (-10.0, 400.0)  # m along the slope: just above the start, past the longest training episode
POLE_SPACING = 4.0  # m
PISTE_HALF_WIDTH = FENCE_HALF_WIDTH + 8.0  # m, the terrain is flat (on the physics plane) out to here
HEADWALL_X = -600.0  # m, the terrain rises into the headwall above this (far enough to leave sky above it)
TERRAIN_X = (-1500.0, 1600.0)
TERRAIN_Y = 1500.0
TERRAIN_STEP = 6.0  # m between mesh vertices
MAX_HEIGHT = 900.0  # m above the slope plane
ROCK_STEEPNESS = 0.8  # |∇h| above which a face is rock (~39° steeper than the slope)

SNOW = (0.96, 0.97, 1.0)
ROCK = (0.12, 0.12, 0.14)
PINE = (0.04, 0.16, 0.08)
BARK = (0.25, 0.16, 0.10)
NET = (1.0, 0.42, 0.18)
POLE = (0.75, 0.15, 0.10)
SKY = (0.42, 0.80, 0.92)  # emissive, the pale cyan of a clear winter sky


def slope_to_world(slope_deg: float, x, y, z=0.0):
    """Slope coordinates → world (rotation about +y by the slope angle; the slope passes through the origin).
    Works on floats and numpy arrays."""
    a = math.radians(slope_deg)
    return (x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a))


def _material(color, roughness=0.8):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Terrain:
    """Height above the slope plane h(x, y) [m] in slope coordinates."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        # ridged fractal noise: sum of |sin| waves in random directions, wavelengths 60-900 m
        n = 24
        self.dirs = rng.uniform(0, 2 * np.pi, n)
        self.freqs = 2 * np.pi / np.geomspace(900.0, 60.0, n)
        self.phases = rng.uniform(0, 2 * np.pi, n)
        self.amps = np.geomspace(1.0, 0.08, n)

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        d_side = np.abs(y) - PISTE_HALF_WIDTH  # distance outside the piste
        d_head = HEADWALL_X - x  # distance above the headwall's foot
        side = np.where(d_side > 0, 0.2 * d_side + 4e-4 * d_side**2, 0.0)  # gentle forested foot, steeper top
        head = np.where(d_head > 0, 0.12 * d_head + 1.2e-4 * d_head**2, 0.0)  # a low crest: sky above it from below
        base = side + head
        ridged = np.zeros_like(x)
        for th, f, p, a in zip(self.dirs, self.freqs, self.phases, self.amps):
            ridged += a * (1 - np.abs(np.sin(f * (x * np.cos(th) + y * np.sin(th)) + p)))
        # roughness grows with height, and fades to nothing at the piste so its edge stays smooth
        mask = np.maximum(_smoothstep(d_side, 0.0, 60.0), _smoothstep(d_head, 0.0, 60.0))
        ridged = ridged / self.amps.sum() - 0.5  # about -0.5..0.5
        h = base + mask * ridged * (0.5 * base + 20.0)
        return MAX_HEIGHT * np.tanh(h / MAX_HEIGHT)

    def steepness(self, x, y, eps=2.0):
        """|∇h|: 0 = parallel to the slope, 1 = 45° steeper than it."""
        dx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps)
        dy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps)
        return np.hypot(dx, dy)


def add_mountain(slope_deg: float, seed: int = 0):
    from isaaclab.sim.utils.prims import set_prim_visibility
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    grid = stage.GetPrimAtPath("/World/ground/Environment")
    if grid.IsValid():
        set_prim_visibility(grid, False)

    terrain = Terrain(seed)
    _add_terrain_mesh(stage, slope_deg, terrain)
    add_fence(slope_deg)
    _add_forest(slope_deg, terrain, random.Random(seed))
    _add_sky()
    sun = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0, color=(1.0, 0.97, 0.9))
    sun.func(f"{ROOT}/sun", sun, orientation=_sun_quat(elevation_deg=35.0, azimuth_deg=60.0))


def _add_terrain_mesh(stage, slope_deg: float, terrain: Terrain):
    """Two meshes over one vertex grid: snow faces and rock faces (steeper than ROCK_STEEPNESS), each with a plain
    material. (Per-vertex colours through a primvar reader don't render in RTX real-time.)"""
    from pxr import UsdGeom, UsdShade, Vt

    xs = np.arange(TERRAIN_X[0], TERRAIN_X[1] + 1e-6, TERRAIN_STEP)
    ys = np.arange(-TERRAIN_Y, TERRAIN_Y + 1e-6, TERRAIN_STEP)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    H = terrain.height(X, Y)
    nx, ny = X.shape

    # normals in slope coordinates from the height gradient, then rotated with the points
    dHdx, dHdy = np.gradient(H, TERRAIN_STEP, TERRAIN_STEP)
    N = np.stack([-dHdx, -dHdy, np.ones_like(H)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    px, py, pz = slope_to_world(slope_deg, X, Y, H)
    nxw, nyw, nzw = slope_to_world(slope_deg, N[..., 0], N[..., 1], N[..., 2])
    points = Vt.Vec3fArray.FromNumpy(np.stack([px, py, pz], -1).reshape(-1, 3).astype(np.float32))
    normals = Vt.Vec3fArray.FromNumpy(np.stack([nxw, nyw, nzw], -1).reshape(-1, 3).astype(np.float32))

    i = np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]
    quads = np.stack([i, i + 1, i + ny + 1, i + ny], -1).reshape(-1, 4)  # counter-clockwise seen from above
    steep = np.hypot(dHdx, dHdy)
    face_steep = (steep[:-1, :-1] + steep[1:, :-1] + steep[:-1, 1:] + steep[1:, 1:]).reshape(-1) / 4
    is_rock = face_steep > ROCK_STEEPNESS

    for name, faces, color in (("snow", quads[~is_rock], SNOW), ("rock", quads[is_rock], ROCK)):
        mesh = UsdGeom.Mesh.Define(stage, f"{ROOT}/terrain/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 4, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
        mesh.CreateNormalsAttr(normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        look = _material(color, 0.7 if name == "snow" else 0.95)
        look.func(f"{ROOT}/Looks/{name}", look)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath(f"{ROOT}/Looks/{name}")))


def _tree(path: str, base, height: float):
    trunk_h, crown_h = 0.25 * height, 0.85 * height
    bx, by, bz = base
    trunk = sim_utils.CylinderCfg(radius=0.06 * height, height=trunk_h, visual_material=_material(BARK))
    trunk.func(f"{path}/trunk", trunk, translation=(bx, by, bz + trunk_h / 2))
    crown = sim_utils.ConeCfg(radius=0.22 * height, height=crown_h, visual_material=_material(PINE))
    crown.func(f"{path}/crown", crown, translation=(bx, by, bz + trunk_h * 0.6 + crown_h / 2))


def _add_forest(slope_deg: float, terrain: Terrain, rng: random.Random, count: int = 1400):
    """Pines on the lower valley sides: outside the fence, below the tree line, not on rock."""
    placed, tries = 0, 0
    while placed < count and tries < 20 * count:
        tries += 1
        x = rng.uniform(-300.0, 1200.0)
        y = rng.choice((-1, 1)) * (FENCE_HALF_WIDTH + 6.0 + rng.expovariate(1 / 140.0))
        h = float(terrain.height(x, y))
        if h > 220.0 or float(terrain.steepness(x, y)) > 0.6:  # tree line, no trees on rock
            continue
        _tree(f"{ROOT}/trees/t{placed}", slope_to_world(slope_deg, x, y, h), rng.uniform(7.0, 14.0))
        placed += 1


def add_fence(slope_deg: float):
    """Orange safety net along both sides of the piste and across its top. Poles stand vertical; the net is a mesh
    of thin strands (real-time rendering ignores material opacity, and strands read as see-through netting)."""
    a = math.radians(slope_deg)
    tilt = (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
    pole = sim_utils.CylinderCfg(radius=0.05, height=1.6, visual_material=_material(POLE, 0.5))
    net = _material(NET, 0.9)
    x0, x1 = FENCE_X
    W = FENCE_HALF_WIDTH
    rows = [0.3 + 0.2 * i for i in range(6)]  # horizontal strands, 0.3-1.3 m above the snow
    t = 0.02  # strand thickness [m]

    def post(path, x, y):
        bx, by, bz = slope_to_world(slope_deg, x, y)
        pole.func(path, pole, translation=(bx, by, bz + 0.8))

    def strand(path, center, size, orientation=None):
        cfg = sim_utils.CuboidCfg(size=size, visual_material=net)
        cfg.func(path, cfg, translation=center, orientation=orientation)

    for side in (-1, 1):
        y, tag = side * W, f"side{side + 1}"
        for k in range(int((x1 - x0) / POLE_SPACING) + 1):
            post(f"{ROOT}/fence/{tag}_pole{k}", x0 + k * POLE_SPACING, y)
        for i, h in enumerate(rows):  # one long strand per row, laid along the slope
            strand(f"{ROOT}/fence/{tag}_row{i}", slope_to_world(slope_deg, (x0 + x1) / 2, y, h), (x1 - x0, t, t), tilt)
        for k in range(int(x1 - x0)):  # vertical strands every metre
            bx, by, bz = slope_to_world(slope_deg, x0 + k, y)
            strand(f"{ROOT}/fence/{tag}_col{k}", (bx, by, bz + 0.8), (t, t, 1.0))

    bx, by, bz = slope_to_world(slope_deg, x0, 0.0)
    for k in range(1, int(2 * W / POLE_SPACING)):
        post(f"{ROOT}/fence/top_pole{k}", x0, -W + k * POLE_SPACING)
    for i, h in enumerate(rows):
        strand(f"{ROOT}/fence/top_row{i}", (bx, by, bz + h), (t, 2 * W, t))
    for k in range(int(2 * W) + 1):
        strand(f"{ROOT}/fence/top_col{k}", (bx, -W + k, bz + 0.8), (t, t, 1.0))


def _add_sky():
    """Four emissive walls 2.5 km out, tall enough to fill the view above the ridges."""
    sky = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), emissive_color=SKY)
    d, h = 2500.0, 5000.0
    for name, center, size in (
        ("up", (-d, 0.0), (1.0, 2 * d, h)),
        ("down", (d, 0.0), (1.0, 2 * d, h)),
        ("left", (0.0, -d), (2 * d, 1.0, h)),
        ("right", (0.0, d), (2 * d, 1.0, h)),
    ):
        wall = sim_utils.CuboidCfg(size=size, visual_material=sky)
        wall.func(f"{ROOT}/sky/{name}", wall, translation=(center[0], center[1], 0.0))


def _sun_quat(elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) turning a distant light's -z beam to come from (azimuth, elevation)."""
    tilt = math.radians(90.0 - elevation_deg)  # tip the beam away from straight down
    az = math.radians(azimuth_deg)
    qz = (0.0, 0.0, math.sin(az / 2), math.cos(az / 2))
    qx = (math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2))
    # qz ∘ qx
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )
