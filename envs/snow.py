"""Board–snow force model for groomed hardpack. PhysX sees a frictionless ground; this supplies the rest.

Glide: friction along the board. Sidecut: on edge, the board's yaw rate is pulled toward the rate of its sidecut
arc, opened up when that arc needs more grip than the edge has. Grip: a stick-slip spring across the board, limited
to μ_lat(θ) · N: it carves while the spring holds and skids past it.

Batched torch, world frame, quaternions (x, y, z, w). Board x = nose, z = out of the base. Edge angle θ > 0 rolls
the board about +x so the -y edge digs in (the toe edge for a regular rider).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

STEER_MIN_COS = 0.2
GLIDE_SIGN_SOFTNESS = 0.1  # m/s


@dataclass
class SnowParams:
    mu_glide: float = 0.05
    mu_flat: float = 0.15
    mu_edge: float = 1.5
    edge_lock_deg: float = 15.0  # grip reaches mu_edge here
    min_edge_deg: float = 3.0  # no sidecut steering below this
    sidecut_radius: float = 7.6  # m
    k_yaw: float = 50.0  # N·m·s
    edge_length: float = 1.14  # m
    cd_area: float = 0.5  # m²
    air_density: float = 1.0  # kg/m³
    # Grip acts on the light board alone (~7 kg with feet) at 120 Hz: stiffer than this shakes the rider off.
    grip_stiffness: float = 2.0e4  # N/m
    grip_damping: float = 800.0  # N·s/m
    system_mass: float = 67.7  # kg, rider + board
    min_normal: float = 1.0  # N, airborne below this
    sidecut_tip_radius: float = 10.7  # m
    camber_flat_load: float = 330.0  # N
    flex_ei: float = 1500.0  # N·m²

    @classmethod
    def from_config(cls, snow: dict, board: dict, rider_mass: float) -> SnowParams:
        known = {k: v for k, v in snow.items() if k in cls.__dataclass_fields__}
        return cls(
            **known,
            system_mass=rider_mass + board["mass"],
            sidecut_radius=board["sidecut_radius_eff"],
            edge_length=board["effective_edge"],
        )

    def lateral_mu(self, abs_edge: torch.Tensor) -> torch.Tensor:
        return self.mu_flat + (self.mu_edge - self.mu_flat) * smoothstep(abs_edge, 0.0, math.radians(self.edge_lock_deg))

    def steering_engaged(self, abs_edge: torch.Tensor) -> torch.Tensor:
        min_edge = math.radians(self.min_edge_deg)
        return smoothstep(abs_edge, min_edge, 2.0 * min_edge)

    def max_yaw_torque(self, mu: torch.Tensor, normal_force: torch.Tensor) -> torch.Tensor:
        return mu * normal_force * self.edge_length / 4.0


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = ((x - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1)


def board_frame(quat: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(long axis on the snow, normal × long axis, edge angle)."""
    def board_axis(i: int) -> torch.Tensor:
        unit = quat.new_zeros(3)
        unit[i] = 1.0
        return rotate(quat, unit.expand_as(quat[..., :3]))

    nose, heel_side, base_normal = board_axis(0), board_axis(1), board_axis(2)
    along = nose - dot(nose, normal).unsqueeze(-1) * normal
    along = along / along.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    across = torch.linalg.cross(normal, along, dim=-1)
    edge_angle = torch.atan2(dot(heel_side, normal), dot(base_normal, normal))
    return along, across, edge_angle


def air_drag(lin_vel: torch.Tensor, p: SnowParams) -> torch.Tensor:
    """Apply to the body, not the board: pulled through the feet it pitches the rider forward."""
    return -0.5 * p.air_density * p.cd_area * lin_vel.norm(dim=-1, keepdim=True) * lin_vel


def glide_force(normal_force, v_long, along, p: SnowParams) -> torch.Tensor:
    return (-p.mu_glide * normal_force * torch.tanh(v_long / GLIDE_SIGN_SOFTNESS)).unsqueeze(-1) * along


def grip_force(v_lat, slip_z, limit, stiffness, damping, dt):
    """Lateral spring-damper that stops stretching at ±limit. Returns (force, needed force, new deflection)."""
    z = (torch.zeros_like(v_lat) if slip_z is None else slip_z) + v_lat * dt
    needed = -(stiffness * z + damping * v_lat)
    force = needed.clamp(-limit, limit)
    z = z.clamp(-limit / stiffness, limit / stiffness)
    return force, needed, z


def yaw_torque(omega_target, omega_n, tau_max, on_snow, p: SnowParams) -> torch.Tensor:
    return torch.maximum(torch.minimum(p.k_yaw * (omega_target - omega_n), tau_max), -tau_max) * on_snow


def snow_wrench(
    quat: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    normal_force: torch.Tensor,
    normal: torch.Tensor,
    dt: float,
    p: SnowParams,
    slip_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Force and torque on a rigid board's centre. Carry ``info["slip_z"]`` into the next call; zero it on reset."""
    normal = normal.expand_as(lin_vel)
    along, across, edge = board_frame(quat, normal)
    load = normal_force.clamp_min(0.0)
    on_snow = (load > p.min_normal).float()
    v_long, v_lat = dot(lin_vel, along), dot(lin_vel, across)
    abs_edge = edge.abs()
    mu = p.lateral_mu(abs_edge)

    radius = p.sidecut_radius * torch.cos(abs_edge).clamp_min(STEER_MIN_COS)
    a_needed = v_long.square() / radius
    a_max = mu * load / p.system_mass
    hold = (a_max / a_needed.clamp_min(1e-6)).clamp(max=1.0)
    omega_target = -torch.sign(edge) * v_long / radius * p.steering_engaged(abs_edge) * hold
    tau = yaw_torque(omega_target, dot(ang_vel, normal), p.max_yaw_torque(mu, load), on_snow, p)

    limit = mu * load
    f_lat, f_needed, z = grip_force(v_lat, slip_z, limit, p.grip_stiffness, p.grip_damping, dt)

    force = glide_force(load, v_long, along, p) + f_lat.unsqueeze(-1) * across
    info = {
        "edge_angle": edge,
        "v_long": v_long,
        "v_lat": v_lat,
        "normal_force": load,
        "grip_usage": f_needed.abs() / limit.clamp_min(1e-6) * on_snow,
        "turn_hold": hold,
        "omega_target": omega_target,
        "slip_z": z,
    }
    return force, tau.unsqueeze(-1) * normal, info


def board_wrench(
    quat: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    normal_force: torch.Tensor,
    normal: torch.Tensor,
    dt: float,
    p: SnowParams,
    slip_z: torch.Tensor | None = None,
    camber_engaged: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Snow wrench on a two-half board (dim 1: nose, tail). Each half glides and grips on its own edge and load;
    the board steers as one arc of curvature

        κ = c / (R(x_p) · cos θ̄)  +  (N − N_c)₊ · L / (6 EI) · sin |θ̄|

    θ̄ = load-weighted edge, x_p = centre of pressure, c = camber_engaged (0 unpressed .. 1 flat), N_c = camber_flat_load.
    """
    normal = normal.expand_as(lin_vel)
    along, across, edges = board_frame(quat, normal)
    loads = normal_force.clamp_min(0.0)
    load = loads.sum(-1)
    airborne = load.unsqueeze(-1) <= p.min_normal
    on_snow = (~airborne.squeeze(-1)).float()
    weights = torch.where(airborne, torch.full_like(loads, 0.5), loads / load.clamp_min(1e-6).unsqueeze(-1))

    v_long_halves, v_lat_halves = dot(lin_vel, along), dot(lin_vel, across)
    limits = p.lateral_mu(edges.abs()) * loads
    f_lat, f_needed, z = grip_force(v_lat_halves, slip_z, limits, p.grip_stiffness / 2, p.grip_damping / 2, dt)
    force = glide_force(loads, v_long_halves, along, p) + f_lat.unsqueeze(-1) * across

    edge = (weights * edges).sum(-1)
    abs_edge = edge.abs()
    mu = p.lateral_mu(abs_edge)
    v_long = (weights * v_long_halves).sum(-1)
    pressure_shift = weights[..., 0] - weights[..., 1]
    sidecut_radius = p.sidecut_radius + (p.sidecut_tip_radius - p.sidecut_radius) * pressure_shift.square()
    camber = torch.ones_like(load) if camber_engaged is None else camber_engaged.clamp(0.0, 1.0)
    sidecut_curvature = camber / (sidecut_radius * torch.cos(abs_edge).clamp_min(STEER_MIN_COS))
    flex_curvature = (load - p.camber_flat_load).clamp_min(0.0) * p.edge_length / (6.0 * p.flex_ei) * torch.sin(abs_edge)
    curvature = sidecut_curvature + flex_curvature
    a_needed = v_long.square() * curvature
    hold = (mu * load / p.system_mass / a_needed.clamp_min(1e-6)).clamp(max=1.0)
    omega_target = -torch.sign(edge) * v_long * curvature * p.steering_engaged(abs_edge) * hold
    omega_n = (weights * dot(ang_vel, normal)).sum(-1)
    tau = yaw_torque(omega_target, omega_n, p.max_yaw_torque(mu, load), on_snow, p)
    torque_per_half = (tau / 2).unsqueeze(-1).unsqueeze(-1) * normal

    grip_usage = f_needed.abs() / limits.clamp_min(1e-6) * (loads > p.min_normal / 2).float()
    info = {
        "edge_angle": edge,
        "v_long": v_long,
        "v_lat": (weights * v_lat_halves).sum(-1),
        "normal_force": load,
        "grip_usage": grip_usage.max(-1).values,
        "turn_hold": hold,
        "omega_target": omega_target,
        "slip_z": z,
        "edge_halves": edges,
        "v_lat_halves": v_lat_halves,
        "grip_usage_halves": grip_usage,
        "load_nose": weights[..., 0],
        "curvature": curvature,
        "camber_engaged": camber,
    }
    return force, torque_per_half, info
