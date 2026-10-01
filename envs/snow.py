"""Board–snow force model for groomed hardpack (Stage 1, D-004).

PhysX sees a frictionless ground; this module supplies everything snow does to the board:

1. Glide     friction along the board (air drag is separate: ``air_drag``, applied to the rider's body)
2. Sidecut   on edge, the board's yaw rate is pulled toward v / (R · cos θ): it follows its sidecut arc,
             unless that arc needs more grip than the edge has. Then the turn opens up (radius v² / a_max).
             Flat, the target is 0: the edge's grip resists pivoting, up to the torque μ_lat · N · L / 4 that
             friction spread along the edge can give (easy to pivot flat, hard on edge, never free-spinning)
3. Grip      stick-slip ("bristle") friction across the board: a lateral spring + damper anchored where the edge
             bites, limited to μ_lat(θ) · N. Carve (no slip) while it holds, skid past it

Pure torch, batched over environments, all inputs and outputs in the world frame. Quaternions are (x, y, z, w),
matching Isaac Lab 3.0. The snow surface normal is passed explicitly, so the model doesn't care whether the
slope comes from tilted terrain or tilted gravity.

Sign convention: board x = toward the nose, z = up out of the base. Edge angle θ > 0 means the board is rolled
about +x so that the -y edge digs in. For a regular rider facing -y, that is the toe edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SnowParams:
    mu_glide: float = 0.05        # along-board friction (waxed base on hardpack 0.03–0.08)
    mu_flat: float = 0.15         # lateral grip with a flat base (skids easily)
    mu_edge: float = 1.5          # lateral grip fully on edge (sharp edge, hardpack)
    edge_lock_deg: float = 15.0   # edge angle at which grip reaches mu_edge
    min_edge_deg: float = 3.0     # below this there is no sidecut steering
    sidecut_radius: float = 7.6   # m, effective single radius
    k_yaw: float = 50.0           # N·m·s, gain pulling yaw rate toward the sidecut rate
    edge_length: float = 1.14     # m, effective edge in contact with the snow (caps the yaw torque)
    cd_area: float = 0.5          # m², C_d · A of rider + board
    air_density: float = 1.0      # kg/m³ (ski-resort altitude)
    grip_stiffness: float = 2.0e4 # N/m, lateral spring of the biting edge (holds a carve with zero steady slip)
    grip_damping: float = 800.0   # N·s/m, damps it. Grip acts on the board alone, so both must stay stable for the
                                  # light board end (~7 kg incl. feet) at 120 Hz. Sizing grip to cancel the whole
                                  # rider's sideslip per step (≈ m/dt) shakes the board and topples the rider.
    system_mass: float = 67.7     # kg, rider + board: the mass whose sideslip the grip force must stop
    min_normal: float = 1.0       # N, below this the board counts as airborne

    @classmethod
    def from_config(cls, snow: dict, board: dict, rider_mass: float) -> SnowParams:
        known = {k: v for k, v in snow.items() if k in cls.__dataclass_fields__}
        return cls(
            **known,
            system_mass=rider_mass + board["mass"],
            sidecut_radius=board["sidecut_radius_eff"],
            edge_length=board["effective_edge"],
        )


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by quaternions q (..., 4) in (x, y, z, w)."""
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def _smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = ((x - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def board_frame(quat: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Board heading on the snow surface.

    Returns (t, l, theta): t = unit long axis projected onto the surface, l = normal × t (toward the +y edge),
    theta = edge angle [rad].
    """
    n = normal
    ex = _rotate(quat, quat.new_tensor([1.0, 0.0, 0.0]).expand_as(quat[..., :3]))
    ey = _rotate(quat, quat.new_tensor([0.0, 1.0, 0.0]).expand_as(quat[..., :3]))
    ez = _rotate(quat, quat.new_tensor([0.0, 0.0, 1.0]).expand_as(quat[..., :3]))
    t = ex - (ex * n).sum(-1, keepdim=True) * n
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    l = torch.linalg.cross(n, t, dim=-1)
    theta = torch.atan2((ey * n).sum(-1), (ez * n).sum(-1))
    return t, l, theta


def air_drag(lin_vel: torch.Tensor, p: SnowParams) -> torch.Tensor:
    """Quadratic air drag [world frame] on rider + board, given the rider's (pelvis) velocity (N, 3).

    Apply it to the body, not the board: at speed it is the main force balancing gravity, and pulling it
    through the feet would pitch the rider forward.
    """
    return -0.5 * p.air_density * p.cd_area * lin_vel.norm(dim=-1, keepdim=True) * lin_vel


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
    """Force and torque [world frame] that the snow applies to the board's centre of mass.

    Args:
        quat: (N, 4) board orientation, (x, y, z, w).
        lin_vel: (N, 3) board linear velocity.
        ang_vel: (N, 3) board angular velocity.
        normal_force: (N,) contact normal force magnitude on the board (≥ 0).
        normal: (N, 3) unit snow-surface normal (a (3,) tensor is broadcast).
        dt: physics step [s].
        p: model parameters.
        slip_z: (N,) grip spring deflection from the previous step [m] (None: relaxed). Carry
            ``info["slip_z"]`` to the next call and zero it on reset.

    Returns:
        force (N, 3), torque (N, 3), and a dict of diagnostics (edge angle, sideslip, grip usage ...).
    """
    n = normal.expand_as(lin_vel)
    t, l, theta = board_frame(quat, n)
    N = normal_force.clamp_min(0.0)
    on_snow = (N > p.min_normal).float()

    v_long = (lin_vel * t).sum(-1)
    v_lat = (lin_vel * l).sum(-1)
    abs_theta = theta.abs()

    # 1. glide friction (smooth sign)
    f_glide = (-p.mu_glide * N * torch.tanh(v_long / 0.1)).unsqueeze(-1) * t

    # 2. sidecut steering: toe edge (θ > 0) turns toward -l, i.e. clockwise when riding nose-first.
    #    Capped so the arc never needs more lateral acceleration than the edge can hold.
    mu_lat = p.mu_flat + (p.mu_edge - p.mu_flat) * _smoothstep(abs_theta, 0.0, math.radians(p.edge_lock_deg))
    steer = _smoothstep(abs_theta, math.radians(p.min_edge_deg), 2.0 * math.radians(p.min_edge_deg))
    radius = p.sidecut_radius * torch.cos(abs_theta).clamp_min(0.2)
    a_needed = v_long.square() / radius
    a_max = mu_lat * N / p.system_mass
    hold = (a_max / a_needed.clamp_min(1e-6)).clamp(max=1.0)
    omega_target = -torch.sign(theta) * v_long / radius * steer * hold
    omega_n = (ang_vel * n).sum(-1)
    tau_max = mu_lat * N * p.edge_length / 4.0
    tau_yaw = (torch.maximum(torch.minimum(p.k_yaw * (omega_target - omega_n), tau_max), -tau_max) * on_snow)
    tau_yaw = tau_yaw.unsqueeze(-1) * n

    # 3. grip: lateral spring-damper; past μ_lat(θ) · N the spring stops stretching and the edge slides
    f_limit = mu_lat * N
    z = (torch.zeros_like(v_lat) if slip_z is None else slip_z) + v_lat * dt
    f_needed = -(p.grip_stiffness * z + p.grip_damping * v_lat)
    f_lat = f_needed.clamp(-f_limit, f_limit)
    z = z.clamp(-f_limit / p.grip_stiffness, f_limit / p.grip_stiffness)
    f_grip = f_lat.unsqueeze(-1) * l

    force = f_glide + f_grip
    info = {
        "edge_angle": theta,
        "v_long": v_long,
        "v_lat": v_lat,
        "normal_force": N,
        "grip_usage": f_needed.abs() / f_limit.clamp_min(1e-6) * on_snow,  # > 1: the edge is slipping (0 airborne)
        "turn_hold": hold,  # < 1 means the carve is wider than the sidecut (edge at its limit)
        "omega_target": omega_target,
        "slip_z": z,
    }
    return force, tau_yaw, info
