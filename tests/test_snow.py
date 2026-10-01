"""Snow model checks without Isaac Sim: sign conventions, then the carve/skid behaviour on a planar integrator."""

import math

import pytest
import torch

from envs.snow import SnowParams, air_drag, board_frame, snow_wrench

UP = torch.tensor([0.0, 0.0, 1.0])
DT = 1.0 / 120.0


def quat_yaw_roll(yaw: float, roll: float) -> torch.Tensor:
    """(x, y, z, w) for yaw about world z, then roll about the board's long axis."""
    cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(roll / 2), math.sin(roll / 2)
    # q = q_yaw * q_roll
    return torch.tensor([[cy * sr, sy * sr, sy * cr, cy * cr]])


def test_edge_angle_sign_and_axes():
    t, l, theta = board_frame(quat_yaw_roll(0.0, math.radians(20)), UP.expand(1, 3))
    assert torch.allclose(t, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(l, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    assert theta.item() == pytest.approx(math.radians(20), abs=1e-5)


def test_flat_straight_run_only_glide_and_drag():
    p = SnowParams()
    v = torch.tensor([[10.0, 0.0, 0.0]])
    f, tau, info = snow_wrench(quat_yaw_roll(0, 0), v, torch.zeros(1, 3), torch.tensor([600.0]), UP, DT, p)
    assert f[0, 1].abs() < 1e-4 and tau.abs().max() < 1e-4
    assert f[0, 0].item() == pytest.approx(-p.mu_glide * 600.0 * math.tanh(100.0), rel=1e-4)
    assert air_drag(v, p)[0, 0].item() == pytest.approx(-0.5 * p.air_density * p.cd_area * 100.0, rel=1e-4)


def test_flat_board_resists_pivoting_up_to_edge_friction():
    p = SnowParams()
    spin = lambda w: snow_wrench(quat_yaw_roll(0, 0), torch.zeros(1, 3), torch.tensor([[0.0, 0.0, w]]),
                                 torch.tensor([600.0]), UP, DT, p)[1][0, 2].item()
    assert spin(0.2) == pytest.approx(-p.k_yaw * 0.2)  # slow pivot: damped
    assert spin(10.0) == pytest.approx(-p.mu_flat * 600.0 * p.edge_length / 4)  # fast: capped by friction


def test_toe_edge_turns_clockwise_when_riding_forward():
    _, _, info = snow_wrench(
        quat_yaw_roll(0, math.radians(30)), torch.tensor([[8.0, 0, 0]]), torch.zeros(1, 3),
        torch.tensor([600.0]), UP, DT, SnowParams(),
    )
    assert info["omega_target"].item() < 0  # toe edge (θ > 0) → turn toward -y → clockwise


def test_grip_limit_saturates_when_unloaded():
    p = SnowParams(cd_area=0.0)
    v = torch.tensor([[5.0, 2.0, 0.0]])
    f_on, _, _ = snow_wrench(quat_yaw_roll(0, math.radians(30)), v, torch.zeros(1, 3), torch.tensor([600.0]), UP, DT, p)
    f_air, _, _ = snow_wrench(quat_yaw_roll(0, math.radians(30)), v, torch.zeros(1, 3), torch.tensor([0.0]), UP, DT, p)
    assert f_on[0, 1].item() == pytest.approx(-p.mu_edge * 600.0, rel=1e-4)  # demand exceeds the limit
    assert abs(f_air[0, 1].item()) < 1.0  # airborne: no grip


def simulate(theta_deg: float, speed: float, seconds: float = 6.0) -> tuple[float, float, float, float]:
    """Planar rigid body (rider + board) on flat snow, edge angle held fixed.

    Returns (turn radius, max |v_lat| in the second half, max |v_lat| in the first second, final speed).
    """
    p = SnowParams(mu_glide=0.0, cd_area=0.0)
    m, inertia = p.system_mass, 6.0
    pos, vel = torch.zeros(1, 3), torch.tensor([[speed, 0.0, 0.0]])
    yaw, omega = 0.0, torch.zeros(1, 3)
    normal_force = torch.tensor([m * 9.81])
    steps = int(seconds / DT)
    yaws, slips, slip_z = [], [], None
    for _ in range(steps):
        q = quat_yaw_roll(yaw, math.radians(theta_deg))
        f, tau, info = snow_wrench(q, vel, omega, normal_force, UP, DT, p, slip_z)
        f = f + air_drag(vel, p)
        slip_z = info["slip_z"]
        vel = vel + f / m * DT
        vel[:, 2] = 0.0
        omega = omega + tau / inertia * DT
        yaw += omega[0, 2].item() * DT
        pos = pos + vel * DT
        yaws.append(omega[0, 2].item())
        slips.append(abs(info["v_lat"].item()))
    w = sum(yaws[steps // 2 :]) / (steps - steps // 2)
    v = vel.norm().item()
    radius = v / abs(w) if abs(w) > 1e-6 else math.inf
    return radius, max(slips[steps // 2 :]), max(slips[: int(1.0 / DT)]), v


@pytest.mark.parametrize("theta_deg", [10, 20, 30, 40])
def test_low_speed_carve_follows_sidecut(theta_deg):
    radius, slip, _, _ = simulate(theta_deg, speed=4.0)
    expected = SnowParams().sidecut_radius * math.cos(math.radians(theta_deg))
    assert radius == pytest.approx(expected, rel=0.05)
    assert slip < 0.2  # carving, not skidding


def test_high_speed_turn_opens_up_at_the_grip_limit():
    # 40° edge: sidecut arc R ≈ 5.8 m. At 15 m/s that arc needs v²/R ≈ 39 m/s² > μ_edge · g ≈ 15 m/s²,
    # so the edge can't hold it: the turn widens to v² / (μ_edge · g) instead of following the sidecut.
    p = SnowParams()
    radius, slip_late, _, v_end = simulate(40, speed=15.0)
    assert radius == pytest.approx(v_end**2 / (p.mu_edge * 9.81), rel=0.1)
    assert radius > 2 * p.sidecut_radius * math.cos(math.radians(40))
    assert slip_late < 0.5


def test_flat_base_does_not_steer():
    radius, *_ = simulate(0, speed=6.0)
    assert radius == math.inf or radius > 1e3
