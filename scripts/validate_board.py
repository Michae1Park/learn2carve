"""Check the two-half board terms of the snow model (envs/snow.py board_wrench) and save docs/snow/board_validation.png.

    scripts/py scripts/validate_board.py

No Isaac Sim: a planar rigid board (both halves moving together, rider + board mass) is driven by board_wrench only,
with fixed edge angles and loads per half. Steady turn radius vs closed form:

    1. Flex: radius vs load             κ = 1/(R_sc cos θ) + (N − N_c)₊ · L/(6 EI) · sin θ
    2. Progressive sidecut: radius vs   R(x_p) cos θ,  R = R_sc + (R_tip − R_sc) · s²,  s = nose share − tail share
       centre of pressure
    3. Torsion: nose and tail at different edge angles: the arc follows the load-weighted edge angle θ̄
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.snow import SnowParams, board_wrench  # noqa: E402
from tests.test_snow import DT, UP, quat_yaw_roll  # noqa: E402

G = 9.81
OUT = ROOT / "docs/snow/board_validation.png"


def ride(theta_nose: float, theta_tail: float, speed: float, load: float, nose_share: float,
         seconds: float = 5.0, p: SnowParams | None = None) -> tuple[float, float]:
    """Planar board on flat snow, edge angles [deg] and load [N] held fixed. Returns (radius, mean sideways slip
    [m/s]) over the second half."""
    p = p or SnowParams(mu_glide=0.0, cd_area=0.0)
    m, inertia = p.system_mass, 6.0
    vel = torch.tensor([[speed, 0.0, 0.0]])
    yaw, omega = 0.0, 0.0
    normal = torch.tensor([[load * nose_share, load * (1 - nose_share)]])
    steps = int(seconds / DT)
    omegas, slips, slip_z = [], [], None
    for _ in range(steps):
        q = torch.cat([quat_yaw_roll(yaw, math.radians(theta_nose)), quat_yaw_roll(yaw, math.radians(theta_tail))])
        f, tau, info = board_wrench(q.unsqueeze(0), vel.unsqueeze(1).expand(1, 2, 3),
                                    torch.tensor([[[0.0, 0.0, omega]] * 2]), normal, UP, DT, p, slip_z)
        slip_z = info["slip_z"]
        vel = vel + f.sum(1) / m * DT
        vel[:, 2] = 0.0
        omega += tau.sum(1)[0, 2].item() / inertia * DT
        yaw += omega * DT
        omegas.append(omega)
        slips.append(info["v_lat_halves"][0].abs().mean().item())
    w = sum(omegas[steps // 2:]) / (steps - steps // 2)
    v = vel.norm().item()
    late = slips[steps // 2:]
    return v / abs(w) if abs(w) > 1e-6 else math.inf, sum(late) / len(late)


def flex_radius(theta: float, load: float, p: SnowParams) -> float:
    t = math.radians(theta)
    k = 1 / (p.sidecut_radius * math.cos(t)) + max(load - p.camber_flat_load, 0.0) * p.edge_length / (6 * p.flex_ei) * math.sin(t)
    return 1 / k


def progressive_radius(theta: float, nose_share: float, p: SnowParams) -> float:
    s = 2 * nose_share - 1
    return (p.sidecut_radius + (p.sidecut_tip_radius - p.sidecut_radius) * s * s) * math.cos(math.radians(theta))


def main():
    p = SnowParams(mu_glide=0.0, cd_area=0.0)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    mg = p.system_mass * G

    # 1. flex
    loads = [0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
    fine = [x / 20 for x in range(6, 42)]
    for theta, color in [(20, "#1f77b4"), (40, "#d62728")]:
        ax1.plot(fine, [flex_radius(theta, f * mg, p) for f in fine], color=color, lw=1.5, alpha=0.6)
        sim = [ride(theta, theta, 5.0, f * mg, 0.5, p=p)[0] for f in loads]
        ax1.scatter(loads, sim, color=color, zorder=3, label=f"θ = {theta}°")
        print(f"[board] flex θ={theta}: " + "  ".join(f"{f}mg:{r:.2f}/{flex_radius(theta, f * mg, p):.2f}"
                                                       for f, r in zip(loads, sim)))
    ax1.axvline(p.camber_flat_load / mg, color="0.5", ls="--", lw=1)
    ax1.text(p.camber_flat_load / mg + 0.03, 3.2, "camber\npressed flat", fontsize=9, color="0.3")
    ax1.set(xlabel="load on the board [× body weight]", ylabel="turn radius [m]", ylim=(3, 8),
            title="Flex: pressing tightens the arc")
    ax1.legend(frameon=False, loc="upper right")

    # 2. progressive sidecut
    shares = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    fine = [x / 50 for x in range(51)]
    theta = 20
    ax2.plot(fine, [progressive_radius(theta, s, p) for s in fine], color="#2ca02c", lw=1.5, alpha=0.6, label="formula")
    sim = [ride(theta, theta, 4.0, 0.45 * mg, s, p=p)[0] for s in shares]  # below the flex load: sidecut only
    ax2.scatter(shares, sim, color="#2ca02c", zorder=3, label="sim")
    print("[board] progressive: " + "  ".join(f"{s}:{r:.2f}/{progressive_radius(theta, s, p):.2f}"
                                               for s, r in zip(shares, sim)))
    ax2.set(xlabel="share of the load on the nose half", ylabel="turn radius [m]",
            title=f"Progressive sidecut (θ = {theta}°)")
    ax2.text(0.5, 7.25, "centred:\ntightest", ha="center", fontsize=9, color="0.3")
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.88))

    # 3. torsion: nose held at 35°, tail twisted flatter; the arc follows the load-weighted edge angle
    tails = [35, 30, 25, 20, 15, 10]
    fine = [t / 2 for t in range(16, 72)]
    nose = 35
    formula = lambda tail: p.sidecut_radius * math.cos(math.radians((nose + tail) / 2))
    ax3.plot(fine, [formula(t) for t in fine], color="#9467bd", lw=1.5, alpha=0.6, label="R_sc · cos(mean edge angle)")
    sim = [ride(nose, t, 4.0, 0.45 * mg, 0.5, p=p)[0] for t in tails]
    ax3.scatter(tails, sim, color="#9467bd", zorder=3, label="sim")
    print("[board] torsion tail: " + "  ".join(f"{t}:{r:.2f}/{formula(t):.2f}" for t, r in zip(tails, sim)))
    ax3.invert_xaxis()
    ax3.set(xlabel="tail edge angle [°] (nose at 35°)", ylabel="turn radius [m]",
            title="Torsion: twisting the tail flat opens the arc")
    ax3.legend(frameon=False, loc="upper left")

    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"[board] -> {OUT}")


if __name__ == "__main__":
    main()
