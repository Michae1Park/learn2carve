"""Check the snow model (envs/snow.py) against textbook formulas and save docs/snow/validation.png.

    scripts/py scripts/validate_snow.py

No Isaac Sim: a planar rigid board (rider + board mass) is driven by snow_wrench + air_drag only.

    1. Turn radius vs edge angle    sim  vs  max(R_sc · cos θ,  v² / (μ(θ) · g))      sidecut arc, or the grip limit
    2. Terminal speed vs slope      sim  vs  sqrt(2 m g (sin α − μ_g cos α) / (ρ C_d A))
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

from envs.snow import SnowParams, smoothstep, air_drag, snow_wrench  # noqa: E402
from tests.test_snow import DT, UP, quat_yaw_roll, simulate  # noqa: E402

G = 9.81
OUT = ROOT / "docs/snow/validation.png"


def expected_radius(theta_deg: float, v: float, p: SnowParams) -> float:
    mu = p.mu_flat + (p.mu_edge - p.mu_flat) * smoothstep(torch.tensor(math.radians(theta_deg)), 0.0,
                                                           math.radians(p.edge_lock_deg)).item()
    return max(p.sidecut_radius * math.cos(math.radians(theta_deg)), v**2 / (mu * G))


def terminal_speed_sim(slope_deg: float, p: SnowParams, seconds: float = 60.0) -> float:
    """Flat board pointing down the fall line, gravity's along-slope part pulling it."""
    a = math.radians(slope_deg)
    m = p.system_mass
    vel = torch.tensor([[1.0, 0.0, 0.0]])
    normal_force = torch.tensor([m * G * math.cos(a)])
    gravity = torch.tensor([[m * G * math.sin(a), 0.0, 0.0]])
    for _ in range(int(seconds / DT)):
        f, _, _ = snow_wrench(quat_yaw_roll(0.0, 0.0), vel, torch.zeros(1, 3), normal_force, UP, DT, p)
        vel = vel + (f + air_drag(vel, p) + gravity) / m * DT
    return vel.norm().item()


def terminal_speed_formula(slope_deg: float, p: SnowParams) -> float:
    a = math.radians(slope_deg)
    drive = p.system_mass * G * (math.sin(a) - p.mu_glide * math.cos(a))
    return math.sqrt(max(drive, 0.0) * 2 / (p.air_density * p.cd_area))


def main():
    p = SnowParams()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    thetas = [10, 15, 20, 25, 30, 35, 40, 45]
    for v, color in [(6.0, "#1f77b4"), (10.0, "#2ca02c"), (14.0, "#d62728")]:
        fine = [t / 2 for t in range(16, 92)]
        ax1.plot(fine, [expected_radius(t, v, p) for t in fine], color=color, lw=1.5, alpha=0.6)
        sim = [simulate(t, speed=v)[0] for t in thetas]
        ax1.scatter(thetas, sim, color=color, zorder=3, label=f"{v:.0f} m/s")
        print(f"[snow] v={v:4.1f}  " + "  ".join(f"{t}°:{r:5.1f}/{expected_radius(t, v, p):5.1f}"
                                             for t, r in zip(thetas, sim)))
    ax1.set(xlabel="edge angle θ [°]", ylabel="turn radius [m]", ylim=(0, 18),
            title="Turn radius: sim (dots) vs formula (lines)")
    ax1.text(44, 4.2, "follows the sidecut: R·cos θ", ha="right", fontsize=9, color="0.3")
    ax1.text(44, 14.4, "edge at its grip limit: v² / μg", ha="right", fontsize=9, color="0.3")
    ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), title="speed", frameon=False, ncol=3)

    slopes = [5, 8, 10, 12, 15, 20, 25]
    fine = [s / 2 for s in range(6, 52)]
    ax2.plot(fine, [terminal_speed_formula(s, p) for s in fine], color="0.4", lw=1.5, label="formula")
    sim = [terminal_speed_sim(s, p) for s in slopes]
    ax2.scatter(slopes, sim, color="#ff7f0e", zorder=3, label="sim")
    print("[snow] terminal  " + "  ".join(f"{s}°:{v:4.1f}/{terminal_speed_formula(s, p):4.1f}"
                                          for s, v in zip(slopes, sim)))
    ax2.set(xlabel="slope [°]", ylabel="terminal speed [m/s]", title="Straight run: terminal speed")
    ax2.legend(loc="upper left", frameon=False)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"[snow] -> {OUT}")


if __name__ == "__main__":
    main()
