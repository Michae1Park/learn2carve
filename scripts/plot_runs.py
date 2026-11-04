"""Learning curves (mean ± std over seeds) from the snowboard runs' TensorBoard logs, plus a markdown table.

    scripts/py scripts/plot_runs.py                       # main tasks      -> docs/results/learning_curves.png
    scripts/py scripts/plot_runs.py --set model           # baseline vs new model -> docs/results/model_curves.png

A group is every run in logs/playground/snowboard/ with a `seed` in run.yaml and exactly the group's task and
overrides. The table is the mean over each run's last 10% of training, then mean ± std over seeds.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs/playground/snowboard"
OUT = ROOT / "docs/results"

NEW_MODEL = {  # the new rider + board (docs/HUMANOID_MODEL.md, docs/BOARD_MODEL.md)
    "thoracic": True, "compliant_drives": True, "board_flex": True, "physics_hz": 240, "solver_iterations": [16, 4],
    "pelvis_mass_frac": 0.2, "neck_stiffness_scale": 0.5, "spine_damping_scale": 2,
}
# label: (task, overrides, colour)
SETS = {
    "main": {
        "Carve": ("carve", {}, "#d62728"),
        "Skid": ("skid", {}, "#2ca02c"),
        "Side slip": ("sideslip", {}, "#1f77b4"),
    },
    "model": {
        "Baseline model": ("carve", {}, "#7f7f7f"),
        "New model": ("carve", dict(NEW_MODEL), "#d62728"),
    },
}
METRICS = [  # tag, title, y label
    ("Episode_Snow/seconds", "Episode length (12 s max)", "s"),
    ("Episode_Termination/fell", "Falls", "fraction of episodes"),
    ("Episode_Snow/turns_per_10s", "Linked turns", "per 10 s"),
    ("Episode_Snow/err_slip_deg", "Slip error", "°"),
    ("Episode_Snow/err_speed", "Speed error", "m/s"),
]


def runs_for(task: str, overrides: dict) -> list[Path]:
    found = []
    for run_yaml in sorted(LOGS.glob("*/run.yaml")):
        run = yaml.safe_load(run_yaml.read_text())
        if "seed" in run and run["task"] == task and run["overrides"] == overrides:
            found.append(run_yaml.parent)
    return found


def load(run_dir: Path, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = ea.Tags()["scalars"]
    return {t: (np.array([s.step for s in ea.Scalars(t)]), np.array([s.value for s in ea.Scalars(t)]))
            for t in tags if t in have}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=list(SETS), default="main")
    args = parser.parse_args()
    groups = SETS[args.set]

    tags = [m[0] for m in METRICS]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.3 * len(METRICS), 3.0))
    table = []
    for label, (task, overrides, color) in groups.items():
        dirs = runs_for(task, overrides)
        if not dirs:
            print(f"[plot] no runs for {label}: {task} {overrides}", file=sys.stderr)
            continue
        data = [load(d, tags) for d in dirs]
        row = [label, str(len(dirs))]
        for ax, (tag, title, unit) in zip(axes, METRICS):
            curves = [d[tag] for d in data if tag in d]
            n = min(len(c[1]) for c in curves)
            steps = curves[0][0][:n] * 2048 / 1e6  # policy steps x envs = million samples
            ys = np.stack([c[1][:n] for c in curves])
            mean, std = ys.mean(0), ys.std(0)
            ax.plot(steps, mean, color=color, lw=1.8, label=label)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2, lw=0)
            ax.set(title=title, xlabel="samples [M]", ylabel=unit)
            final = ys[:, -max(n // 10, 1):].mean(1)
            row.append(f"{final.mean():.2f} ± {final.std():.2f}")
        table.append(row)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    name = {"main": "learning_curves", "model": "model_curves"}[args.set]
    fig.savefig(OUT / f"{name}.png", dpi=130)
    print(f"[plot] -> {OUT / name}.png\n")

    print("| Group | Seeds | " + " | ".join(m[1] for m in METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for row in table:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
