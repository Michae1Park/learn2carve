"""Physical facts (SI units): rider, board, snow, piste, and the stance poses.

By default from envs/physical.yaml and envs/assets/stances/. A saved run carries its own copies; use(run_dir) points
every later load at them, so a replay rides the world it was trained in.
"""

from __future__ import annotations

import functools
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent
PHYSICAL_YAML = HERE / "physical.yaml"
STANCE_DIR = HERE / "assets/stances"

_active = {"physical": PHYSICAL_YAML, "stances": STANCE_DIR}


def use(run_dir: pathlib.Path):
    """Read physical.yaml and stance files from run_dir where it has them."""
    run_dir = pathlib.Path(run_dir)
    if (run_dir / "physical.yaml").exists():
        _active["physical"] = run_dir / "physical.yaml"
    if any(run_dir.glob("stance_*.yaml")):
        _active["stances"] = run_dir


def load() -> dict:
    return _read(_active["physical"])


def stance_path(name: str) -> pathlib.Path:
    """The run's copy of a stance file if it has one, else the repo's."""
    own = _active["stances"] / name
    return own if own.exists() else STANCE_DIR / name


@functools.cache
def _read(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text())
