"""Loads config.yaml (SI units) once. Every env/script reads parameters from here."""

from __future__ import annotations

import functools
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


@functools.cache
def load(path: str | None = None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)
