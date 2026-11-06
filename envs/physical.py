"""Loads physical.yaml (SI units) once: rider, board and snow. Every env/script reads them from here."""

from __future__ import annotations

import functools
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent


@functools.cache
def load(path: str | None = None) -> dict:
    with open(path or HERE / "physical.yaml") as f:
        return yaml.safe_load(f)
