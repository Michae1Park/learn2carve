"""Helpers shared by train.py and play.py."""

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOG_ROOT = ROOT / "logs/playground/quadruped"
sys.path.insert(0, str(ROOT))  # so `playground.quadruped.env` imports without scripts/py


def parse_overrides(pairs: list[str]) -> dict:
    """["rew_gait=2", "cmd_vx=[0.5,1]"] -> {"rew_gait": 2, "cmd_vx": (0.5, 1)}"""
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        value = yaml.safe_load(value)
        out[key] = tuple(value) if isinstance(value, list) else value
    return out


def make_env(gait: str, overrides: dict, num_envs: int, launched_sim_cfg):
    """Build the Go2 env. Call inside ``launch_simulation(launched_sim_cfg, args)``.

    launch_simulation writes the run's device and visualizers (e.g. the Kit viewport that --livestream needs)
    into the SimulationCfg it was given; copy them, or the env never renders.
    """
    from playground.quadruped.env import Go2Env, Go2EnvCfg

    cfg = Go2EnvCfg(gait=gait)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Go2EnvCfg has no field {key!r}")
        setattr(cfg, key, value)
    cfg.scene.num_envs = num_envs
    cfg.sim.device = launched_sim_cfg.device
    cfg.sim.visualizer_cfgs = launched_sim_cfg.visualizer_cfgs
    return Go2Env(cfg)


def load_agent_cfg(algo: str) -> dict:
    return yaml.safe_load((HERE / f"{algo}.yaml").read_text())
