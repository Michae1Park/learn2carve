"""Helpers shared by the walk/ and backflip/ train and play scripts.

Safe to import before Isaac Sim launches (only `train` / `load_policy` / `make_env` touch Isaac, and they import
it lazily, so call them inside ``launch_simulation``).
"""

import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOG_ROOT = ROOT / "logs/playground/quadruped"
sys.path.insert(0, str(ROOT))  # so `playground.quadruped.*` imports without scripts/py


def add_train_args(parser, default_timesteps: int | None = None):
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--num_envs", type=int, default=None, help="default: from <algo>.yaml")
    parser.add_argument("--timesteps", type=int, default=default_timesteps,
                        help="env steps per env; default: " + (str(default_timesteps) if default_timesteps else "from <algo>.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="env cfg overrides, see env.py")
    parser.add_argument("--seed", type=int, default=42)
    add_suppress_arg(parser)


def add_play_args(parser):
    parser.add_argument("run", type=Path, help="run directory from train.py")
    parser.add_argument("--checkpoint", default="best_agent.pt", help="file in <run>/checkpoints/")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=10.0)
    add_suppress_arg(parser)


def add_suppress_arg(parser):
    parser.add_argument("--suppress_warnings", choices=["true", "false"], default="true",
                        help="hide the thousands of harmless 'Failed to find rigid body' warnings (default: true)")


def parse_overrides(pairs: list[str]) -> dict:
    """["rew_gait=2", "cmd_vx=[0.5,1]"] -> {"rew_gait": 2, "cmd_vx": (0.5, 1)}"""
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        value = yaml.safe_load(value)
        out[key] = tuple(value) if isinstance(value, list) else value
    return out


def load_agent_cfg(algo: str) -> dict:
    return yaml.safe_load((HERE / f"{algo}.yaml").read_text())


def new_run(args, task: str, **extra) -> tuple[Path, dict, dict]:
    """Create logs/playground/quadruped/<time>_<algo>_<task>/ with a run.yaml. Returns (run_dir, run, agent_cfg)."""
    agent_cfg = load_agent_cfg(args.algo)
    run_dir = LOG_ROOT / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.algo}_{task}"
    run_dir.mkdir(parents=True)
    run = {
        "task": task,
        "algo": args.algo,
        **extra,
        "overrides": parse_overrides(args.set),
        "num_envs": args.num_envs or agent_cfg["num_envs"],
        "timesteps": args.timesteps or agent_cfg["timesteps"],
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    return run_dir, run, agent_cfg


def make_env(env_cls, cfg, overrides: dict, num_envs: int, launched_sim_cfg, suppress_warnings: str = "true"):
    """Build an env from its cfg. Call inside ``launch_simulation(launched_sim_cfg, args)``.

    launch_simulation writes the run's device and visualizers (e.g. the Kit viewport that --livestream needs)
    into the SimulationCfg it was given; copy them, or the env never renders.
    """
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"{type(cfg).__name__} has no field {key!r}")
        setattr(cfg, key, value)
    cfg.scene.num_envs = num_envs
    cfg.sim.device = launched_sim_cfg.device
    cfg.sim.visualizer_cfgs = launched_sim_cfg.visualizer_cfgs
    if suppress_warnings == "true":
        quiet_physx_tensors()
    return env_cls(cfg)


def quiet_physx_tensors():
    """Hide the plugin's warnings (errors still show). Building the env makes it scan every prim on the stage and
    warn "Failed to find rigid body at '/World/...'" for each non-body: thousands of harmless lines."""
    import carb

    carb.settings.get_settings().set("/log/channels/omni.physx.tensors.plugin", "error")


def train(raw_env, agent_cfg: dict, run_dir: Path, run: dict, seed: int):
    """Train with skrl until `run["timesteps"]` (Ctrl+C stops early) and save a final checkpoint."""
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils import set_seed
    from skrl.utils.runner.torch import Runner

    timesteps = run["timesteps"]
    agent_cfg["seed"] = seed
    agent_cfg["agent"]["experiment"] = {"directory": str(LOG_ROOT), "experiment_name": run_dir.name}
    agent_cfg["trainer"]["timesteps"] = timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    set_seed(seed)
    env = SkrlVecEnvWrapper(raw_env)
    runner = Runner(env, agent_cfg)
    print(f"[train] {run['algo']} / {run['task']} / {run['num_envs']} envs / {timesteps} steps -> {run_dir}")
    try:
        runner.run()
    except KeyboardInterrupt:
        print("[train] interrupted")
    runner.agent.write_checkpoint(timestep=timesteps, timesteps=timesteps)
    env.close()


def load_policy(raw_env, run: dict, run_dir: Path, checkpoint: str):
    """Wrap the env for skrl and load a trained agent in eval mode. Returns (env, agent)."""
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils.runner.torch import Runner

    env = SkrlVecEnvWrapper(raw_env)
    agent_cfg = load_agent_cfg(run["algo"])
    agent_cfg["agent"]["experiment"] = {"write_interval": 0, "checkpoint_interval": 0}
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent = Runner(env, agent_cfg).agent
    agent.load(str(run_dir / "checkpoints" / checkpoint))
    agent.enable_training_mode(False, apply_to_models=True)
    return env, agent


def close_view_cfg():
    """Camera for play: starts close on env 0. "env" places it once and leaves orbit/zoom to you; "asset_root"
    would re-aim it every frame and fight the mouse."""
    from isaaclab.envs import ViewerCfg

    return ViewerCfg(eye=(2.5, 2.5, 1.2), lookat=(1.0, 0.0, 0.3), origin_type="env")
