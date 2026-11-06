"""Run folders, --set overrides, the train loop and policy loading. Importable before Isaac Sim launches."""

import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "playground/snowboard"
ENV_YAML = CFG_DIR / "env.yaml"
STAGE_DIR = CFG_DIR / "stages"
LOG_ROOT = ROOT / "logs/playground/snowboard"
DEFAULT_CHECKPOINT = "best_agent.pt"
sys.path.insert(0, str(ROOT))


def add_train_args(parser, default_timesteps: int | None = None):
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--num_envs", type=int, default=None, help="default: from <algo>.yaml")
    parser.add_argument("--timesteps", type=int, default=default_timesteps,
                        help="env steps per env; default: " + (str(default_timesteps) if default_timesteps else "from <algo>.yaml"))
    parser.add_argument("--stage", default=None, metavar="NAME|YAML",
                        help="stage2, stage3, stage4 (playground/snowboard/stages/) or a YAML path; default: env.yaml")
    parser.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE",
                        help="one-off overrides of env.yaml keys; rewards.<key>=... for a reward; may repeat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_from", type=Path, default=None, metavar="RUN[/FILE]",
                        help="warm start: load policy, value and normalisers from a run's checkpoint "
                             "(default file best_agent.pt); the observation and action sizes must match")
    add_suppress_arg(parser)


def add_play_args(parser):
    parser.add_argument("run", type=Path, help="run directory from train.py")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="file in <run>/checkpoints/")
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


def read_settings(path: Path) -> dict:
    """{"env": {...}, "rewards": {...}} from an env.yaml-style file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return {"env": data.get("env") or {}, "rewards": data.get("rewards") or {}}


def stage_file(stage: str) -> Path:
    path = Path(stage)
    return path if path.suffix else STAGE_DIR / f"{stage}.yaml"


def settings_overrides(settings: dict) -> dict:
    """Settings as one dict of cfg keys, rewards as rewards.<key>."""
    return {f"rewards.{k}": v for k, v in settings["rewards"].items()} | settings["env"]


def apply_settings(cfg, settings: dict):
    for key, value in settings_overrides(settings).items():
        set_field(cfg, key, tuple(value) if isinstance(value, list) else value)


def load_agent_cfg(algo: str, cfg_dir: Path = CFG_DIR) -> dict:
    return yaml.safe_load((cfg_dir / f"{algo}.yaml").read_text())


def new_run(args, task: str, cfg_dir: Path = CFG_DIR, log_root: Path = LOG_ROOT, **extra) -> tuple[Path, dict, dict]:
    """Create <log_root>/<time>_<algo>_<task>/ with a run.yaml. Returns (run_dir, run, agent_cfg)."""
    agent_cfg = load_agent_cfg(args.algo, cfg_dir)
    run_dir = log_root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.algo}_{task}"
    run_dir.mkdir(parents=True)
    run = {
        "task": task,
        "algo": args.algo,
        **extra,
        **run_settings(args.stage, args.set),
        "num_envs": args.num_envs or agent_cfg["num_envs"],
        "timesteps": args.timesteps or agent_cfg["timesteps"],
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    return run_dir, run, agent_cfg


def run_settings(stage: str | None, pairs: list[str]) -> dict:
    """What a run.yaml records: env.yaml, then the stage file, then --set; complete, so a replay needs no env.yaml."""
    base = read_settings(ENV_YAML)
    changes = read_settings(stage_file(stage)) if stage else {"env": {}, "rewards": {}}
    return {"stage": stage, "rewards": base["rewards"] | changes["rewards"],
            "overrides": base["env"] | changes["env"] | parse_overrides(pairs)}


def run_overrides(run: dict) -> dict:
    """A run.yaml's rewards and overrides as one dict of cfg keys (rewards as rewards.<name>)."""
    return {f"rewards.{k}": v for k, v in run.get("rewards", {}).items()} | run["overrides"]


def set_field(cfg, key: str, value, legacy: bool = False):
    """key: a field, or rewards.<field>. legacy: map an older run's rew_<term> and reward-shape keys into
    cfg.rewards and skip fields since removed, instead of raising."""
    *parents, name = key.split(".")
    target = cfg
    for part in parents:
        target = getattr(target, part)
    if hasattr(target, name):
        setattr(target, name, value)
    elif legacy and not parents and hasattr(cfg, "rewards") and hasattr(cfg.rewards, name.removeprefix("rew_")) \
            and not isinstance(value, bool):  # the old on/off flag edge_cmd: true is not the edge_cmd weight
        setattr(cfg.rewards, name.removeprefix("rew_"), value)
    elif not legacy:
        raise KeyError(f"{type(target).__name__} has no field {name!r}")


def make_env(env_cls, cfg, overrides: dict, num_envs: int, launched_sim_cfg, suppress_warnings: str = "true",
             legacy: bool = False):
    """Call inside ``launch_simulation(launched_sim_cfg, args)``, which fills in the device and visualizers.
    legacy: overrides come from a saved run.yaml, which may name fields since removed (see set_field)."""
    for key, value in overrides.items():
        set_field(cfg, key, value, legacy)
    cfg.scene.num_envs = num_envs
    cfg.sim.device = launched_sim_cfg.device
    cfg.sim.visualizer_cfgs = launched_sim_cfg.visualizer_cfgs
    if suppress_warnings == "true":
        quiet_physx_tensors()
    return env_cls(cfg)


def quiet_physx_tensors():
    """Hides thousands of harmless "Failed to find rigid body" warnings; errors still show."""
    import carb

    carb.settings.get_settings().set("/log/channels/omni.physx.tensors.plugin", "error")


def train(raw_env, agent_cfg: dict, run_dir: Path, run: dict, seed: int, init_from: Path | None = None):
    """Ctrl+C stops early; a final checkpoint is saved either way."""
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils import set_seed
    from skrl.utils.runner.torch import Runner

    timesteps = run["timesteps"]
    agent_cfg["seed"] = seed
    agent_cfg["agent"]["experiment"] = {"directory": str(run_dir.parent), "experiment_name": run_dir.name}
    agent_cfg["trainer"]["timesteps"] = timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    set_seed(seed)
    env = SkrlVecEnvWrapper(raw_env)
    runner = Runner(env, agent_cfg)
    if init_from is not None:
        warm_start(runner.agent, init_from, agent_cfg["models"]["policy"].get("initial_log_std", -1.0))
    print(f"[train] {run['algo']} / {run['task']} / {run['num_envs']} envs / {timesteps} steps -> {run_dir}")
    try:
        runner.run()
    except KeyboardInterrupt:
        print("[train] interrupted")
    runner.agent.write_checkpoint(timestep=timesteps, timesteps=timesteps)
    env.close()


def warm_start(agent, init_from: Path, log_std: float):
    """Load a checkpoint, then reset the exploration std the entropy bonus may have inflated."""
    checkpoint = init_from if init_from.suffix == ".pt" else init_from / "checkpoints" / DEFAULT_CHECKPOINT
    agent.load(str(checkpoint))
    print(f"[train] warm start from {checkpoint}")
    for name, param in agent.policy.named_parameters():
        if "log_std" in name:
            param.data.fill_(log_std)
    print(f"[train] exploration std reset to e^{log_std:g}")


def load_policy(raw_env, run: dict, run_dir: Path, checkpoint: str, cfg_dir: Path = CFG_DIR):
    """Returns (skrl-wrapped env, agent in eval mode)."""
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils.runner.torch import Runner

    env = SkrlVecEnvWrapper(raw_env)
    agent_cfg = load_agent_cfg(run["algo"], cfg_dir)
    agent_cfg["agent"]["experiment"] = {"write_interval": 0, "checkpoint_interval": 0}
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent = Runner(env, agent_cfg).agent
    agent.load(str(run_dir / "checkpoints" / checkpoint))
    agent.enable_training_mode(False, apply_to_models=True)
    return env, agent


def close_view_cfg():
    """Placed once near env 0 ("asset_root" would re-aim every frame and fight the mouse)."""
    from isaaclab.envs import ViewerCfg

    return ViewerCfg(eye=(2.5, 2.5, 1.2), lookat=(1.0, 0.0, 0.3), origin_type="env")
