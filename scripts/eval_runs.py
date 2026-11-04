"""Evaluate trained snowboard policies: deterministic rollouts on many envs, one summary row per run.

    scripts/py scripts/eval_runs.py logs/playground/snowboard/<run> --out docs/results/eval.yaml   # one run per call

Each run's own env cfg (run.yaml overrides) is used, so baseline and new-model runs are each evaluated on the model
they were trained on. Per run: --episodes complete episodes over --num_envs envs (the policy's mean action).

    fell / fence     share of episodes ending in a fall / past a fence (the rest time out: rode the full 12 s)
    turns_per_10s    linked turns (the path crosses the fall line after swinging ≥ min_turn_deg)
    slip_deg         mean |slip angle error|
    carve_frac       of the time spent turning (|heading| > 15°), the share on a clean edge (|slip| < 5°)
    edge_deg         mean |edge angle| while turning: how hard the board is tipped
    speed_err        mean |speed - command| [m/s]
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.quadruped.common import load_policy, make_env

from isaaclab.app import add_launcher_args, launch_simulation

HERE = Path(__file__).resolve().parents[1] / "playground/snowboard"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path)
parser.add_argument("--checkpoint", default="best_agent.pt")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--out", type=Path, default=None, help="append results as YAML")
parser.add_argument("--suppress_warnings", default="true")
add_launcher_args(parser)
args = parser.parse_args()

import torch
import yaml

import isaaclab.sim as sim_utils

METRICS = ("fell", "fence", "turns_per_10s", "slip_deg", "carve_frac", "edge_deg", "speed_err", "seconds")


def evaluate(run_dir: Path, sim_cfg) -> dict:
    from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

    run = yaml.safe_load((run_dir / "run.yaml").read_text())
    overrides = run["overrides"] | {"guide_steps": 0}  # evaluate without the training guide
    raw = make_env(SnowboardEnv, SnowboardEnvCfg(task=run["task"]), overrides, args.num_envs, sim_cfg,
                   args.suppress_warnings)
    env, agent = load_policy(raw, run, run_dir, args.checkpoint, cfg_dir=HERE)
    n, dev = raw.num_envs, raw.device
    acc = {k: torch.zeros(n, device=dev) for k in ("steps", "slip", "edge_turn", "turn_steps", "carve_steps", "speed")}
    done_rows = []
    obs, _ = env.reset()
    t = 0
    while len(done_rows) < args.episodes:
        with torch.inference_mode():
            _, outputs = agent.act(obs, None, timestep=t, timesteps=10**9)
            obs, _, terminated, truncated, _ = env.step(outputs["mean_actions"])
        t += 1
        terminated, truncated = terminated.view(-1).bool(), truncated.view(-1).bool()
        done = terminated | truncated
        # stats of the step just taken; envs that finished were reset already, so leave them out of this step
        live = ~done
        turning = raw.path_heading().abs() > math.radians(15)
        slip = raw.slip_error().abs()
        acc["steps"] += live
        acc["slip"] += slip.rad2deg() * live
        acc["speed"] += (raw.board_speed() - raw.v_cmd).abs() * live
        acc["turn_steps"] += (turning & live).float()
        acc["carve_steps"] += (turning & live & (raw.slip_angle().abs() < math.radians(5))).float()
        acc["edge_turn"] += raw.snow_info["edge_angle"].abs().rad2deg() * (turning & live)
        for i in done.nonzero().flatten().tolist():
            steps = max(acc["steps"][i].item(), 1.0)
            seconds = steps * raw.step_dt
            turn_steps = max(acc["turn_steps"][i].item(), 1.0)
            done_rows.append({
                "fell": float(terminated[i] and not raw.out_of_bounds[i]),
                "fence": float(raw.out_of_bounds[i]),
                "turns_per_10s": raw.last_turns[i].item() / seconds * 10,
                "slip_deg": acc["slip"][i].item() / steps,
                "carve_frac": acc["carve_steps"][i].item() / turn_steps,
                "edge_deg": acc["edge_turn"][i].item() / turn_steps,
                "speed_err": acc["speed"][i].item() / steps,
                "seconds": seconds,
            })
            for v in acc.values():
                v[i] = 0.0
    env.close()
    rows = done_rows[: args.episodes]
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in METRICS}
    return {"run": run_dir.name, "task": run["task"], "seed": run.get("seed"), "overrides": run["overrides"],
            **{k: round(v, 4) for k, v in summary.items()}}


def main():
    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):  # one env per process: Isaac Lab builds one scene per stage
        r = evaluate(args.run, sim_cfg)
        print("[eval] " + "  ".join([r["run"]] + [f"{k} {r[k]:.3f}" for k in METRICS]), flush=True)
        if args.out:  # inside the context: Isaac Sim's shutdown ends the process when it closes
            args.out.parent.mkdir(parents=True, exist_ok=True)
            old = yaml.safe_load(args.out.read_text()) if args.out.exists() else []
            args.out.write_text(yaml.safe_dump(old + [r], sort_keys=False))
            print(f"[eval] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
