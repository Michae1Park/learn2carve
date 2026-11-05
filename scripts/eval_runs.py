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

  Signs of a good carve (only "low" is in the reward; the rest should emerge):
    pelvis_m         mean pelvis height above the board (stance 0.80; low = 0.30-0.45)
    edge_change_s    mean time from 20° on one edge to 20° on the other: a quick edge change
    press_turn       mean load on the board N / mg while turning hard (|yaw rate| > 0.3 rad/s): the press
    press_change     ... and while changing edges (|edge| < 5°): the release
    facing           chest toward the travel direction, mean cos (1 = facing forward)
    hip_flex_deg     mean hip flexion (both hips, pitch); hip_flex_min_deg: the least folded it gets
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.common import load_policy, make_env

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

METRICS = ("fell", "fence", "turns_per_10s", "slip_deg", "carve_frac", "edge_deg", "speed_err", "seconds",
           # good carve, not rewarded except low: are they learned?
           "pelvis_m", "edge_change_s", "press_turn", "press_change", "facing", "hip_flex_deg", "hip_flex_min_deg")


@torch.inference_mode()  # the env's tensors are inference tensors; keep every stat in the same mode
def evaluate(run_dir: Path, sim_cfg) -> dict:
    from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

    run = yaml.safe_load((run_dir / "run.yaml").read_text())
    overrides = run["overrides"] | {"guide_steps": 0}  # evaluate without the training guide
    raw = make_env(SnowboardEnv, SnowboardEnvCfg(task=run["task"]), overrides, args.num_envs, sim_cfg,
                   args.suppress_warnings)
    env, agent = load_policy(raw, run, run_dir, args.checkpoint, cfg_dir=HERE)
    n, dev = raw.num_envs, raw.device
    keys = ("steps", "slip", "edge_turn", "turn_steps", "carve_steps", "speed", "pelvis", "facing", "hip", "press_turn",
            "press_turn_n", "press_change", "press_change_n", "edge_changes", "edge_change_t")
    acc = {k: torch.zeros(n, device=dev) for k in keys}
    hip_min = torch.full((n,), 1e9, device=dev)
    edge_side = torch.zeros(n, device=dev)  # last edge held past 20°: -1 / +1
    edge_left = torch.zeros(n, device=dev)  # step it was last past 20° on that side
    hips = [raw.robot.joint_names.index(j) for j in ("right_hip_y", "left_hip_y")]
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
        # signs of a good carve
        info, tech = raw.snow_info, raw._technique()
        theta = info["edge_angle"]
        press = info["normal_force"] / (9.81 * raw.snow.system_mass)
        hard_turn = (raw.body_ang_vel(raw.board_id)[:, 2].abs() > 0.3) & live
        changing = (theta.abs() < math.radians(5)) & live
        acc["pelvis"] += raw.pelvis_height() * live
        acc["facing"] += tech["facing"] * live
        flex = raw.robot.data.joint_pos.torch[:, hips].abs().mean(-1).rad2deg()
        acc["hip"] += flex * live
        hip_min = torch.where(live, torch.minimum(hip_min, flex), hip_min)
        acc["press_turn"] += press * hard_turn
        acc["press_turn_n"] += hard_turn.float()
        acc["press_change"] += press * changing
        acc["press_change_n"] += changing.float()
        side = torch.sign(theta) * (theta.abs() > math.radians(20))
        switched = (side != 0) & (edge_side != 0) & (side != edge_side) & live
        acc["edge_changes"] += switched.float()
        acc["edge_change_t"] += (t - edge_left) * raw.step_dt * switched
        held = (side != 0) & live
        edge_left = torch.where(held, torch.full_like(edge_left, t), edge_left)
        edge_side = torch.where(held, side, edge_side)
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
                "pelvis_m": acc["pelvis"][i].item() / steps,
                "edge_change_s": acc["edge_change_t"][i].item() / max(acc["edge_changes"][i].item(), 1.0),
                "press_turn": acc["press_turn"][i].item() / max(acc["press_turn_n"][i].item(), 1.0),
                "press_change": acc["press_change"][i].item() / max(acc["press_change_n"][i].item(), 1.0),
                "facing": acc["facing"][i].item() / steps,
                "hip_flex_deg": acc["hip"][i].item() / steps,
                "hip_flex_min_deg": hip_min[i].item(),
            })
            for v in acc.values():
                v[i] = 0.0
            hip_min[i], edge_side[i] = 1e9, 0.0
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
