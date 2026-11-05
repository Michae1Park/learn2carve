"""Can the rider get from a low toe carve to the heel edge at all? Feasibility test, no learning.

    scripts/py scripts/switch_test.py logs/playground/snowboard/<stage 2 run>

Every env starts in a balanced toe carve and the trained policy holds it (held-edge command) for --hold seconds,
which brings it into the low, bowed toe carve where stage 2 gets stuck. Then the command flips to the heel and each
env group gets a different override of the two synergy actions (edge, lean) for --switch seconds, while the policy
keeps driving the 28 residual joints:
  policy        no override (the policy's own answer to the flipped command)
  edge -g       edge action forced to -g (heel), lean from the policy
  edge -g lean l  both forced
Reported per group: share that reached the heel edge (θ < -10°), share that fell, time to the heel, pelvis height
and torso tilt at the flip.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.quadruped.common import load_policy, make_env, parse_overrides

from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("run", type=Path)
p.add_argument("--checkpoint", default="best_agent.pt")
p.add_argument("--hold", type=float, default=3.5, help="toe carve before the flip [s]")
p.add_argument("--switch", type=float, default=2.5, help="time allowed to reach the heel [s]")
p.add_argument("--reps", type=int, default=16)
p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
p.add_argument("--heel", default=None, help="synergy_heel_rad (yaml dict), switched on only at the flip")
add_launcher_args(p)
args = p.parse_args()
args.headless = True

import torch
import yaml

import isaaclab.sim as sim_utils

# (name, edge override or None, lean override or None, toe spike [s] before the override: edge +1, lean from policy)
GROUPS = [("policy", None, None, 0.0), ("edge -1.0", -1.0, None, 0.0), ("edge -1.0 lean -1.0", -1.0, -1.0, 0.0)]
for spike in (0.2, 0.4, 0.6):
    GROUPS += [(f"spike {spike} edge -1.0", -1.0, None, spike), (f"spike {spike} edge -1 lean -1", -1.0, -1.0, spike),
               (f"spike {spike} edge -0.5", -0.5, None, spike)]

with launch_simulation(sim_utils.SimulationCfg(), args):
    from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

    run = yaml.safe_load((args.run / "run.yaml").read_text())
    ov = dict(run["overrides"])
    ov |= {"carve_start": True, "edge_cmd_hold": True, "carve_start_toe_frac": 1.0, "rsi_frac": 0.0,
           "episode_length_s": args.hold + args.switch + 5, "guide_steps": 0, "start_heading_deg": 0.0}
    ov |= parse_overrides(args.set)
    if args.heel:
        ov["synergy_heel_rad"] = yaml.safe_load(args.heel)
    n = len(GROUPS) * args.reps
    raw = make_env(SnowboardEnv, SnowboardEnvCfg(task=run["task"]), ov, n, sim_utils.SimulationCfg(), "true")
    env, agent = load_policy(raw, run, args.run, args.checkpoint)
    dev = raw.device
    names = raw.robot.joint_names
    for j in ("abdomen_x", "abdomen_y", "abdomen_z", "left_hip_y", "right_hip_y", "left_knee", "right_knee",
              "right_shoulder_x", "right_shoulder_y", "right_shoulder_z"):
        i = names.index(j)
        print(f"[switch] limit {j:17s} stance {raw.stance_pos[0, i].item():+.2f}  "
              f"range [{raw.joint_lower[i].item():+.2f}, {raw.joint_upper[i].item():+.2f}] rad")
    heel_rad = raw.heel_rad.clone()
    raw.heel_rad.zero_()  # the hold phase rides as trained
    group = torch.arange(n, device=dev) // args.reps
    edge_ov = torch.tensor([float("nan") if g[1] is None else g[1] for g in GROUPS], device=dev)[group]
    lean_ov = torch.tensor([float("nan") if g[2] is None else g[2] for g in GROUPS], device=dev)[group]
    spike_s = torch.tensor([g[3] for g in GROUPS], device=dev)[group]

    obs, _ = env.reset()
    hold_steps, sw_steps = int(args.hold / raw.step_dt), int(args.switch / raw.step_dt)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    t_heel = torch.full((n,), float("nan"), device=dev)
    pelvis0 = torch.zeros(n, device=dev)
    tilt0 = torch.zeros(n, device=dev)
    edge0 = torch.zeros(n, device=dev)
    edge_min = torch.full((n,), 90.0, device=dev)
    for t in range(hold_steps + sw_steps):
        switching = t >= hold_steps
        if t == hold_steps:  # the flip: command the heel side
            raw.cmd_hold[:] = -raw.cmd_hold.abs().clamp(min=0.1)
            raw.heel_rad.copy_(heel_rad)
            from isaaclab.utils.math import quat_apply
            up = quat_apply(raw.body_quat(raw.torso_id), raw.up.expand(n, 3))
            tilt0 = torch.rad2deg(torch.acos((up * raw.up).sum(-1).clamp(-1, 1)))
            pelvis0 = raw.pelvis_height().clone()
            edge0 = torch.rad2deg(raw.snow_info["edge_angle"]).clone()
            alive_at_flip = alive.clone()
        with torch.inference_mode():
            action, out = agent.act(obs, None, timestep=t, timesteps=hold_steps + sw_steps)
            action = out.get("mean_actions", action).clone()
            if switching:
                e, l = ~torch.isnan(edge_ov), ~torch.isnan(lean_ov)
                spiking = (t - hold_steps) * raw.step_dt < spike_s
                action[e, 0] = torch.where(spiking[e], torch.ones_like(edge_ov[e]), edge_ov[e])
                l &= ~spiking
                action[l, 1] = lean_ov[l]
            obs, _, term, trunc, _ = env.step(action)
        done = (term | trunc).view(-1)
        theta = torch.rad2deg(raw.snow_info["edge_angle"])
        if switching:
            hit = alive & ~reached & ~done & (theta < -10.0)
            t_heel[hit] = (t - hold_steps) * raw.step_dt
            reached |= hit
            edge_min = torch.where(alive & ~done, torch.minimum(edge_min, theta), edge_min)
            fell |= alive & term.view(-1) & ~raw.out_of_bounds
        alive &= ~done

    ok = alive_at_flip & (edge0 > 5.0)  # really on the toe edge at the flip
    print(f"\n[switch] {args.run.name}: {ok.sum().item()}/{n} envs still up and carving at the flip")
    print(f"[switch] at the flip: toe edge {edge0[ok].mean():.1f}°, pelvis {pelvis0[ok].mean():.2f} m, "
          f"torso tilt {tilt0[ok].mean():.1f}°")
    print(f"[switch] {'group':22s}  reached heel  fell   t_heel s  most heel edge°")
    for i, (name, *_) in enumerate(GROUPS):
        m = ok & (group == i)
        k = max(m.sum().item(), 1)
        th = t_heel[m & reached]
        print(f"[switch] {name:22s}  {reached[m].sum().item() / k:11.0%}  {fell[m].sum().item() / k:4.0%}   "
              f"{th.mean().item() if len(th) else float('nan'):6.2f}   {edge_min[m].mean().item():8.1f}")
    env.close()
