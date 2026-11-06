"""Where does the rider shake? Ride a policy and rank joints by high-frequency motion.

    scripts/py scripts/jitter_probe.py logs/playground/snowboard/<run> [--seconds 10]

For every joint: RMS joint velocity, and RMS of its velocity's step-to-step change (jerk-like, shows chatter), plus
the policy's action-to-action change. Envs that fall are dropped from the stats from then on.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.common import load_policy, make_env, parse_overrides, run_overrides

from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("run", type=Path)
p.add_argument("--checkpoint", default="best_agent.pt")
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
add_launcher_args(p)
args = p.parse_args()
args.headless = True

import torch
import yaml

import isaaclab.sim as sim_utils

with launch_simulation(sim_utils.SimulationCfg(), args):
    from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

    run = yaml.safe_load((args.run / "run.yaml").read_text())
    ov = run_overrides(run) | {"rsi_frac": 0.0, "episode_length_s": args.seconds + 5} | parse_overrides(args.set)
    raw = make_env(SnowboardEnv, SnowboardEnvCfg(), ov, args.num_envs, sim_utils.SimulationCfg(), "true", legacy=True)
    env, agent = load_policy(raw, run, args.run, args.checkpoint)
    names = raw.robot.joint_names
    n, J, dev = args.num_envs, len(names), raw.device
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    sum_v2 = torch.zeros(J, device=dev)
    sum_dv2 = torch.zeros(J, device=dev)
    sum_da2 = torch.zeros(raw.cfg.action_space, device=dev)
    count = 0
    obs, _ = env.reset()
    prev_v = raw.robot.data.joint_vel.torch.clone()
    prev_a = None
    for t in range(int(args.seconds / raw.step_dt)):
        with torch.inference_mode():
            a, out = agent.act(obs, None, timestep=t, timesteps=1)
            a = out.get("mean_actions", a)
            obs, _, term, trunc, _ = env.step(a)
        alive &= ~(term | trunc).view(-1)
        v = raw.robot.data.joint_vel.torch
        if t > 10 and alive.any():
            m = alive
            sum_v2 += v[m].square().sum(0)
            sum_dv2 += (v[m] - prev_v[m]).square().sum(0)
            if prev_a is not None:
                sum_da2 += (a[m] - prev_a[m]).square().sum(0)
            count += int(m.sum())
        prev_v, prev_a = v.clone(), a.clone()
    c = max(count, 1)
    rms_v, rms_dv = (sum_v2 / c).sqrt(), (sum_dv2 / c).sqrt()
    print(f"\n[jitter] {args.run.name}: {alive.sum().item()}/{n} envs still up after {args.seconds:.0f} s")
    print(f"[jitter] {'joint':18s}  rms vel rad/s  rms Δvel/step")
    for i in rms_dv.argsort(descending=True)[:12].tolist():
        print(f"[jitter] {names[i]:18s}  {rms_v[i].item():12.2f}  {rms_dv[i].item():12.3f}")
    da = (sum_da2 / c).sqrt()
    print(f"[jitter] action change per step (rms): synergy edge {da[0]:.3f}, lean {da[1]:.3f}, "
          f"joints mean {da[2:].mean():.3f}  (actions are in [-1, 1])")
