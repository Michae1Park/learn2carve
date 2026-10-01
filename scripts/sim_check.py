"""Stage 1 check: rider in stance on the slope, no policy.

    scripts/py scripts/sim_check.py                    # 9 riders, ankle offsets swept from heel to toe edge
    scripts/py scripts/sim_check.py --num_envs 1 --edge 0 --seconds 10
    scripts/py scripts/sim_check.py --slope 5 --start_speed 2      # gentle slope

Each env holds the snowboard stance (zero action) and, after --hold seconds, adds a fixed ankle-pitch action that
tips the board: + presses the toes (toe edge), - lifts them (heel edge). Reports per env: survival, speed, edge
angle, turn radius and grip usage, so the snow model can be checked with a real rider on top.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--hold", type=float, default=2.0, help="seconds of plain stance before edging")
parser.add_argument("--slope", type=float, default=None, help="slope angle [deg] (default: config.yaml)")
parser.add_argument("--start_speed", type=float, default=None, help="speed at reset [m/s] (default: env range)")
parser.add_argument("--edge", type=float, default=None, help="one ankle action for all envs (default: sweep -1..1)")
add_launcher_args(parser)
args = parser.parse_args()
if args.hold >= args.seconds:
    parser.error("--hold must be shorter than --seconds")

import math

import torch

import isaaclab.sim as sim_utils


def main():
    with launch_simulation(sim_utils.SimulationCfg(), args):  # PhysX + Kit
        from envs.carve_env import CarveEnv, CarveEnvCfg, tilted_gravity

        cfg = CarveEnvCfg()
        if args.slope is not None:
            cfg.sim.gravity = tilted_gravity(args.slope)
        if args.start_speed is not None:
            cfg.start_speed = (args.start_speed, args.start_speed)
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device or cfg.sim.device
        cfg.episode_length_s = args.seconds + 1.0
        env = CarveEnv(cfg)
        env.reset()
        print(f"[check] mass scale {env.mass_scale:.3f} -> rider + board "
              f"{env.robot.data.body_mass.torch[0].sum().item():.1f} kg")

        n = env.num_envs
        edge = torch.full((n,), args.edge, device=env.device) if args.edge is not None else torch.linspace(-1, 1, n, device=env.device)
        ankle_ids = [env.robot.data.joint_names.index(j) for j in ("left_ankle_y", "right_ankle_y")]
        action = torch.zeros(n, cfg.action_space, device=env.device)

        steps = int(args.seconds / env.step_dt)
        hold_steps = int(args.hold / env.step_dt)
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        fell_at = torch.full((n,), float("nan"), device=env.device)
        log = {k: [] for k in ("edge", "speed", "yaw_rate", "grip", "slip")}
        for step in range(steps):
            if step == hold_steps:
                action[:, ankle_ids] = edge.unsqueeze(-1)
            _, _, terminated, _, _ = env.step(action)
            newly = terminated & alive
            fell_at[newly] = step * env.step_dt
            alive &= ~terminated
            info = env.snow_info
            board_vel = env.robot.data.body_lin_vel_w.torch[:, env.board_id]
            if step >= hold_steps:
                log["edge"].append(torch.rad2deg(info["edge_angle"]))
                log["speed"].append(board_vel[:, :2].norm(dim=-1))
                log["yaw_rate"].append(env.robot.data.body_ang_vel_w.torch[:, env.board_id, 2])
                log["grip"].append(info["grip_usage"])
                log["slip"].append(info["v_lat"].abs())

        # stats over the edging phase, while each env was still up (episodes auto-reset after a fall)
        L = {k: torch.stack(v) for k, v in log.items()}
        T = L["edge"].shape[0]
        t_idx = torch.arange(T, device=env.device).unsqueeze(-1) * env.step_dt + args.hold
        valid = torch.isnan(fell_at).unsqueeze(0) | (t_idx < fell_at.unsqueeze(0))

        def mean(x):
            return (x * valid).sum(0) / valid.sum(0).clamp_min(1)

        print("\n[check]  ankle  fell@s  speed  edge°  yaw°/s  radius_m  R·cosθ  grip_use  slip")
        R = env.snow.sidecut_radius
        for i in range(n):
            e, s, w = mean(L["edge"])[i].item(), mean(L["speed"])[i].item(), mean(L["yaw_rate"])[i].item()
            radius = s / abs(w) if abs(w) > 1e-3 else math.inf
            expect = R * math.cos(math.radians(e)) if abs(e) > env.snow.min_edge_deg else math.inf
            fell = "-" if math.isnan(fell_at[i].item()) else f"{fell_at[i].item():.1f}"
            print(f"[check] {edge[i].item():+5.2f}  {fell:>6}  {s:5.1f}  {e:+5.1f}  {math.degrees(w):+6.1f}  "
                  f"{radius:8.1f}  {expect:6.1f}  {mean(L['grip'])[i].item():8.2f}  {mean(L['slip'])[i].item():4.2f}")
        env.close()


if __name__ == "__main__":
    main()
