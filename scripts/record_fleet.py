"""Film many riders at once, as in training: the Kit viewport (which renders every env) follows the middle of the pack.

    scripts/py scripts/record_fleet.py saved/03_stage3_centred --num_envs 64 --seconds 12 --out /tmp/fleet

Writes an mp4 clip to --out (convert to GIF separately). The scene sensors used by play.py --record only see env 0's
rider, so this goes through the viewport instead. Camera: --back m uphill, --side m across and --up m above the
centroid of all riders, looking --ahead m down the hill from it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playground.common import load_policy, make_env, parse_overrides

from isaaclab.app import add_launcher_args, launch_simulation

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("run", type=Path)
p.add_argument("--checkpoint", default="best_agent.pt")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--back", type=float, default=22.0)
p.add_argument("--side", type=float, default=14.0)
p.add_argument("--up", type=float, default=14.0)
p.add_argument("--ahead", type=float, default=8.0)
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
add_launcher_args(p)
args = p.parse_args()
args.visualizer = "kit"
args.livestream = args.livestream or 2  # loads the Kit viewport (headless alone does not)
args.enable_cameras = True  # replicator, which the viewport recorder uses

import torch
import yaml

import isaaclab.sim as sim_utils

sim_cfg = sim_utils.SimulationCfg()
with launch_simulation(sim_cfg, args):
    from isaaclab.envs import ViewerCfg
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg

    run = yaml.safe_load((args.run / "run.yaml").read_text())
    for viz in sim_cfg.visualizer_cfgs or []:
        if hasattr(viz, "window_width"):
            viz.window_width, viz.window_height = args.width, args.height
    steps = int(args.seconds / (1 / 30))
    ov = dict(run["overrides"]) | {"edge_cmd_level_init": 1.0, "edge_cmd_level_fixed": True} | parse_overrides(args.set)
    ov["viewer"] = ViewerCfg(origin_type="world")  # the default follows env 0's rider and would override the camera
    cfg = SnowboardEnvCfg(task=run["task"])
    cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=str(args.out), video_length=steps,
                                            output_filename_prefix="fleet")]
    raw = make_env(SnowboardEnv, cfg, ov, args.num_envs, sim_cfg, "true")
    env, agent = load_policy(raw, run, args.run, args.checkpoint)
    down = torch.tensor([1.0, 0.0, 0.0], device=raw.device)
    side = torch.tensor([0.0, 1.0, 0.0], device=raw.device)
    up = torch.tensor([0.0, 0.0, 1.0], device=raw.device)
    obs, _ = env.reset()
    for t in range(steps + 2):
        mid = raw.body_pos(raw.pelvis_id).mean(0)
        eye = raw.to_world(mid - args.back * down - args.side * side + args.up * up)
        target = raw.to_world(mid + args.ahead * down)
        raw.sim.set_camera_view(tuple(eye.tolist()), tuple(target.tolist()))
        with torch.inference_mode():
            a, out = agent.act(obs, None, timestep=t, timesteps=steps)
            obs, *_ = env.step(out.get("mean_actions", a))
    env.close()
    print(f"[fleet] wrote {sorted(args.out.glob('*.mp4'))}")
