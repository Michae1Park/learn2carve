"""Ride a trained snowboard policy, draw its track on the snow and save a top-down picture of it.

    scripts/py playground/snowboard/play.py logs/playground/snowboard/<run>
    scripts/py playground/snowboard/play.py <run> --livestream 2 --seconds 120
    scripts/py playground/snowboard/play.py <run> --livestream 2 --follow    # camera tracks the rider
    scripts/py playground/snowboard/play.py --no_policy --task sideslip      # stance held, zero actions

Play is a series of short runs, each starting the same way (piste centre, straight down, mid-range speed command)
and ending --run_length metres down (default 100 m; 25 m for the side slip). The rider then returns to the top.
--seconds is the total play time. One line is printed per run (env 0), and the longest run's track is saved to
<run>/trail.png.

The camera starts low on the snow at the bottom of the run, looking up at the start, then it is yours (orbit, pan,
zoom). --cam_x moves it along the run; --follow tracks the rider instead.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so this runs with plain python too

from playground.quadruped.common import ROOT, add_suppress_arg, load_policy, make_env, parse_overrides

from isaaclab.app import add_launcher_args, launch_simulation

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, nargs="?", help="run directory from train.py")
parser.add_argument("--checkpoint", default="best_agent.pt", help="file in <run>/checkpoints/")
parser.add_argument("--no_policy", action="store_true", help="hold the stance (zero actions) instead of a policy")
parser.add_argument("--task", choices=["sideslip", "skid", "carve"], default=None, help="default: the run's task")
parser.add_argument("--seconds", type=float, default=60.0, help="total play time")
parser.add_argument("--run_length", type=float, default=None, help="one run [m]; default: two S's (2 × turn length)")
parser.add_argument("--cam_x", type=float, default=None,
                    help="camera distance down the run from the start [m]; default: just past where a run ends")
parser.add_argument("--focal", type=float, default=35.0, help="camera focal length [mm] (Kit's default is ~18)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--follow", action="store_true", help="camera tracks the rider (default: free camera)")
parser.add_argument("--record", type=Path, default=None, metavar="DIR",
                    help="film one run with two follow cameras (front, top) -> DIR/front.gif, DIR/top.gif; needs --enable_cameras")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="env cfg overrides on top of the run's")
add_suppress_arg(parser)
add_launcher_args(parser)
args = parser.parse_args()
if args.record:
    args.enable_cameras = True  # the launcher only turns RTX cameras on for sensors it finds in a cfg
if not args.no_policy and args.run is None:
    parser.error("give a run directory, or --no_policy")

import torch
import yaml

import isaaclab.sim as sim_utils


class RunStats:
    """Averages over one run of env 0, printed as one line. Turns come from the env's own counter (linked turns)."""

    HEADER = "run  end    seconds  speed (cmd)  |edge|°  slip_err°  |heading|°  max|y|_m  turns  m/turn"

    def __init__(self):
        self.steps = 0
        self.sums = dict.fromkeys(("speed", "edge", "slip_err", "heading"), 0.0)
        self.max_y = 0.0
        self.x0 = self.x = 0.0
        self.turns = 0
        self.v_cmd = 0.0
        self.rows = []  # per step: x, y, speed, slip°, edge° (saved as ride.csv for the longest run)

    def add(self, env):
        x, y = env.position()[0].tolist()
        if self.steps == 0:
            self.x0 = x
        self.steps += 1
        self.x = x
        self.turns = int(env.turn_count[0].item())
        self.v_cmd = env.v_cmd[0].item()
        self.sums["speed"] += env.board_speed()[0].item()
        self.sums["edge"] += abs(math.degrees(env.snow_info["edge_angle"][0].item()))
        self.sums["slip_err"] += abs(math.degrees(env.slip_error()[0].item()))
        self.sums["heading"] += abs(math.degrees(env.path_heading()[0].item()))
        self.max_y = max(self.max_y, abs(y))
        self.rows.append((x, y, env.board_speed()[0].item(), math.degrees(env.slip_angle()[0].item()),
                          math.degrees(env.snow_info["edge_angle"][0].item()),
                          math.degrees(env.edge_command()[0].item()) if env.cfg.edge_cmd else 0.0))

    def line(self, number: int, end: str, step_dt: float) -> str:
        mean = {k: v / max(self.steps, 1) for k, v in self.sums.items()}
        per_turn = f"{(self.x - self.x0) / self.turns:6.1f}" if self.turns else "     -"
        return (f"{number:3d}  {end:5s}  {self.steps * step_dt:7.1f}  {mean['speed']:5.1f} ({self.v_cmd:4.1f})  "
                f"{mean['edge']:6.1f}  {mean['slip_err']:9.1f}  {mean['heading']:10.1f}  {self.max_y:8.1f}  "
                f"{self.turns:5d}  {per_turn}")


def main():
    run = yaml.safe_load((args.run / "run.yaml").read_text()) if args.run else {"task": "sideslip", "overrides": {}}
    task = args.task or run["task"]
    out = args.run / "trail.png" if args.run else ROOT / f"logs/playground/snowboard/no_policy_{task}_trail.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from envs import config
        from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg
        from playground.snowboard.trail import SnowTrail

        overrides, run_length = play_overrides(run["overrides"], SnowboardEnvCfg(task=task))
        env_cfg = SnowboardEnvCfg(task=task)
        if args.record:
            Recorder.add_cameras(env_cfg)
        raw_env = make_env(SnowboardEnv, env_cfg, overrides, args.num_envs, sim_cfg,
                           args.suppress_warnings)
        if args.no_policy:
            env, agent = raw_env, None
        else:
            env, agent = load_policy(raw_env, run, args.run, args.checkpoint, cfg_dir=HERE)
        if not args.follow:
            set_focal_length(raw_env.cfg.viewer.cam_prim_path, args.focal)
        board = config.load()["board"]
        trail = SnowTrail(board["length"], board["waist_width"], raw_env.q_slope)

        print(f"\n[play] task={task}  {'no policy' if args.no_policy else args.run.name}  run {run_length:.0f} m, "
              f"speed command {raw_env.v_cmd[0].item():.1f} m/s, fence ±{raw_env.cfg.fence_half_width:.0f} m")
        print(f"[play] {RunStats.HEADER}")
        recorder = Recorder(raw_env, trail) if args.record else None
        longest_secs, longest_marks, rows = ride(env, raw_env, agent, trail, recorder)
        csv = out.with_name(out.stem.replace("trail", "ride") + ".csv")  # next to trail.png
        csv.write_text("x,y,speed,slip_deg,edge_deg,cmd_deg\n" + "".join(",".join(f"{v:.3f}" for v in r) + "\n" for r in rows))
        if recorder:
            recorder.save(args.record)

        trail.marks = longest_marks
        trail.save_png(out, f"{task}: {longest_secs:.1f} s at {raw_env.v_cmd[0].item():.1f} m/s commanded",
                       fence=raw_env.cfg.fence_half_width, x_end=run_length)
        print(f"[play] track of the longest episode ({longest_secs:.1f} s) -> {out}")
        env.close()


def play_overrides(run_overrides: dict, defaults) -> tuple[dict, float]:
    """Env overrides for short, repeatable runs. Returns (overrides, run length).

    Training's cfg is the base; --set wins over everything. Unless set, every run starts the same way: mid-range
    speed command, on the piste centre, pointing straight down the fall line.
    """
    user = parse_overrides(args.set)
    overrides = run_overrides | user

    def middle(name: str) -> float:
        lo, hi = overrides.get(name, getattr(defaults, name))
        return (lo + hi) / 2

    if "speed_cmd" not in user:
        overrides["speed_cmd"] = (middle("speed_cmd"),) * 2
    if "start_heading_deg" not in user:
        overrides["start_heading_deg"] = 0.0
    overrides["guide_steps"] = 0  # the guide S is a training aid only: play rides without it

    run_length = args.run_length or (25.0 if defaults.task == "sideslip" else 100.0)
    overrides["run_length"] = run_length
    overrides["episode_length_s"] = 60.0  # runs end by distance; this only catches a rider who stops
    if not args.follow:
        overrides["viewer"] = free_view_cfg(overrides.get("slope_deg"), args.cam_x or run_length + 10.0)
    return overrides, run_length


def ride(env, raw_env, agent, trail, recorder=None) -> tuple[float, list, list]:
    """Play for --seconds, one printed line per run. Returns the longest run's (seconds, trail marks, stat rows)."""
    steps = int(args.seconds / raw_env.step_dt)
    obs, _ = env.reset()
    stats, runs = RunStats(), 0
    longest = (-1.0, [], [])
    for t in range(steps):
        with torch.inference_mode():
            if agent is None:
                action = torch.zeros(raw_env.num_envs, raw_env.cfg.action_space, device=raw_env.device)
            else:
                action, outputs = agent.act(obs, None, timestep=t, timesteps=steps)
                action = outputs.get("mean_actions", action)
            obs, _, terminated, truncated, _ = env.step(action)
        terminated, truncated = bool(terminated.view(-1)[0]), bool(truncated.view(-1)[0])
        done = terminated or truncated

        if not done:  # on a done step the env has already reset to the start
            stats.add(raw_env)
            trail.add(raw_env.body_pos(raw_env.board_id)[0], raw_env.slip_angle()[0].item())
            if recorder:
                recorder.capture()
        if done or t == steps - 1:
            runs += 1
            end = "fence" if raw_env.out_of_bounds[0] else "fell" if terminated else "done" if truncated else "-"
            print(f"[play] {stats.line(runs, end, raw_env.step_dt)}")
            seconds = stats.steps * raw_env.step_dt
            if seconds > longest[0]:
                longest = (seconds, list(trail.marks), stats.rows)
            trail.marks.clear()
            trail.reset()
            stats = RunStats()
            if recorder:
                break
    return longest


class Recorder:
    """Two cameras that follow env 0's rider: "front" faces the rider's chest (frontside) from 5 m, "top" looks down
    on the board from 16 m up so the track behind it shows. Frames are kept on the CPU, written as GIFs at the end."""

    SIZE = (480, 320)  # w, h

    VIEWS = {"front": 30.0, "top": 18.0}  # name: focal length [mm]

    @classmethod
    def add_cameras(cls, env_cfg):
        """Cameras go in the scene cfg so they are built with the env (built later, the rider never renders)."""
        from isaaclab.sensors import CameraCfg

        for name, focal in cls.VIEWS.items():
            setattr(env_cfg.scene, f"record_{name}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Record_{name}", width=cls.SIZE[0], height=cls.SIZE[1],
                data_types=["rgb"], update_period=0.0,
                spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.1, 2000.0))))

    def __init__(self, env, trail):
        self.env, self.trail = env, trail
        self.frames = {name: [] for name in self.VIEWS}
        self.cams = {name: env.scene[f"record_{name}"] for name in self.VIEWS}
        self.facing = None

    def capture(self):
        env = self.env
        pelvis = env.body_pos(env.pelvis_id)[0]
        # chest direction: pelvis local +x, flattened onto the snow and smoothed so turns don't shake the camera
        from isaaclab.utils.math import quat_apply

        fwd = quat_apply(env.body_quat(env.pelvis_id)[0], torch.tensor([1.0, 0.0, 0.0], device=env.device))
        fwd[2] = 0.0
        fwd = fwd / fwd.norm().clamp(min=1e-6)
        self.facing = fwd if self.facing is None else 0.92 * self.facing + 0.08 * fwd
        f = self.facing / self.facing.norm().clamp(min=1e-6)
        up = torch.tensor([0.0, 0.0, 1.0], device=env.device)
        down = torch.tensor([1.0, 0.0, 0.0], device=env.device)  # downhill in the slope frame
        views = {
            "front": (pelvis + 5.0 * f + 0.6 * up + 1.0 * down, pelvis),
            # eye a little downhill of its target: image up = uphill, so the track trails up the frame behind the rider
            "top": (pelvis + 2.0 * down + 16.0 * up, pelvis - 4.0 * down - 0.8 * up),
        }
        for name, (eye, target) in views.items():
            cam = self.cams[name]
            cam.set_world_poses_from_view(env.to_world(eye)[None], env.to_world(target)[None])
        env.sim.render()
        for name, cam in self.cams.items():
            self.frames[name].append(cam.data.output["rgb"][0, ..., :3].cpu().numpy())

    def save(self, out_dir: Path):
        import imageio.v2 as imageio

        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frames in self.frames.items():
            path = out_dir / f"{name}.gif"
            imageio.mimsave(path, frames[::2], duration=2 * self.env.step_dt * 1000, loop=0)  # 15 fps, real time
            print(f"[play] {len(frames[::2])} frames -> {path}")


def free_view_cfg(slope_deg: float | None, cam_x: float):
    """Camera placed once, low on the snow cam_x down the run, looking back up at env 0's start. Kit leaves it alone
    after that (origin_type "env"), so the mouse controls it. "asset_root" (--follow) re-aims it every frame."""
    from isaaclab.envs import ViewerCfg

    from envs import config
    from envs.mountain import slope_to_world

    a = slope_deg if slope_deg is not None else config.load()["slope"]["angle_deg"]
    # 0.6 m up: on a plane the horizon sits at eye height, so the 1.6 m rider rises above it into the sky
    return ViewerCfg(eye=slope_to_world(a, cam_x, 3.0, 0.6), lookat=slope_to_world(a, 0.0, 0.0, 1.0), origin_type="env")


def set_focal_length(cam_prim_path: str, focal_mm: float):
    """Longer lens than Kit's ~18 mm default: the rider fills more of the frame from the bottom of the run."""
    from isaaclab.sim.utils.stage import get_current_stage

    prim = get_current_stage().GetPrimAtPath(cam_prim_path)
    if prim.IsValid():  # only exists with a viewer (e.g. --livestream)
        prim.GetAttribute("focalLength").Set(focal_mm)


if __name__ == "__main__":
    main()
