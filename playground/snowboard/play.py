"""Ride a trained snowboard policy, draw its track on the snow and save a top-down picture of it.

    scripts/py playground/snowboard/play.py logs/playground/snowboard/<run>
    scripts/py playground/snowboard/play.py <run> --livestream 2 --seconds 120
    scripts/py playground/snowboard/play.py <run> --livestream 2 --follow    # camera tracks the rider
    scripts/py playground/snowboard/play.py --no_policy                      # stance held, zero actions

Play is a series of identical runs (piste centre, straight down, mid-range speed command), each --run_length
metres long. One line is printed per run of env 0; the longest run's track goes to <run>/trail.png and its
per-step state to <run>/ride.csv. The camera starts low at the bottom of the run, looking up; then it is yours.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.common import (DEFAULT_CHECKPOINT, LOG_ROOT, add_suppress_arg, load_policy, make_env,
                               load_run, parse_overrides, run_overrides)

from isaaclab.app import add_launcher_args, launch_simulation

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path, nargs="?", help="run directory from train.py")
parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="file in <run>/checkpoints/")
parser.add_argument("--no_policy", action="store_true", help="hold the stance (zero actions) instead of a policy")
parser.add_argument("--seconds", type=float, default=60.0, help="total play time")
parser.add_argument("--run_length", type=float, default=None, help="one run [m]; default 100")
parser.add_argument("--cam_x", type=float, default=None,
                    help="camera distance down the run from the start [m]; default: just past where a run ends")
parser.add_argument("--focal", type=float, default=35.0, help="camera focal length [mm]")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--follow", action="store_true", help="camera tracks the rider (default: free camera)")
parser.add_argument("--record", type=Path, default=None, metavar="DIR",
                    help="film one run with two follow cameras -> DIR/front.gif, DIR/top.gif; needs --enable_cameras")
parser.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE", help="env cfg overrides on top of the run's")
add_suppress_arg(parser)
add_launcher_args(parser)
args = parser.parse_args()
if args.record:
    args.enable_cameras = True
if not args.no_policy and args.run is None:
    parser.error("give a run directory, or --no_policy")

import torch
import yaml

import isaaclab.sim as sim_utils

RIDE_COLUMNS = ("x", "y", "speed", "slip_deg", "edge_deg", "cmd_deg", "torso_tilt_deg", "pelvis_h")


def env0(tensor: torch.Tensor) -> float:
    return tensor[0].item()


class RunStats:
    """Averages over one run of env 0, printed as one line."""

    HEADER = "run  end    seconds  speed (cmd)  |edge|°  |slip|°  |heading|°  max|y|_m  turns  m/turn"

    def __init__(self):
        self.steps = 0
        self.sums = dict.fromkeys(("speed", "edge", "slip", "heading"), 0.0)
        self.max_y = 0.0
        self.x0 = self.x = 0.0
        self.turns = 0
        self.v_cmd = 0.0
        self.rows = []

    def add(self, env):
        x, y = env.position()[0].tolist()
        if self.steps == 0:
            self.x0 = x
        self.steps += 1
        self.x = x
        self.turns = int(env0(env.turn_count))
        self.v_cmd = env0(env.v_cmd)
        speed = env0(env.board_speed())
        edge_deg = math.degrees(env0(env.edge_angle()))
        self.sums["speed"] += speed
        self.sums["edge"] += abs(edge_deg)
        self.sums["slip"] += abs(math.degrees(env0(env.slip_angle())))
        self.sums["heading"] += abs(math.degrees(env0(env.path_heading())))
        self.max_y = max(self.max_y, abs(y))
        command_deg = math.degrees(env0(env.edge_command()))
        tilt_deg = math.degrees(env0(env.torso_incline()))
        self.rows.append((x, y, speed, math.degrees(env0(env.slip_angle())), edge_deg, command_deg, tilt_deg,
                          env0(env.pelvis_height())))

    def line(self, number: int, end: str, step_dt: float) -> str:
        mean = {k: v / max(self.steps, 1) for k, v in self.sums.items()}
        per_turn = f"{(self.x - self.x0) / self.turns:6.1f}" if self.turns else "     -"
        return (f"{number:3d}  {end:5s}  {self.steps * step_dt:7.1f}  {mean['speed']:5.1f} ({self.v_cmd:4.1f})  "
                f"{mean['edge']:6.1f}  {mean['slip']:7.1f}  {mean['heading']:10.1f}  {self.max_y:8.1f}  "
                f"{self.turns:5d}  {per_turn}")


def main():
    run = load_run(args.run) if args.run else {"overrides": {}}
    trail_png = args.run / "trail.png" if args.run else LOG_ROOT / "no_policy_trail.png"
    trail_png.parent.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg()
    with launch_simulation(sim_cfg, args):
        from envs import physical
        from playground.snowboard.env import SnowboardEnv, SnowboardEnvCfg
        from playground.snowboard.trail import SnowTrail

        overrides, run_length = play_overrides(run_overrides(run), SnowboardEnvCfg())
        env_cfg = SnowboardEnvCfg()
        if args.record:
            Recorder.add_cameras(env_cfg)
        raw_env = make_env(SnowboardEnv, env_cfg, overrides, args.num_envs, sim_cfg, args.suppress_warnings, legacy=True)
        if args.no_policy:
            env, agent = raw_env, None
        else:
            env, agent = load_policy(raw_env, run, args.run, args.checkpoint, cfg_dir=HERE)
        if not args.follow:
            set_focal_length(raw_env.cfg.viewer.cam_prim_path, args.focal)
        board = physical.load()["board"]
        trail = SnowTrail(board["length"], board["waist_width"], raw_env.q_slope)

        print(f"\n[play] {'no policy' if args.no_policy else args.run.name}  run {run_length:.0f} m, "
              f"speed command {env0(raw_env.v_cmd):.1f} m/s, fence ±{raw_env.fence_half_width:.0f} m")
        print(f"[play] {RunStats.HEADER}")
        recorder = Recorder(raw_env) if args.record else None
        longest_seconds, longest_marks, rows = ride(env, raw_env, agent, trail, recorder)
        write_ride_csv(trail_png.with_name(trail_png.stem.replace("trail", "ride") + ".csv"), rows)
        if recorder:
            recorder.save(args.record)

        trail.marks = longest_marks
        trail.save_png(trail_png, f"carve: {longest_seconds:.1f} s at {env0(raw_env.v_cmd):.1f} m/s commanded",
                       fence=raw_env.fence_half_width, x_end=run_length)
        print(f"[play] track of the longest episode ({longest_seconds:.1f} s) -> {trail_png}")
        env.close()


def write_ride_csv(path: Path, rows: list):
    lines = [",".join(RIDE_COLUMNS)] + [",".join(f"{v:.3f}" for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def play_overrides(run_overrides: dict, defaults) -> tuple[dict, float]:
    """Training's overrides, made repeatable for play; --set wins over everything. Returns (overrides, run length)."""
    user = parse_overrides(args.set)
    overrides = run_overrides | user

    def middle(name: str) -> float:
        lo, hi = overrides.get(name, getattr(defaults, name))
        return (lo + hi) / 2

    if "speed_cmd" not in user:
        overrides["speed_cmd"] = (middle("speed_cmd"),) * 2
    if "start_heading_deg" not in user:
        overrides["start_heading_deg"] = 0.0
    if "edge_cmd_level_init" not in user:
        overrides["edge_cmd_level_init"], overrides["edge_cmd_level_fixed"] = 1.0, True
    overrides["rsi_frac"] = 0.0

    run_length = args.run_length or 100.0
    overrides["run_length"] = run_length
    overrides["episode_length_s"] = 60.0  # only catches a rider who stops
    if not args.follow:
        overrides["viewer"] = free_view_cfg(overrides.get("slope_deg"), args.cam_x or run_length + 10.0)
    return overrides, run_length


def ride(env, raw_env, agent, trail, recorder=None) -> tuple[float, list, list]:
    """Returns the longest run's (seconds, trail marks, ride rows)."""
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

        if not done:  # a done step has already reset env 0 to the start
            stats.add(raw_env)
            trail.add(raw_env.body_pos(raw_env.board_id)[0], env0(raw_env.slip_angle()))
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
    """Follow cameras on env 0: "front" faces the rider's chest from 7 m, "top" looks down from 48 m so a full S of
    the track shows. Frames stay on the CPU and are written as GIFs at the end."""

    SIZE = (900, 600)
    FOCAL_MM = {"front": 30.0, "top": 18.0}
    FACING_SMOOTHING = 0.08

    @classmethod
    def add_cameras(cls, env_cfg):
        """Cameras must be in the scene cfg: added after the env is built, the rider never renders."""
        from isaaclab.sensors import CameraCfg

        for name, focal in cls.FOCAL_MM.items():
            setattr(env_cfg.scene, f"record_{name}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Record_{name}", width=cls.SIZE[0], height=cls.SIZE[1],
                data_types=["rgb"], update_period=0.0,
                spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.1, 2000.0))))

    def __init__(self, env):
        self.env = env
        self.frames = {name: [] for name in self.FOCAL_MM}
        self.cams = {name: env.scene[f"record_{name}"] for name in self.FOCAL_MM}
        self.facing = None

    def _smoothed_facing(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        env = self.env
        chest = quat_apply(env.body_quat(env.pelvis_id)[0], torch.tensor([1.0, 0.0, 0.0], device=env.device))
        chest[2] = 0.0
        chest = chest / chest.norm().clamp(min=1e-6)
        a = self.FACING_SMOOTHING
        self.facing = chest if self.facing is None else (1 - a) * self.facing + a * chest
        return self.facing / self.facing.norm().clamp(min=1e-6)

    def capture(self):
        env = self.env
        pelvis = env.body_pos(env.pelvis_id)[0]
        facing = self._smoothed_facing()
        up = torch.tensor([0.0, 0.0, 1.0], device=env.device)
        downhill = torch.tensor([1.0, 0.0, 0.0], device=env.device)
        views = {
            "front": (pelvis + 7.0 * facing + 1.0 * up + 1.5 * downhill, pelvis),
            # eye downhill of its target: uphill is image-up, so the track trails up the frame
            "top": (pelvis + 6.0 * downhill + 48.0 * up, pelvis - 14.0 * downhill - 0.8 * up),
        }
        for name, (eye, target) in views.items():
            self.cams[name].set_world_poses_from_view(env.to_world(eye)[None], env.to_world(target)[None])
        env.sim.render()
        for name, cam in self.cams.items():
            self.frames[name].append(cam.data.output["rgb"][0, ..., :3].cpu().numpy())

    def save(self, out_dir: Path):
        import imageio.v2 as imageio

        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frames in self.frames.items():
            path = out_dir / f"{name}.gif"
            every_other = frames[::2]  # 15 fps, real time
            imageio.mimsave(path, every_other, duration=2 * self.env.step_dt * 1000, loop=0)
            print(f"[play] {len(every_other)} frames -> {path}")


def free_view_cfg(slope_deg: float | None, cam_x: float):
    """Placed once, 0.6 m above the snow cam_x down the run, looking back at env 0's start; the mouse owns it after."""
    from isaaclab.envs import ViewerCfg

    from envs.carve_env import DEFAULT_SLOPE_DEG
    from envs.mountain import slope_to_world

    angle = slope_deg if slope_deg is not None else DEFAULT_SLOPE_DEG
    return ViewerCfg(eye=slope_to_world(angle, cam_x, 3.0, 0.6), lookat=slope_to_world(angle, 0.0, 0.0, 1.0),
                     origin_type="env")


def set_focal_length(cam_prim_path: str, focal_mm: float):
    from isaaclab.sim.utils.stage import get_current_stage

    prim = get_current_stage().GetPrimAtPath(cam_prim_path)
    if prim.IsValid():  # only with a viewer
        prim.GetAttribute("focalLength").Set(focal_mm)


if __name__ == "__main__":
    main()
