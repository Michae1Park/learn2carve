# Playground: snowboarding humanoid (PPO)

The Stage 1 rider (`envs/carve_env.py`: AMP humanoid bound to a board, `envs/snow.py` snow model) rides down a
fenced piste on a mountain (`envs/mountain.py`) and learns three tasks with plain RL, no demo clip yet:

- **Side slip**: board across the fall line, sliding straight down on the heel edge.
- **Skid**: S-turns where the board pivots and slides through each turn. Leaves a wide S.
- **Carve**: S-turns on a clean edge, no sideways slip. Leaves a thin S line.

Turns follow an S line laid on the snow with a set **turn length** (metres per full S), and must stay between
the fences: crossing one ends the episode.

## Files

| Path | What |
|---|---|
| `env.py` | `SnowboardEnv(CarveEnv)`: task targets, observations, rewards, start state |
| `../../envs/mountain.py` | Scenery: terrain (snow + rock), fenced piste, forest, sky, sun (visual only) |
| `../../envs/rider.py` | Rider + board; also the outfit (red jacket, dark pants, black helmet, blue board) |
| `trail.py` | The track on the snow: live markers in the viewport + top-down `trail.png` |
| `train.py`, `play.py` | Train / ride a policy |
| `ppo.yaml` | Networks + PPO hyperparameters (skrl) |

Shared helpers (run folders, `--set`, train loop, policy loading) come from `playground/quadruped/common.py`.
Runs land in `logs/playground/snowboard/<date>_<time>_ppo_<task>/`.

## Train and play

```bash
scripts/py playground/snowboard/train.py --task sideslip
scripts/py playground/snowboard/train.py --task skid  --set slope_deg=10
scripts/py playground/snowboard/train.py --task carve --set slope_deg=10

scripts/py playground/snowboard/play.py logs/playground/snowboard/<run>                        # prints + trail.png
scripts/py playground/snowboard/play.py logs/playground/snowboard/<run> --livestream 2 --seconds 120
scripts/py playground/snowboard/play.py --no_policy --task sideslip                            # stance only
```

Play repeats **short evaluation runs**: each ends two full S's down the piste (`--run_length`, default
2 × turn length; 25 m for the side slip), always from the same start (piste centre, fixed phase, mid-range speed
and turn length), then the rider goes back to the top. `--seconds` is the total play time. It prints one line per
run (end: done / fell / fence, seconds, speed, edge angle, heading / slip / line errors, widest swing, turns) and
saves the longest run's track, with the target line and the fences, to `<run>/trail.png` (downhill = down). With `--livestream`, the same
track is drawn on the snow behind the rider (blue strips; width = snow the board sweeps sideways).

**Camera:** free by default, like the quadruped's. It starts like a filmed carving clip: low on the snow
(0.6 m) just past the end of the run, looking back up at the start through a 35 mm lens (`--focal`), so the
rider carves toward you against the sky. Then it's yours: orbit, pan, zoom. `--cam_x <m>` moves it along the
run; `--follow` tracks the rider instead.

**The mountain:** the piste runs down a valley. Its sides rise from gentle forested slopes into rocky ridges, and
a crest closes it above the start, low enough that the bottom camera sees sky above it. An orange safety net on
poles fences the piste at **±20 m** (a 40 m piste, from just above the start to 400 m down). All of it is visual
only (the physics is one tilted plane), but the env ends an episode when the board crosses a fence line.

Other options are the quadruped's: `--checkpoint`, `--num_envs`, `--set`, `--suppress_warnings`, `--livestream`
(see `playground/quadruped/README.md`).

**TensorBoard:** `.venv-sim/bin/tensorboard --logdir logs/playground/snowboard --bind_all`. Watch
`Episode_Termination/fell` and `/fence`, `Episode_Snow/seconds`, `err_heading_deg`, `err_slip_deg` and
`err_line_m`.

## The MDP (`env.py`)

Turns follow an S line on the snow, centred on the piste:

```
y_ref(x) = Y · sin(2π x / λ + φ)     λ = turn length (m per full S),  Y = λ · tan(A) / 2π  (crosses the fall line at ±A)

path heading  ψ_ref   the line's direction at the board, plus a pull back onto it (pure pursuit over `lookahead`)
slip angle    β*      angle between the board's long axis and its travel direction

task       line        β*                            track
sideslip   straight    -90° (heelside)               wide straight band
skid       S, λ        slip_max · |sin(2πx/λ + φ)|   wide S (slides most at each turn's apex)
carve      S, λ        0                             thin S line
```

Targets depend on **where** the board is, not on time, so turn size is independent of speed and the S can't drift
off the piste. Turn length: short ≈ 20 m, medium 40 m (default), long 80 m. The tightest bend of the line is
λ² / (4π² Y); below the board's sidecut radius (~5–7 m) it can only be skidded, so short turns are a skid exercise.

| | |
|---|---|
| **Observation** (129) | Stage 1 obs (AMP humanoid 81 + board/snow 6), last action, line phase sin/cos, both targets, heading / slip / line errors, speed, speed command, board yaw sin/cos, position between the fences, λ, Y |
| **Action** (28) | joint targets = snowboard stance + 0.25 · half-range · action |
| **Reward** | `alive`, `upright`, `heading`, `line`, `slip`, `speed` (Gaussian around targets), penalties `pose`, `action_rate`, `torso_ang_vel`. All × dt |
| **Done** | pelvis below 0.45 m (fell), board past a fence, or 12 s |
| **Start** | side slip: at rest on the heel edge, centre of mass above it. Turns: on the S line at x = 0, commanded speed |

The policy never steers the board directly. It tips the board with its body, and the snow model turns an edged
board along its sidecut, or lets a flat / over-pivoted one skid.

## Knobs

Pass any `SnowboardEnvCfg` field with `--set`.

| Knob | Default | Effect |
|---|---|---|
| `slope_deg` | 15 (config.yaml) | gentler = slower, easier turns |
| `mountain` | true | false = Stage 1's flat ground + tilted gravity (same physics, no scenery) |
| `turn_length` | (40, 40) m | λ per full S, sampled per episode: `"turn_length=[20,80]"` trains all sizes in one policy |
| `heading_amp_deg` | 35 | angle at which the line crosses the fall line (sets the swing Y) |
| `lookahead` | 6 m | how hard the heading target pulls back onto the line |
| `slip_max_deg` | 25 | skid only: slip at each turn's apex |
| `fixed_phase` | None (play: 0) | where on the S each episode starts; 0 = piste centre |
| `fence_half_width` | 20 m | episode ends past it. Visual fence: `envs/mountain.py` `FENCE_HALF_WIDTH` |
| `run_length` | None (play: 2 λ) | end the episode this far down (as a time-out) |
| `speed_cmd` | sideslip (0.5, 2.5), turns (4, 7) m/s | sampled per episode |
| `sigma_deg`, `sigma_line`, `sigma_speed` | 30°, 2 m, 2 m/s | reward tolerance. Too tight and there's no gradient toward turning |
| `rew_*` | see env.py | term weights |
| `action_scale` | 0.25 | 0.5 (Stage 1's) flails and falls |

## Results

_Line-following version (fenced piste): see below once trained._

### First version (time-based targets, no fence)

PPO, 2048 envs, 12k steps (~10 min alone on an L40S, ~25 min sharing it). Play = 20 s episode (twice the training
length), deterministic policy.

| Run | Slope | Fell (train) | Play | Track |
|---|---|---|---|---|
| sideslip | 15° | 0% | 20 s up, 2.3 m/s sideways, heading err 4°, slip err 3°, edge 2.5° | straight 1.5 m band, 45 m |
| skid | 10° | 4% | 20 s up, 6.5 m/s, 10 turns, heading err 11°, slip err 6° | clean S, 110 m, smears at turn ends |
| carve | 10° | 4% | 20 s up, thin line (slip err 0.6°), 7 turns | S for ~10 s, then speeds past 20 m/s and the turns flatten |
| carve, `heading_amp_deg=50 turn_period=5 episode_length_s=20` | 8° | 5% | 1–3 turns, heading err 25° | mostly straight. Still improving at 12k: needs more steps |

**Carve's open problem is speed.** A clean carve barely brakes, so ±35° turns still run away downhill past the
training horizon. Next things to try: more training on the wide-turn setup above (`--timesteps 30000`), or a
speed-dependent target (wider turns as speed grows), or a gentler slope.

## Gotchas

- **Mountain vs. tilted gravity is the same MDP.** The env reads every body state in the slope frame
  (`body_pos` / `body_quat` / `body_lin_vel` / `body_ang_vel`) and writes poses and forces back with `to_world`.
  Policies trained on Stage 1's flat-ground setup ride the mountain unchanged (side slip and skid measured
  within 0.1 m/s and 0.1° of their flat-ground numbers). Don't read `robot.data.*_w` directly in env code.

- **Start the side slip from rest, on the heel edge.** The stance is built perpendicular to the snow, so flat
  on 15° it leans the rider downhill and they catch the toe edge in ~0.7 s. With any downhill speed at reset,
  the biting edge stops the board in one step and the body trips over it.
- Bound rider + board tip as one inverted pendulum on a 0.25 m base. With zero action (`--no_policy`) the
  side slip falls in ~1.5 s, but nose-first riding stays up and even wobbles into small turns on its own.
- Changing `rider.binding_angles_deg` in `config.yaml` needs `scripts/make_stance.py` re-run. A flatter stance
  (e.g. 15° / 0°) gives a wider base for heel/toe balance if the forward 42° / 27° stance won't learn.
