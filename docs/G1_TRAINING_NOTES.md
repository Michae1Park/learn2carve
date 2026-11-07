# G1 training notes

How the stock Unitree G1 learned to carve: the setup that worked, the training chain, what failed on the way and why.
Companion to [RL_TRAINING.md](RL_TRAINING.md) (the stages, rewards and env) and
[HUMANOID_MODEL.md](HUMANOID_MODEL.md). SI units; angles in ° unless marked rad.

## Result

The final policy is `saved/g1/chain/07_stage7_g1_7065_finish60`. Play it with `--set robot=g1`.

| Eval (`play.py`, 120 s at the full command) | G1 (`chain/07`) | AMP rider (`saved/04_stage4_tighter`) |
|---|---|---|
| Runs that reach the end of the 100 m course | **6 / 6**, no falls | 5 / 5 |
| Run time | 17.3–18.5 s | 15–17.5 s |
| Linked turns per run | **6** (16.6 m per turn) | 4–5 |
| Edge, mean | **57–59°** | 39–40° |
| Mean \|heading\| (how far across the hill) | 42–44° | 35–39° |
| Slip | 1.5° | 3° |
| Track width | within ±11 m (fence ±20 m) | within ±15 m |

Demo GIFs: `docs/demo/05_g1_front.gif`, `05_g1_top.gif` (a thin S) and `05_g1_training.gif` (64 riders).

## The rider and its setup

The robot is not modified: Isaac Lab's `G1_29DOF_CFG` masses, joint ranges and motor limits (`envs/rider_g1.py`). Only
the board, the stance, the bindings and the drive gains are tuned.

| | Value | Where |
|---|---|---|
| Board | 1.42 m long, 0.230 m waist, nose/tail 0.270 m, effective sidecut 7.1 m, 1.5 kg, setback −0.015 m | `envs/physical_g1.yaml` |
| Bindings | **70° front / 65° back** (both toward the nose) | stage files |
| Stance width | **0.42 m** binding centre to centre | stage files |
| Stance pose | `stance_70_65_g1`: hips 0.10 m toward the tail, pelvis hinged **50° forward** (abdomen down toward the front thigh), pelvis 0.60 m above the board, pelvis opened 45° | `envs/assets/stances/stance_70_65_g1.yaml` |
| Legs, waist drives | Isaac Lab stock (`g1_gains: isaac`, implicit drives) | `envs/rider_g1.py` `isaac_actuators` |
| Arm drives | **kp 300 / kd 15** (`g1_arm_gains`), stock is kp 3000 / kd 10 | `envs/rider_g1.py`, `envs/carve_env.py` |
| Arm actions | `arm_action_scale: 0.5` | stage files |
| Edge synergy | knees +0.5 rad, ankle pitch −0.3 rad per unit | stage files |

### Building the stance

The stance comes from `scripts/make_stance.py`, which pins the board and pelvis, lets the soft-driven joints settle into
the bindings and reports how far each foot misses its binding:

```
scripts/py scripts/make_stance.py --robot g1 --tag g1 --angles 70 65 --stance_width 0.42 \
    --hip_height 0.60 --open_deg 45 --tail 0.10 --pitch 50 --seconds 10
```

A usable stance fits both bindings with near-zero residual. What the sweeps showed (back = right foot):

| Bindings | Hip height | Pelvis open | Tail shift | Forward hinge | Back binding residual |
|---|---|---|---|---|---|
| 72/60 | 0.65 | 45° | 0 | 0 | 11 mm, 6.1° |
| 72/60 | 0.60 | 45° | 0 | 0 | 25 mm, 13.8° |
| 72/60 | 0.57 | 45° | 0 | 0 | 28 mm, 18.8° |
| 70/65 | 0.65 | 45° | 0 | 0 | 11 mm, 5.9° |
| 70/65 | 0.60 | 45° | 0 | 0 | 25 mm, 13.5° |
| 70/65 | 0.65 / 0.60 | 55° | 0 | 0 | 12 mm, 6.3° / 27 mm, 14.1° |
| 70/65 | 0.65 / 0.60 | 65° | 0 | 0 | 13 mm, 6.5° / 28 mm, 14.7° |
| 70/65 | 0.65 | 45° | 0.05 m | 0 | 7.6 mm, 4.0° |
| 70/65 | 0.65 | 45° | 0.10 m | 0 | 2.4 mm, 1.3° |
| 70/65 | 0.60 | 45° | 0.10 m | 0 | 14.8 mm, 7.8° |
| 70/65 | 0.65 | 45° | 0.10 m | 30° / 45° / 60° | 0.07 mm, 0° |
| **70/65** | **0.60** | **45°** | **0.10 m** | **50°** | **0.08 mm, 0°** |
| 70/65 | 0.55 | 45° | 0.10 m | 50° | 7.2 mm, 3.8° |

- **The limit is the back ankle.** In every failing stance the back ankle pitch sits at its dorsiflexion limit,
  −0.873 rad (−50°): the back shin cannot lean further forward, so lower hips pull the back foot off its binding.
- **Opening the pelvis does nothing** for it (45°, 55°, 65° all miss by the same amount).
- **Shifting the hips toward the tail** stands the back shin up; **hinging the pelvis forward** stands it up further, and
  with both the hips can go 5 cm lower. In the chosen stance the back ankle is at −0.83 rad, just inside its limit.
- 0.55 m is too low again: the ankle is back at its limit.
- The settle can leave a joint a rounding hair past its limit (−0.87271 vs −0.87266), which Isaac Lab refuses at
  launch ("default positions out of the limits"). Clamp it inside by hand in the stance file.

## Training chain

Each stage is warm-started from the last run of the stage before (`--init_from <run>`); within a stage, a run that was
still improving was continued with the same settings. Every run is 16 000 steps (~40 min) with `--livestream 2`.

```
scripts/py playground/snowboard/train.py --stage stage2_g1_7065 --set robot=g1 --livestream 2
scripts/py playground/snowboard/train.py --stage stage3_g1_7065 --set robot=g1 --livestream 2 --init_from <previous run>
...
```

### Stage files

| Stage file | Differs from the stage before | Why |
|---|---|---|
| `stage2_g1_7065` | Stage 2 (`stage2.yaml`: edge flips on a timer, carve start, 7° slope, 10 s episodes, amplitude 15–30°) on the 70/65 setup | Learn the edge change first, from scratch |
| `stage3_g1_7065` | Stage 3 (`env.yaml`: full S, edge switch on heading, 9° slope, 20 s episodes, amplitude 25–45°) | Link the flips into an S |
| `stage3_g1_7065` + `g1_arm_gains: [300, 15]` | Arm drives only | Stop the arm wobble (see below) |
| `stage4_g1_7065_edge` | `edge_cmd_amp_deg: [35, 55]`, `edge_cmd_level_init: 0.5` | Push the edge past 40° |
| `stage5_g1_7065_edge65` | `[45, 65]`, start at 55° | Deeper |
| `stage6_g1_7065_edge75` | `[55, 75]`, start at 65° | Deeper |
| `stage7_g1_7065_finish60` | `edge_cmd_heading_deg: 60` (was 50), `edge_cmd_level_init: 1.0` | Finish each turn further across the hill |

Each edge push starts its curriculum halfway, at the previous stage's top command, so the warm-started policy is never
asked for less than it already does.

### Metrics, run by run

Training metrics, averaged over the last 20 % of each run. Runs kept in `saved/g1/chain` are marked with their folder.

| # | Stage | Turns / 10 s | Edge | Falls | Speed | \|Heading\| | Saved as |
|---|---|---|---|---|---|---|---|
| 1 | stage 2 | 0.07 | 17.8° | 16 % | 8.3 m/s | 15° | `chain/01_stage2_g1_7065` |
| 2 | stage 3 | 0.54 | 32.0° | 17 % | 10.9 | 21° | |
| 3 | stage 3, continued | 0.95 | 37.5° | 8 % | 10.9 | 24° | |
| 4 | stage 3, continued | 1.23 | 38.9° | 5 % | 10.9 | 25° | `chain/02_stage3_g1_7065` |
| 5 | stage 3 + arm gains | 2.12 | 30.4° | 9 % | 9.5 | 31° | |
| 6 | continued | 2.53 | 38.0° | 17 % | 10.0 | 36° | |
| 7 | continued | 2.86 | 37.6° | 2 % | 10.6 | 35° | `chain/03_stage3_g1_7065_arms` |
| 8 | stage 4 (55°) | 3.42 | 43.4° | 4 % | 10.1 | 38° | |
| 9 | continued | 3.76 | 44.6° | 2 % | 9.7 | 37° | `chain/04_stage4_g1_7065_edge` |
| 10 | stage 5 (65°) | 4.36 | 50.0° | 4 % | 8.3 | 39° | |
| 11 | continued | 4.24 | 51.7° | 4 % | 8.6 | 39° | `chain/05_stage5_g1_7065_edge65` |
| 12 | stage 6 (75°) | 4.02 | 55.0° | 6 % | 8.5 | 39° | |
| 13 | continued | 4.06 | 55.6° | 5 % | 8.5 | 39° | `chain/06_stage6_g1_7065_edge75` |
| 14 | stage 7 (switch at 60°) | 3.52 | 56.0° | 6 % | 8.4 | 43° | |
| 15 | continued | 3.42 | 57.2° | 6 % | 8.7 | 43° | `chain/07_stage7_g1_7065_finish60` |

Other end-of-run numbers: stage 2 ended at curriculum level 1.0, edge-command error 12.0°, CoM lean 26°, 9.2 s of 10 s
episodes. From stage 4 on the fence was hit in 0–2 % of episodes, and the curriculum level stayed at 1.0. The
edge-command error fell to ~6° by stage 4 and settled at ~10° from stage 6 (a 75° command is about what the G1
holds).

Reading the curve:

- **Stage 2 → 3:** stage 2 teaches the flip (CoM lean rose from 2° in the failed 60/50 stage 3 to 26°); stage 3 turns
  it into an S, edge 32 → 39° over three runs, but only ~1 turn per 10 s.
- **Arm gains:** turns nearly doubled at once (1.23 → 2.12), edge dropped to 30° for one run and recovered to 38° in
  the next. Falls spiked to 17 % in run 6 and fell to 2 % in run 7.
- **Edge pushes:** each raised the edge by 5–7° and the turn rate with it, up to 4.36 turns / 10 s at 50°. At 75° the
  edge-command error stopped improving (~10°) and turns dipped slightly: the push has reached the robot's limit.
- **Turn finish at 60°:** heading 39 → 43°, as intended; turns per 10 s dropped (3.5) because each turn takes longer to
  come round. Edge kept rising (57°).

## What failed, and what was learned

### Before the 70/65 stance

| Attempt | Outcome | Lesson |
|---|---|---|
| Unitree deployment gains (`g1_gains: unitree`) fine-tuned from the stock-gain 60/50 carver | Policy flattened, edge ~6° | Waist kp 40 can't make torque; with waist kp 200 edges improved but the legs (knee 150, ankle 40 / kd 2) sit down. The stock carver uses the waist at its torque limit 62–69 % of the time |
| 30/15 bindings (pelvis open 60° + waist yaw 30°, facing the nose) | Stalled at ~6° edges, even with a refit edge synergy | 60/50 with stock gains carves; 30/15 does not |
| Adult board | Worse | The smaller board (1.42 m / 0.230 m / 1.5 kg) was the big win |
| 60/50 + stock gains, stage 3 **from scratch** | Never flipped edges: CoM lean 2° vs edge 16°, 60 % of falls on the side | The edge was tipped but nothing counter-balanced it. Learn the flip in stage 2 first |
| 60/50 stage 2 from scratch | Promising at 26 % (falls 22 %, CoM lean 16°), stopped for the stance change | — |

The best 60/50 policy reached 34° edge, 2.17 turns / 10 s, 5.6 % falls; the 70/65 chain passed it on every metric.

### During the 70/65 chain

| Attempt | Outcome | Lesson |
|---|---|---|
| Upright 70/65 stance (0.65 m hips, no hinge) | Stopped early for the hinged stance | Hips back and the upper body low on the front thigh is the starting pose |
| Stage 3 continued with no change (from run 4) | Turns 1.23 → 0.99 | Plain continuation had plateaued; change one thing instead |
| `edge_cmd_max_s: 2.5` (hold an edge at most 2.5 s, was 4) | **Worse:** turns 0.29, heading 18°, faster (11.4 m/s). Reverted | Switching sooner makes shallow turns that never come round far enough to count (the counter needs a 55° path swing); the rider just runs straighter |
| Stage 6 continued a third time | Plateau (55.8°, 3.89 turns) | Moved on to the turn-finish change |

### The arm wobble

The stock arm drives are kp 3000 / kd 10 with 0.001 armature. For an arm link, critical damping at kp 3000 is roughly
kd 30–40, so the stock drive is 3–4× underdamped and every small action change rings. `arm_stiffness_scale` (the
AMP rider's fix) does not reach the G1 path, so `g1_arm_gains: [kp, kd]` was added: it replaces the stiffness and
damping of the `arms` actuator group only, leaving legs and waist on stock gains. kp 300 / kd 15 is near critical,
still holds the arms against the carve's load, and is closer to the real robot. It also changed the riding: slower,
further across the hill, nearly twice the turns.

## Code changes

| File | Change |
|---|---|
| `envs/rider_g1.py` | `rider_cfg(..., arm_gains=(kp, kd))` overrides the `arms` actuator group |
| `envs/carve_env.py` | `g1_arm_gains: tuple[float, float] \| None` config field, passed to the G1 rider |
| `envs/assets/stances/stance_70_65_g1.yaml` | The 70/65 hinged stance |
| `playground/snowboard/stages/stage2_g1_6050.yaml` | 60/50 stage 2 from scratch (superseded) |
| `playground/snowboard/stages/stage2_g1_7065.yaml` | 70/65 stage 2 |
| `playground/snowboard/stages/stage3_g1_7065.yaml` | 70/65 stage 3, with the arm gains |
| `playground/snowboard/stages/stage4_g1_7065_edge.yaml` | Edge command 35–55° |
| `playground/snowboard/stages/stage5_g1_7065_edge65.yaml` | Edge command 45–65° |
| `playground/snowboard/stages/stage6_g1_7065_edge75.yaml` | Edge command 55–75° |
| `playground/snowboard/stages/stage7_g1_7065_finish60.yaml` | Edge switch at 60° across the hill |
| `docs/demo/05_g1_{front,top,training}.gif` | G1 demo GIFs |
| `README.md` | GIFs show the G1 |
| `saved/g1/chain/01–07` | Last run of each stage |

Runs 1–4 were trained before `g1_arm_gains` was added to `stage3_g1_7065`; to reproduce them, use the `run.yaml`
saved with each run rather than the current stage file.

## Saved runs

`saved/g1/chain/NN_<stage>/` holds the last run of each stage: `checkpoints/best_agent.pt`, `run.yaml` (the full
config it trained with), `physical.yaml`, `ppo.yaml`, the stance file and a README with its metrics. `07` also has
`ride.csv` and `trail.png` from the eval above. The G1 runs are kept apart from the AMP rider's `saved/01`–`04`.

## Recording the demos

```
scripts/py playground/snowboard/play.py <run> --seconds 90 --record <dir>     # front.gif, top.gif (cameras on)
scripts/py playground/snowboard/play.py <run> --seconds 120                   # eval table, ride.csv, trail.png
scripts/py scripts/record_fleet.py <run> --num_envs 64 --seconds 12 --out <dir>
ffmpeg -i <dir>/fleet_0000.mp4 -vf "fps=50/3,crop=1080:720,scale=900:600:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" training.gif
```

`--record` films one run and enables the cameras itself (no `--enable_cameras` flag). `play.py` overwrites the run's
`trail.png`.

## Practical notes

- Kill a training run by PID: `pgrep -f "^<repo>/.venv-sim/bin/python playground/snowboard/train.py"`. An
  unanchored pattern also matches the calling shell.
- Python occasionally crashes at startup (seen here inside the YAML reader with unchanged files); relaunch.
- A run's training metrics live in its tfevents file under `Episode_Snow/`, `Episode_Termination/` and
  `EdgeCmd/level`; average the last 20 % of a run when comparing, single points are noisy.
- One Isaac Sim training run with livestream uses ~8.5 GB of VRAM.

## Next steps

- **Sim2real:** move toward Unitree's deployment gains one joint group at a time, starting with the legs.
- **Edges past 75°:** likely trades turns and falls; the edge-command error already plateaus at 75°.
- **Turn finish:** switch at 70° across the hill if 43° heading is not enough.
- **README text** still describes the AMP rider.
