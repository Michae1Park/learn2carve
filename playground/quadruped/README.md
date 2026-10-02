# Playground: Go2 quadruped (PPO / SAC)

A warm-up before the snowboarder. A Unitree Go2 learns two separate tasks on flat ground:

- **Walk**: follow a velocity command, with an emergent or clock-imposed gait.
- **Backflip**: jump, rotate backward 360° and land, once per 2 s episode.

## Files

| Path | What |
|---|---|
| `common.py` | **Shared.** Run folders, `--set` overrides, train loop, policy loading, play camera |
| `go2.py` | **Shared.** Scene, robot, actions, reward bookkeeping, reset to stance |
| `ppo.yaml`, `sac.yaml` | **Shared.** Networks + algorithm hyperparameters (skrl) |
| `walk/` | `env.py` (walking MDP), `train.py`, `play.py` |
| `backflip/` | `env.py` (backflip MDP), `train.py`, `play.py` |

Runs from both tasks land in `logs/playground/quadruped/<date>_<time>_<algo>_<task>/`.

## Walk: train and play

```bash
# train (~4 min with PPO)
scripts/py playground/quadruped/walk/train.py --algo ppo --gait trot     # or free | walk | pace
scripts/py playground/quadruped/walk/train.py --algo sac --gait trot

# play
scripts/py playground/quadruped/walk/play.py logs/playground/quadruped/<run> --livestream 2 --seconds 60
scripts/py playground/quadruped/walk/play.py logs/playground/quadruped/<run> --cmd 1.0 0 0   # vx vy yaw
```

Play prints the speed and a footfall diagram.

## Backflip: train and play

```bash
# train (PPO, 30k steps by default)
scripts/py playground/quadruped/backflip/train.py

# play
scripts/py playground/quadruped/backflip/play.py logs/playground/quadruped/<run> --livestream 2 --seconds 60
```

Play prints one line per attempt (`rotations`, `landed` / `fell`, `FLIP`) and a total.

**Is it learning?** In TensorBoard, watch `Episode_Flip/rotations` (1.0 = one full flip) and
`Episode_Flip/success`. Early on it throws itself backward and crashes (`fell` near 1.0). That's normal: it
earns partial rotation credit. A 6k-step test went from 0 to 0.13 rotations. A full flip is **untested**, so
expect to tune (see [Backflip: knobs](#backflip-knobs)).

## Options (both tasks)

| Option | What it does |
|---|---|
| `<run>` | The folder `train.py` prints, e.g. `logs/playground/quadruped/2026-10-02_13-07-58_ppo_trot` |
| `--checkpoint agent_4000.pt` | Checkpoint from `<run>/checkpoints/` (default `best_agent.pt`). `ls` it first |
| `--set key=value` | Override an env field when training, e.g. `--set rew_gait=3 "cmd_vx=[0.5,1.5]"` |
| `--num_envs N` | Robots on screen. Play defaults to 1 |
| `--livestream 0` | No video (default) |
| `--livestream 1` | Stream over the internet. Needs `PUBLIC_IP=<ip>` and TCP 49100 + UDP 47998 open |
| `--livestream 2` | Stream on the LAN. Connect the WebRTC client to `192.168.33.118` |
| `--suppress_warnings false` | Show the ~4800 harmless `Failed to find rigid body` warnings (hidden by default) |
| `Ctrl+C` | Stop training early; still saves a checkpoint |

**Livestream tips**
- Connect once Kit logs that streaming is up (first start takes a few minutes).
- Use a long `--seconds` with play, or it ends before you connect.
- The camera starts close; orbit and zoom freely. It doesn't follow the robot.
- Streaming training is ~2.5× slower. More in `docs/STAGE1_SIM.md`.

**Safe to ignore at startup**
- 3 × `[Error] ... CUDA error ... capturing blocking stream` / `Failed to fetch DOF position attribute`:
  a one-time read clash at startup; the next step reads the state again.
- `env_cfg.viewer is deprecated`: Isaac Lab forwards the setting automatically.

**TensorBoard:** `.venv-sim/bin/tensorboard --logdir logs/playground/quadruped --bind_all` (port 6006)

## Walk: the MDP (`walk/env.py`)

| | |
|---|---|
| **Observation** (56) | base lin/ang velocity, gravity direction, command, joint pos/vel, last action, clock (sin/cos per foot; zeros for `free`) |
| **Action** (12) | joint position targets = default pose + 0.25 · action; PD motors track them |
| **Reward** | weighted sum of the `rew_*` terms below, ×dt |
| **Done** | base touches the ground, or 20 s |
| **Command** | random (vx, vy, yaw rate) per episode |

## Walk: gaits

**`free`.** No clock. The only gait-related term is `rew_air_time`, which rewards steps that last longer than
0.5 s. With the default weights, PPO tracks the command well but with an irregular, shuffling footfall. You
steer the gait only indirectly, through reward weights. Raising `rew_air_time` is the first thing to try.

**`walk` / `trot` / `pace`.** A clock goes around once per gait cycle. Each foot has its own phase offset, and
a foot should be on the ground while `(clock + offset) mod 1 < duty`. `rew_gait` pays for the fraction of
feet that match. The policy sees the clock, so it can learn to step in time with it.

| Gait | Offsets FL FR RL RR | Duty | Feet together |
|---|---|---|---|
| walk | .25 .75 0 .5 | 0.75 | none, 4 beats, ≥3 feet down |
| trot | 0 .5 .5 0 | 0.5 | diagonals (FL+RR, FR+RL) |
| pace | 0 .5 0 .5 | 0.5 | same side (FL+RL, FR+RR) |

A new gait is one line in `GAITS`. For example, bound (fronts together, rears together) is `(0, 0, .5, .5)`.

## Walk: knobs

Pass any `Go2EnvCfg` field with `--set`, e.g. `--set rew_gait=3 "cmd_vx=[0.5,1.5]"`.

| Knob | Try | Expect |
|---|---|---|
| `rew_alive` | 0 | per-step reward goes negative early on, so falling fast becomes optimal (SAC collapses) |
| `rew_gait` | 0 → 3 | 0 = clock ignored; high = clean pattern, worse velocity tracking |
| `rew_air_time` | 0 / 1.0 | 0 = shuffling, tiny steps; high = long, high steps (or hopping) |
| `rew_action_rate` | 0 / -0.1 | 0 = jittery legs; strong = smooth but sluggish |
| `rew_orientation`, `rew_base_height` | 0 | body pitches, crouches, or rests on its legs |
| `rew_torque` | -2e-3 | lazier, energy-saving gait |
| `cmd_vx` | `[1.5,2.5]` | fast commands push `free` toward bounding/galloping |
| `action_scale` | 0.1 / 0.5 | smaller = stiff, easier to learn; larger = more reach, more chaos |
| `GAITS[...]["freq"]` | 1 → 3 Hz | step frequency (edit `walk/env.py`) |

In `ppo.yaml` / `sac.yaml`:

| Knob | Algo | Effect |
|---|---|---|
| `entropy_loss_scale` | PPO | more = keeps exploring longer |
| `rollouts`, `mini_batches`, `learning_epochs` | PPO | batch size and how hard each batch is reused |
| `gradient_steps`, `batch_size` | SAC | updates per env step (speed vs stability) |
| `target_entropy`, `learn_entropy` | SAC | how much randomness SAC keeps |
| `rewards_shaper_scale` | SAC | at 1 the entropy bonus outweighs the tiny ×dt rewards, so SAC learns to stand still |
| `num_envs` | both | PPO loves thousands; SAC is fine with fewer |

TensorBoard shows each reward term separately under `Episode_Reward/*`. Watch which term moves when you
change a weight.

## Backflip: the MDP (`backflip/env.py`)

Every episode follows the same 2 s timeline, and the policy sees it as a phase in [0, 1]:

```
0 s ───── 0.5 s ═══════ 1.0 s ──────────── 2.0 s
  stand / crouch   jump + rotate    land, stand still
```

| | |
|---|---|
| **Observation** (47) | base lin/ang velocity, gravity direction, joint pos/vel, last action, phase, flip progress (rotations so far) |
| **Action** (12) | joint position targets = default pose + 0.5 · action (twice the walk's, to jump high enough) |
| **Main reward** | `track_flip`: the flip angle should follow a target that ramps 0 → 360° across the jump window |
| **Helper rewards** | `spin` (backward pitch rate) and `height` in the window; `feet_down` and `pose` outside it |
| **Done** | base or head touches the ground, or 2 s. All robots start in sync (no reset spreading) |

## Backflip: knobs

Pass any `BackflipEnvCfg` field with `--set`.

| Knob | Try | Expect |
|---|---|---|
| `rew_height` | 20 | jumps higher, more airtime to finish the rotation |
| `rew_spin` | 1.0 | spins harder; too high and it over-rotates or tumbles |
| `rew_track_flip` | 10 | sticks closer to the 0 → 360° schedule |
| `action_scale` | 0.7 | more leg travel and a stronger push-off, but noisier early on |
| `flip_start`, `flip_end` | 0.5, 1.1 | a longer window gives a slower, easier rotation |
| `--timesteps` | 60000 | more training if `rotations` is still climbing |

## PPO vs SAC in one paragraph

**PPO** is on-policy. It collects a fresh batch from all envs, takes a few small gradient steps that it clips
so the policy can't move far, then throws the batch away. It's simple, stable, and scales with parallel envs,
which is why it's the default for sim locomotion. **SAC** is off-policy. It keeps a replay buffer of past
transitions and learns a Q-function (two of them, to avoid overestimating values), and the policy maximises
Q plus an entropy bonus. It's more sample-efficient per env step, but each step costs more compute and it is
more sensitive to hyperparameters.

Measured here on an L40S:

| Run | Time | Result |
|---|---|---|
| PPO, 15k steps | ~4 min | clean trot / pace / walk at the commanded speed |
| SAC trot, 8k steps | ~9 min | doesn't fall and steps to the clock, but only 0.12 of 0.8 m/s. Still improving; use the default 20k |
