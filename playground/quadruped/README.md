# Playground: Go2 quadruped gaits (PPO / SAC)

A warm-up before the snowboarder. A Unitree Go2 learns to follow a velocity command on flat ground, trained with
**PPO** or **SAC**. The gait is either left to emerge from the rewards (`free`) or imposed with a clock
(`walk`, `trot`, `pace`).

| File | What |
|---|---|
| `env.py` | The whole MDP: observations, actions, rewards, terminations, gait clock |
| `ppo.yaml`, `sac.yaml` | Networks + algorithm hyperparameters (skrl) |
| `train.py` | `--algo ppo\|sac --gait free\|walk\|trot\|pace --set key=value ...` |
| `play.py` | Runs a trained policy with a fixed command and prints a footfall diagram |

## Run

```bash
scripts/py playground/quadruped/train.py --algo ppo --gait free          # emergent gait
scripts/py playground/quadruped/train.py --algo ppo --gait pace          # clock-imposed gait
scripts/py playground/quadruped/train.py --algo sac --gait trot

scripts/py playground/quadruped/play.py logs/playground/quadruped/<run> --cmd 1.0 0 0
scripts/py playground/quadruped/play.py <run> --livestream 2 --seconds 120 # watch it (see docs/STAGE1_SIM.md)

.venv-sim/bin/tensorboard --logdir logs/playground/quadruped                # curves, per-term rewards
```

`Ctrl+C` stops training early and still saves a checkpoint.

**Watch it learn.** Add `--livestream 2` to `train.py` and connect the WebRTC client (see `docs/STAGE1_SIM.md`).
The robots on screen always run the current policy, so you see falls turn into steps as updates land. This is
about 2.5× slower than headless training (every frame is rendered), and Kit's first start takes a few
minutes. Each robot you see is one of the `num_envs` parallel copies. Run TensorBoard (with
`--bind_all`, port 6006) alongside it for the numbers.

## The MDP (`env.py`)

| | |
|---|---|
| **Observation** (56) | base lin/ang velocity, gravity direction, command, joint pos/vel, last action, clock (sin/cos per foot; zeros for `free`) |
| **Action** (12) | joint position targets = default pose + 0.25 · action; PD motors track them |
| **Reward** | weighted sum of the `rew_*` terms below, ×dt |
| **Done** | base touches the ground, or 20 s |
| **Command** | random (vx, vy, yaw rate) per episode |

## Gaits

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

## Knobs to play with

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
| `GAITS[...]["freq"]` | 1 → 3 Hz | step frequency (edit `env.py`) |

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
