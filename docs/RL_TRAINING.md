# RL training

One env (`playground/snowboard/env.py`) teaches three tasks with plain **PPO** (`skrl`): **side slip**, **skid**
and **carve**. There's no demonstration and no style reward yet; the policy learns from task rewards alone. Knobs:
`SnowboardEnvCfg` (`--set key=value`), `playground/snowboard/ppo.yaml`. SI units; angles in rad unless marked °.

![Training loop: policy at 30 Hz, PD joints, snow model and PhysX at 120 Hz, rewards, PPO update every 24 steps](rider/training_loop.svg)

## 1. The task: learn the turn

**No line is laid on the snow.** The turn tasks command a **speed** and reward **linked turns**. On a slope, the
only way to hold a speed below the straight-run speed is to turn across the hill, so the S (its length, its width,
its rhythm) is *learned*: it comes out of the physics of speed, slope, edge angle and grip.

| Task | Heading | Slip target $\beta^*$ | Track on the snow |
|---|---|---|---|
| Side slip | Straight down the fall line | $-90°$ (heel edge) | Wide straight band |
| Skid | **Free**: linked turns | $20°$ (the board pivots and slides) | Wide S |
| Carve | **Free**: linked turns | $0$ (clean edge) | Thin S |

| Quantity | Definition |
|---|---|
| Path heading | $\psi = \operatorname{atan2}(v_y, v_x)$: travel direction, 0 = straight down the fall line |
| Slip angle | $\beta = \operatorname{atan2}(v_\perp, v_\parallel)$: angle between the board's axis and its travel |
| **Linked turn** | $\psi$ crosses the fall line after swinging $\lvert\psi\rvert \ge 15°$ to one side |

**Why turns emerge.** Training runs on a **15° slope**. Going straight down it runs to ~24 m/s ([snow model §4](SNOW_MODEL.md)). Commanded
5–9 m/s, the rider must spend part of each turn heading across the hill, where gravity pulls less. How far across,
and how long each turn is, depends on speed, slope and how hard the edge can hold: the policy finds it.

## 2. MDP

| | |
|---|---|
| **Observation** | Humanoid: joint positions and velocities, torso height, orientation, velocities, hands and feet relative to the root (AMP layout) |
| | Board: $\theta$, $v_\parallel$, $v_\perp$, yaw rate, $N / mg$, grip usage $\lvert F_\perp\rvert / \mu N$ (+ two-half board: nose and tail edge angles, load on the nose, camber flattened) |
| | Last action |
| | Task (13): speed and command, $\psi$, $\beta$, board yaw, position between the fences, turn side, swing so far, time since the last turn, $\beta^*$ |
| **Action** | Joint targets around the stance (see [humanoid model §5](HUMANOID_MODEL.md#5-actions--joint-targets)), 30 Hz |
| **Done** | Pelvis below 0.45 m (**fell**), board past a fence at ±20 m (**fence**), or 12 s (**time-out**) |
| **Start** | Side slip: at rest on the heel edge, CoM above it. Turns: piste centre, heading within ±20° of the fall line, at the commanded speed |

| Model | DoF (actuated) | Observation | Action |
|---|---|---|---|
| Baseline rider, rigid board | 28 | 128 | 28 |
| New rider, two-half board | 30 (+2 passive board) | 142 | 30 |

## 3. Reward

Each step, with $\Delta t = 1/30$ s, plus a bonus per completed turn:

$$r = \Delta t \sum_i w_i\, r_i \;+\; w_{\text{turn}}\,\mathbb{1}[\text{turn completed}] \qquad r_{\text{track}}(e) = \exp\!\left(-\frac{e^2}{\sigma^2}\right)$$

| Term | $r_i$ | $w_i$ | $\sigma$ |
|---|---|---|---|
| Alive | 1 while not fallen | 1.0 | |
| Upright | $\max(0,\ \hat z_{\text{torso}} \cdot \hat n)$ | 1.0 | |
| **Speed** | $r_{\text{track}}(\lVert v\rVert - v^*)$, $v^* \sim U(5, 9)$ m/s (side slip: 0.5–2.5) | 2.0 | 2 m/s |
| Slip | $r_{\text{track}}(\lvert\beta\rvert - \beta^*)$ (side slip: $\beta - \beta^*$) | 2.0 | 15° |
| Heading | Side slip only: $r_{\text{track}}(\psi)$ | 2.0 | 30° |
| **Turn** | +1 per linked turn (turn tasks) | 1.0 | |
| Pose | $\lVert q - q^{\text{stance}} \rVert^2$ | −0.1 | |
| Action rate | $\lVert a_t - a_{t-1} \rVert^2$ | −0.01 | |
| Torso wobble | $\lVert \omega_{\text{torso}} \rVert^2$ | −0.05 | |

**What's not in the reward:** turn length, turn width, where on the piste to turn. Only the fences bound the piste.

## 4. PPO

| Setting | Value |
|---|---|
| Envs | 2048 in parallel |
| Rollout | 24 steps × 2048 = **49k samples per update** |
| Update | 5 epochs × 4 mini-batches |
| Training | 12 000 steps per env = **24.6M samples**, 500 updates (~30 min alone on an L40S) |
| Networks | Policy and value: separate MLPs, 512-256-128, ELU |
| Policy | Gaussian, initial $\sigma_a = e^{-1} \approx 0.37$ (bigger flails throw the bound rider) |
| Discount, GAE | $\gamma = 0.99$, $\lambda_{\text{GAE}} = 0.95$ |
| Learning rate | $10^{-3}$, KL-adaptive (target KL 0.01) |
| Clipping | Ratio 0.2, value 0.2, grad norm 1.0 |
| Entropy bonus | 0.01 |
| Normalisation | Running mean/std on observations and values |
| Time-outs | Bootstrapped (a time-out is not a failure) |

## 5. Evaluation (`playground/snowboard/play.py`)

A deterministic policy (mean action) rides **fixed, repeatable runs**: same start (piste centre, straight down),
middle of the speed range, 100 m long (25 m for side slip). Each run prints turns and metres per turn. Each run prints one line of stats, and the longest one is saved
as `trail.png` + `ride.csv`. `--record` films it for the README.

## 6. Learning curves (baseline)

![Learning curves of the baseline: episode length, falls, slip error, distance from the line, speed](results/learning_curves.png)

*Baseline, seed 1 (side slip: seeds 1–2), stopped at ~70% of training to switch to the new rider model.*

| Task | Falls | Slip error | Distance from line | Speed (target) |
|---|---|---|---|---|
| Carve | 11% | 3.3° | 2.1 m | 9.8 m/s (4–7) |
| Skid | 7% | 7.7° | 0.6 m | 7.0 m/s (4–7) |
| Side slip | 0% | 9.1° | 0.1 m | 1.6 m/s (0.5–2.5) |

**Reading the curves:**
- **Staying up comes first.** Falls drop from 100% to ~5% within 3M samples (side slip: 10M).
- **Line tracking comes second.** Turn tasks first ride straight and drift 9 m off the line, then learn to turn.
- **The carve can't slow down.** It holds 10 m/s against a 4–7 m/s command: on this slope and line, a clean carve
  settles at ~17 m/s and can only hold the tightest bend up to ~11.6 m/s, so it brakes by skidding a little
  (slip error rises late in training).
