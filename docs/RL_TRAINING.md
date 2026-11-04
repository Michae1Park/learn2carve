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
| **Linked turn** | $\psi$ crosses the fall line after swinging $\lvert\psi\rvert \ge 45°$ to one side: a real turn, across the hill |

**Why turns emerge.** Training runs on a **15° slope**. Going straight down it runs to ~24 m/s ([snow model §4](SNOW_MODEL.md)). Commanded
8–11 m/s and capped at 12 m/s, the rider must spend part of each turn heading across the hill, where gravity pulls less. How far across,
and how long each turn is, depends on speed, slope and how hard the edge can hold: the policy finds it.

**A guide to start from.** Learning to turn from nothing is slow, so the turn tasks begin with a guide S that the
policy traces, then fades out:

$$y_{\text{guide}}(x) = Y\sin\!\left(\frac{2\pi x}{\lambda} + \varphi\right),\quad Y = \frac{\lambda\tan A}{2\pi} \qquad A \sim U(35°, 50°),\ \varphi \in \{0, \pi\}$$

$\lambda$ is drawn per episode so the S stays carvable: tightest bend $\lambda / (2\pi\tan A) \ge 9$ m (sidecut 5–7 m),
swing $Y \le 15$ m (fences at ±20 m). The rider starts on the guide, heading along it, at 4–6 m/s.

| | Start of training | After 6000 steps (half of training) |
|---|---|---|
| Guide reward $w_{\text{guide}}\,r_{\text{track}}(\psi - \psi_{\text{guide}})$ | Weight 2.0 | **0**: faded out linearly |
| Guide in the observation | Heading error to the guide | **Zeros** (faded the same way) |
| Turn length, width, rhythm, how to press | Shaped by the guide | **The policy's own** |

Each episode draws its own S, so no single shape is learned. Play and evaluation always run with the guide off.

## 2. MDP

| | |
|---|---|
| **Observation** | Humanoid: joint positions and velocities, torso height, orientation, velocities, hands and feet relative to the root (AMP layout) |
| | Board: $\theta$, $v_\parallel$, $v_\perp$, yaw rate, $N / mg$, grip usage $\lvert F_\perp\rvert / \mu N$ (+ two-half board: nose and tail edge angles, load on the nose, camber flattened) |
| | Last action |
| | Task (16): speed and command, $\psi$, $\beta$, board yaw, position between the fences, turn side, swing so far, time since the last turn, $\beta^*$, guide heading error and fade |
| **Action** | Joint targets around the stance (see [humanoid model §5](HUMANOID_MODEL.md#5-actions--joint-targets)), 30 Hz |
| **Done** | Pelvis below 0.45 m (**fell**), board past a fence at ±20 m (**fence**), or 12 s (**time-out**) |
| **Start** | Side slip: at rest on the heel edge, CoM above it. Turns: piste centre, heading within ±20° of the fall line, at the commanded speed |

| Model | DoF (actuated) | Observation | Action |
|---|---|---|---|
| Baseline rider, rigid board | 28 | 131 | 28 |
| New rider, two-half board | 30 (+2 passive board) | 145 | 30 |

## 3. Reward

Each step, with $\Delta t = 1/30$ s, plus a bonus per completed turn:

$$r = \Delta t \sum_i w_i\, r_i \;+\; w_{\text{turn}}\,\mathbb{1}[\text{turn completed}] \qquad r_{\text{track}}(e) = \exp\!\left(-\frac{e^2}{\sigma^2}\right)$$

| Term | $r_i$ | $w_i$ | $\sigma$ |
|---|---|---|---|
| Alive | 1 while not fallen | 1.0 | |
| Upright | $\max(0,\ \hat z_{\text{torso}} \cdot \hat n)$ | 1.0 | |
| **Speed** | $r_{\text{track}}(\lVert v\rVert - v^*)$, $v^* \sim U(8, 11)$ m/s (side slip: 0.5–2.5) | 2.0 | 5 m/s |
| **Over-speed** | $\operatorname{clip}\big((\lVert v\rVert - v^*)/5,\ 0,\ 1\big)$ | −1.0 | |
| **Speed cap** | $\operatorname{clip}\big((\lVert v\rVert - 12)/2,\ 0,\ 1\big)$: forces turning | −3.0 | |
| **Fall** | Once, when the episode ends in a fall or past a fence | −10 | |
| **Edge** | Turn tasks: $\operatorname{clip}(\lvert\theta\rvert / \theta^*,\ 0,\ 1)$, $\theta^*$ = **40° carve**, 20° skid. Full reward from $\theta^*$ up, no penalty beyond: carving uses **40° and more** (up to ~85°) | 1.0 | |
| Slip | $r_{\text{track}}(\lvert\beta\rvert - \beta^*)$ (side slip: $\beta - \beta^*$) | 2.0 | 15° |
| Heading | Side slip only: $r_{\text{track}}(\psi)$ | 2.0 | 30° |
| **Turn** | +1 per linked turn (turn tasks) | 1.0 | |
| Guide | $r_{\text{track}}(\psi - \psi_{\text{guide}})$ × fade (turn tasks, first half of training) | 4.0 → 0 | 30° |
| Guide line | $r_{\text{track}}(y - y_{\text{guide}})$ × fade | 1.0 → 0 | 3 m |
| Pose | $\lVert q - q^{\text{stance}} \rVert^2$ | −0.1 | |
| Action rate | $\lVert a_t - a_{t-1} \rVert^2$ | −0.01 | |
| Torso wobble | $\lVert \omega_{\text{torso}} \rVert^2$ | −0.05 | |

**What's not in the reward:** turn length, turn width, where on the piste to turn. Only the fences bound the piste.

**Lessons from the reward (each one cost a training run):**

| Try | What happened | Fix |
|---|---|---|
| Speed Gaussian only, $\sigma$ = 2 m/s | Rode straight at 15 m/s (command 5–9): at 8 m/s over, $e^{-16} \approx 0$, no gradient toward slowing | $\sigma$ = 5 m/s + an over-speed term |
| Over-speed unbounded, $-0.5\,(v - v^*)$ | The rider speeds up through each episode, so late seconds cost more than staying up earns: it learned to **fall at ~5 s** | Penalties bounded (a second alive is worth ≥ 0) + a −10 fall penalty |
| Guide S of 25–60 m at 30–55° | Asked for 2.8 m bends, tighter than any carve: riders threw themselves at it | Guide ranges kept carvable |
| Speed command 5–9 m/s on 15°, mild cap | Rode straight at the 15 m/s cap, ignored the guide: the cap cost at most 1/s, less than staying up earned | Rider starts on the guide; guide line term |
| Start at 11–15 m/s on the guide | 77–98% of episodes ended at a fence in ~2.5 s: 8–10 m/s sideways, no time to learn the first turn | Start at 4–6 m/s; **cap 12 m/s at −3/s**, fall −30 |
| Cap 12 m/s, fall −30 | Learned to **ride flat**: edge angle 18° → 4°, straight at 13 m/s taking the cap penalty. No edge, no sidecut turn; −30 made trying to edge too costly | **Edge reward**; fall −10; guide weight 4 |
| A turn = a 15° swing | Learned to turn (edge 17–19°, 0 falls) but **gamed the turn bonus**: ±15° wiggles every 0.7 s (14–16 "turns" per 10 s), still 5 m/s over the command | A turn needs a **45°** swing |

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
