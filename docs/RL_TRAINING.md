# RL training

One env (`playground/snowboard/env.py`) teaches three tasks with plain **PPO** (`skrl`): **side slip**, **skid**
and **carve**. There's no demonstration and no style reward yet; the policy learns from task rewards alone. Knobs:
`SnowboardEnvCfg` (`--set key=value`), `playground/snowboard/ppo.yaml`. SI units; angles in rad unless marked °.

![Training loop: policy at 30 Hz, PD joints, snow model and PhysX at 120 Hz, rewards, PPO update every 24 steps](rider/training_loop.svg)

## 1. The task: follow an S line

Every task follows a line laid on the snow, with $x$ downhill and $y$ across:

$$y_{\text{ref}}(x) = Y \sin\!\left(\frac{2\pi x}{\lambda} + \varphi\right) \qquad Y = \frac{\lambda \tan A}{2\pi}$$

$\lambda$ is the turn length (40 m), and $A$ is the angle at which the line crosses the fall line (35°). Targets
depend on **where** the board is, not on time, so the turn size is independent of speed.

| Target | Formula |
|---|---|
| Path heading | $\psi^* = \arctan y_{\text{ref}}'(x) - \arctan\dfrac{y - y_{\text{ref}}}{\ell}$ (pure pursuit, $\ell$ = 6 m) |
| Slip angle | $\beta = \operatorname{atan2}(v_\perp, v_\parallel)$, the angle between the board's axis and its travel |

| Task | Line | $\beta^*$ | Track on the snow |
|---|---|---|---|
| Side slip | Straight | $-90°$ (heel edge) | Wide straight band |
| Skid | S | $25° \cdot \lvert\sin(2\pi x/\lambda + \varphi)\rvert$ | Wide S |
| Carve | S | $0$ | Thin S |

## 2. MDP

| | |
|---|---|
| **Observation** (129) | Humanoid 81: joint positions and velocities, torso height, orientation, velocities, hands and feet relative to the root (AMP layout) |
| | Board 6: $\theta$, $v_\parallel$, $v_\perp$, yaw rate, $N / mg$, grip usage $\lvert F_\perp\rvert / \mu N$ |
| | Last action 28 |
| | Task 14: line phase, $\psi^*$, $\beta^*$, heading / slip / line errors, speed and command, board yaw, position between the fences, $\lambda$, $Y$ |
| **Action** (28) | Joint targets around the stance (see [humanoid model §5](HUMANOID_MODEL.md#5-actions--joint-targets)), 30 Hz |
| **Done** | Pelvis below 0.45 m (**fell**), board past a fence at ±20 m (**fence**), or 12 s (**time-out**) |
| **Start** | Side slip: at rest on the heel edge, CoM above it. Turns: on the line at $x = 0$, at the commanded speed |

## 3. Reward

Each step, with $\Delta t = 1/30$ s and errors $e$:

$$r = \Delta t \sum_i w_i\, r_i \qquad r_{\text{track}}(e) = \exp\!\left(-\frac{e^2}{\sigma^2}\right)$$

| Term | $r_i$ | $w_i$ | $\sigma$ |
|---|---|---|---|
| Alive | 1 while not fallen | 1.0 | |
| Upright | $\max(0,\ \hat z_{\text{torso}} \cdot \hat n)$ | 1.0 | |
| Heading | $r_{\text{track}}(\psi - \psi^*)$ | 2.0 | 30° |
| Line | $r_{\text{track}}(y - y_{\text{ref}})$ | 2.0 | 2 m |
| Slip | $r_{\text{track}}(\lvert\beta\rvert - \beta^*)$ | 2.0 | 30° |
| Speed | $r_{\text{track}}(\lVert v\rVert - v^*)$ | 1.0 | 2 m/s |
| Pose | $\lVert q - q^{\text{stance}} \rVert^2$ | −0.1 | |
| Action rate | $\lVert a_t - a_{t-1} \rVert^2$ | −0.01 | |
| Torso wobble | $\lVert \omega_{\text{torso}} \rVert^2$ | −0.05 | |

$\sigma = 30°$ is deliberately wide: tighter, and riding straight earns ~nothing on a turn, so there's no gradient
toward turning.

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

A deterministic policy (mean action) rides **fixed, repeatable runs**: same start, middle of the speed and
turn-length ranges, two S's long (25 m for side slip). Each run prints one line of stats, and the longest one is saved
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
