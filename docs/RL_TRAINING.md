# RL training

One env (`playground/snowboard/env.py`) and plain **PPO** (`skrl`). The rider learns linked carved turns by
**tracking a commanded edge angle**: the command plays the part of the turn planner, and the policy learns to put the
board on that edge and stay up. Training runs in **stages**, each warm-started from the last. No demonstration, no
motion data. SI units; angles in ° unless marked rad.

![Training loop: policy at 30 Hz, PD joints, snow model and PhysX at 120 Hz, rewards, PPO update every 48 steps](rider/training_loop.svg)

**One training step.** The env gives the rider a target edge angle $\theta^\ast$ (the *edge command*). The policy sees
the body state and $\theta^\ast$, and outputs joint targets. The body tips the board, the [snow model](SNOW_MODEL.md)
turns that edge into a turn, and the reward measures how close the edge came to $\theta^\ast$. PPO improves the policy
from 2048 riders in parallel. The policy never steers or plans a path: the command is the planner, the body is the skill.

## 1. The task: follow the edge command

**No line is laid on the snow.** Each step the env commands a signed edge angle $\theta^\ast$ (heel $< 0$, toe $> 0$).
The board's sidecut turns whatever edge the rider holds ([snow model §2](SNOW_MODEL.md#2-sidecut-steering)), so a
command that alternates heel ↔ toe draws an S. The policy only has to learn the body: get the board onto that edge,
hold it, and change it without falling.

| Quantity | Definition |
|---|---|
| Edge angle $\theta$ | Board roll relative to the snow surface ([snow model](SNOW_MODEL.md)) |
| Path heading $\psi$ | $\mathrm{atan2}(v_y, v_x)$: travel direction, 0 = straight down the fall line. A heel edge turns $\psi$ up, a toe edge down |
| Slip angle $\beta$ | Angle between the board's axis and its travel; ~0 when carving |
| Balanced edge | The edge a rider can hold at speed $v$ on a slope $\alpha$ (formula below) |

The sidecut radius is $R_{sc}$ = 7.6 m ([board model §1](BOARD_MODEL.md#1-geometry)). The balanced edge is

$$\sin\theta = \frac{v^2}{g\cos\alpha \cdot R_{sc}}$$

which is 20° at 5 m/s and 41° at 7 m/s on a 9° slope. A 40°+ edge only balances above ~7 m/s, so edge angle and speed
have to rise together.

**Why a command and not a reward for turning.** Five runs that rewarded turns, speed and carving (with a guide S to
start from, then without) all converged to holding one deep edge into the fence. Tracking an edge command gives the S
directly. A planar double-pendulum sandbox (`scripts/pendulum_rl.py`) confirmed first that edge-command tracking plus
a fall penalty is learnable: it linked an edge change every 2 s at ±45–50°, and counter-angulation emerged on its own.

### The edge command

| Mode | Used in | Command |
|---|---|---|
| **Timed** | Stage 2 | A trapezoid heel → flat → toe, period $T$ |
| **Heading** | Stages 3–4 | Hold one edge at $\pm A$ until the turn has come round ($\psi$ past the switch heading), then ramp through flat to the other edge over 0.6 s. Never sooner than 1 s, never later than 4 s on one edge |
| **Centring** | Stages 3–4 | The switch heading moves 2° per metre off the piste centre (capped): the turn back toward the centre runs longer, the one away from it ends sooner, so the S stays between the fences |

The timed command, with amplitude $A$ and random phase $\varphi$:

$$\theta^\ast(t) = A \cdot \mathrm{clip}\left(1.8\,\sin\left(\frac{2\pi t}{T} + \varphi\right),\ -1,\ 1\right)$$

**Curriculum.** The amplitude $A$ and period $T$ step from easy to hard by a level in $[0, 1]$. It rises while the
running edge error is below 12° and falls above 24°. The policy sees the command now, +0.3 s and +0.6 s.

## 2. Stages

Each stage learns a different thing, but the difference lies in the **task setup** (start, edge command, slope), not
the reward: all track the edge command with the same terms. Stage 2 learns the switch move, stage 3 uses it to ride
the course, stage 4 tightens the turns.

| | **Stage 2: change edges** | **Stage 3: full S** | **Stage 4: tighter turns** |
|---|---|---|---|
| Learns | Get from one edge to the other without falling | Link turns into an S that stays on the piste | End each turn sooner, without the overshoot |
| Start | Already in a carve, balanced on an edge (80% toe), 4–5.5 m/s; 50% from recorded toe-carve states | Riding flat, heading within ±20° of the fall line, 4–5.5 m/s; 30% from recorded states | As stage 3 |
| Edge command | **Timed**: heel ↔ toe, period $T$ 6 → 4 s | **Heading**: switch at 50° across the hill, centring shift up to 25° | Heading at **40°**, centring shift up to 10° |
| Edge angle | ±15 → 30° | ±25 → 45° | As stage 3 |
| Slope, episode | 7°, 10 s | 9°, 20 s | As stage 3 |
| Speed | Starts 4–5.5 m/s | Free (eval mean 7.9 m/s): the steeper slope reaches the speed a 40°+ edge needs | As stage 3 |
| Reward | Hand grab weight 5, Gaussian | Hand grab weight 10, linear (§4) | As stage 3 |
| Saved run | `saved/01_stage2_switches` | `saved/03_stage3_centred` | `saved/04_stage3_tighter` |

**No hold-only stage.** An earlier stage 1 trained only to hold a carve; stage 2 warm-started from it never learned
to change edges (it kept holding the edge stage 1 taught). Stage 2 from scratch learns the hold anyway, because its
episodes start in a carve. The demo runs stage 2 from scratch, continues it, then warm-starts stage 3 from it, and
stage 4 from stage 3.

**Recorded-state starts** (`rsi_frac`). During training the env records rider states from more than 3 s into an
episode on the toe edge (the low, bowed carve where the switch to the heel is hardest) and restarts a share of
episodes from them, with the command at the toe peak. Without them every episode started tall, the only pose the
policy had learned to switch from.

## 3. MDP

| | |
|---|---|
| **Observation** (140) | Humanoid: joint positions and velocities, torso height, orientation, velocities, hands and feet relative to the root (AMP layout) |
| | Board: $\theta$, $v_\parallel$, $v_\perp$, yaw rate, $N/mg$, grip usage ([snow model inputs](SNOW_MODEL.md#inputs)) |
| | Last action, and the two synergy actions |
| | Task (19): speed, $\psi$, $\beta$, board yaw, position between the fences, turn state, the edge command now / +0.3 s / +0.6 s (4 slots are unused, kept so saved policies load) |
| **Action** (30) | 2 synergy actions (edge, lean) + 28 joint residuals, 30 Hz, smoothed: see [humanoid model §5](HUMANOID_MODEL.md#5-actions--joint-targets) |
| **Fall** | Torso tilted > 95° (`lying_deg`; a toe carve bows the torso 55–70°), pelvis < 0.12 m above the board, pelvis < 0.15 m above the snow, board past 85° of edge, or riding tail first (slip > 100° above 2 m/s) |
| **Fence** | Board past ±20 m across the slope; ends the episode like a fall |
| **Time-out** | End of the episode (stage length above) |

The rider rides the rigid board ([board model §1](BOARD_MODEL.md#1-geometry)); the two-half board
([§4](BOARD_MODEL.md#4-two-half-board-opt-in-not-used-in-the-demo)) is never trained on.

## 4. Reward

Each step, with $\Delta t = 1/30$ s and the edge error $e = \theta - \theta^\ast$:

$$r = \Delta t \sum_i w_i r_i + w_{\mathrm{fall}}\ \mathbb{1}[\text{fall or fence}]$$

Weights $w_i$ are those of stage 3.

| Term | $r_i$ | $w_i$ |
|---|---|---|
| **Edge command** | $e^{-(e/\sigma)^2}$, $\sigma$ = 30°, × 0 while the board is flat (edge < 10°, full from 20°): a flat board never pays | 30 |
| **Edge side** | +1 on the commanded side, −1 on the other (command > 10°, edge > 3°): starting the switch pays before the angle matches | 20 |
| **Pose reference** | Legs lean past the edge, centre of mass lean less (angulation), from the pendulum model (formula below); 0 near flat | 10 |
| **Rise** | While the command is within 8° of flat: clip((pelvis height − 0.35 m) / 0.2 m), i.e. come up through the edge change | 30 |
| **Tilt** | clip((torso tilt − 50°) / 20°): a bow past ~65° can't get back to the heel | −10 |
| **Hand grab** | clip(1 − $d$ / 1.2 m), $d$ the inside hand to its grab point (below). Counter-angulation for higher edges | 10 |
| **Fall** | Once, a fall or a fence | −50 |

**Pose reference.** With $\phi_{\mathrm{legs}}$ and $\phi_{\mathrm{CoM}}$ the lean of the pelvis and of the rider's
centre of mass from the board, in its roll plane:

$$r_{\mathrm{pose}} = \exp\left(-\left(\frac{\phi_{\mathrm{legs}} - 1.15\,\theta^\ast}{15°}\right)^2\right) \exp\left(-\left(\frac{\phi_{\mathrm{CoM}} - 0.77 \cdot 1.15\,\theta^\ast}{15°}\right)^2\right)$$

**Hand grab.** Heel turn: right hand → toe edge at the front foot. Toe turn: left hand → heel edge at the back foot
(front and back foot as in the [bindings](BOARD_MODEL.md#3-bindings)). Stage 2 used a Gaussian,
$e^{-(d/0.6)^2}$ ($d$ in metres), at weight 5.

Nothing rewards speed, staying alive, posture or style: survival pays only through the edge terms it allows.

**Lessons (each cost at least one run):**

| Try | What happened | Fix |
|---|---|---|
| Turn, speed and carve rewards, guide S | Every run converged to one deep edge held into the fence | Edge-command tracking |
| Edge command, alive reward on | Rode straight and flat: being alive paid more than turning | No alive reward; a flat board never pays |
| Edge command with a side-slip allowance | Slid flat sideways to keep speed and hit the fence | Edge reward requires an edge |
| $\sigma$ = 12° only | Held the old edge: crossing flat paid nothing, so starting a switch only cost | $\sigma$ = 30° and the edge-side term |
| Fall −10 | Switched edges but fell after 2–3 s: a fall cost less than 2 s of edge reward | −50 |
| Command ±50° at 4–5.5 m/s | 75% of falls were "lying": a 50° edge can't balance below ~7 m/s | Stage 2 at ±15 → 30°; speed and amplitude rise together in stage 3 |
| Lying = torso > 80° | Legitimate deep toe carves were counted as falls | 95° |
| Toe → heel never learned | The board could not leave a low toe carve under any forced action: the torso twist sat at its limit | Rider rebuild ([humanoid model §6](HUMANOID_MODEL.md#6-rider-options-carveenvcfg-flags---set)) |
| Boots added | Edge and lean actions flipped every step, the rider shook | Action smoothing |
| Grab as a Gaussian, $\sigma$ = 0.25 / 0.6 m | The toe-turn hand sat 0.9 m away with no pull | Linear in distance to 1.2 m |
| Edge change on a timer | Turns stopped short of 50° or drifted toward a fence | Heading switch with centring |

## 5. PPO

| Setting | Value |
|---|---|
| Envs | 2048 in parallel |
| Rollout | 48 steps (1.6 s, an edge change fits) × 2048 = **98k samples per update** |
| Update | 5 epochs × 8 mini-batches |
| Training | 16 000 steps per env per run (~26 min on one GPU) |
| Networks | Policy and value: separate MLPs, 512-256-128, ELU |
| Policy | Gaussian, initial std $e^{-1}$, capped at $e^{-0.3}$ (it had grown to ~6 against clipped actions) |
| Discount, GAE | $\gamma$ = 0.995 (~6.7 s, more than one S), $\lambda$ = 0.98 |
| Learning rate | $10^{-3}$, KL-adaptive (target KL 0.01) |
| Clipping | Ratio 0.2, value 0.2, grad norm 1.0 |
| Entropy bonus | 0.003 |
| Warm start | Each stage starts from the previous stage's weights; the exploration noise is reset |

## 6. Evaluation (`playground/snowboard/play.py`)

A deterministic policy (mean action) rides fixed runs: piste centre, straight down, 100 m long, no recorded-state
starts, and the edge command held at the level training ended on (`edge_cmd_level_init=1`, `edge_cmd_level_fixed`).
Each run prints one line of stats; the longest is saved as `trail.png` + `ride.csv` (edge, command, slip, torso tilt,
pelvis height per step). 

**Diagnostics:**

| Script | Shows |
|---|---|
| `scripts/switch_test.py` | From a low toe carve, the command flips to the heel: does the policy (or a forced edge / lean action) reach the heel edge? |
| `scripts/jitter_probe.py` | Which joints shake: velocity and step-to-step velocity change per joint, action change per step |
| `scripts/grab_probe.py` | How close each inside hand gets to its grab point, per turn side |

## 7. Results

### Stage 3 (`saved/03_stage3_centred`)

| | Training (end) | Eval (90 s) |
|---|---|---|
| Runs reaching the end of the course | episodes 19.6 of 20 s | **5 of 5** |
| Falls / fence | 0% / 2% | 0 / 0 |
| Edge error | 5° at ±45° | |
| Edge (toe / heel) | | mean 42° / 38°, max 52° / 48° |
| Slip | | 2.5°: a clean carve, a thin line |
| Turns | 1.7 per 10 s | 3–4 per run, an edge change every ~3.8 s |
| Track | | −13 to +8 m from the centre (fences at ±20 m) |

![Track of the longest eval run: a thin centred S down 100 m](../saved/03_stage3_centred/trail.png)

The turns overshoot: the heading reaches 56–91° across the hill at the switch, aimed at 50–60°. The board also skids
briefly through each edge change.

### Stage 4: tighter turns (`saved/04_stage3_tighter`)

The overshoot is the heading switch firing late: the edge command only flips once $\psi$ passes the switch heading,
and the board keeps swinging while the command ramps through flat. Switching earlier ends the turn nearer the target.
Stage 4 does this, warm-started from stage 3:

| Setting | Stage 3 | Stage 4 | Effect |
|---|---|---|---|
| `edge_cmd_heading_deg` | 50 | 40 | Switch the edge at 40° across the hill: no overshoot, no flat cross-hill segment |
| `edge_cmd_center_max_deg` | 25 | 10 | Smaller centring shift, so the switch heading stays near 40° |

| | Stage 3 | Stage 4 |
|---|---|---|
| Turns per 100 m | 3–4 | **4–5** |
| Mean speed | 7.9 m/s | 7.9 m/s |
| Edge (toe / heel) | 42° / 38° | 43° / 38° |
| Slip | 2.5° | ~3° |

GIFs: `docs/demo/04_stage3_tighter_front.gif`, `_top.gif`.

## Not solved

| Gap | Effect |
|---|---|
| Turn sharpness | Going below a 40° switch heading should add turns until the sidecut sets the limit ($R_{sc}\cos\theta$, see the [snow model §5](SNOW_MODEL.md#5-in-the-trained-carve)) |
| Sharper turns at the same edge angle | Need the two-half board's progressive sidecut and flex ([board model §4](BOARD_MODEL.md#4-two-half-board-opt-in-not-used-in-the-demo)), not trained on |
| Skid through edge changes | Small smudges remain at each change |
