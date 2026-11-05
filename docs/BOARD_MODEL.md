# Board model

The board is one rigid link authored onto the humanoid at spawn (`envs/rider.py::_author_board`), with dimensions
from a real 152 cm all-mountain board. PhysX sees a thin box that touches frictionless snow; everything the board's
*shape* does on snow (sidecut, grip) lives in the [snow model](SNOW_MODEL.md). The round tips, stripes and chevrons are
visual only (no collision, no mass). The board is not scaled with the rider (the humanoid is scaled to 1.65 m).
Knobs: `config.yaml` → `board`, `rider`; SI units.

![Snowboard top view to scale: sidecut, effective edge, bindings at +54 and +45 degrees](rider/board_top.svg)

## 1. Geometry

| Parameter | Value | Used for |
|---|---|---|
| Length | 1.52 m | Collision box |
| Waist width $w$ | 246 mm | Collision box, trail width |
| Tip width | 289 mm | Sidecut depth |
| Effective edge $L$ | 1.14 m | Sidecut radius, yaw-torque cap |
| Thickness | 8 mm | Collision box |
| Mass | 2.7 kg | Rigid body |

**Sidecut radius.** The edge between the two contact points is an arc of depth $d$:

$$d = \frac{w_{\text{tip}} - w_{\text{waist}}}{2} = 21.5\ \text{mm} \qquad R_{sc} \approx \frac{L^2}{8d} = \frac{1.14^2}{8 \times 0.0215} = 7.6\ \text{m}$$

The real board's progressive sidecut (10.7 / 6.8 / 10.7 m nose / waist / tail) is replaced by this one radius.

## 2. Frames and edges

| Axis | Points to |
|---|---|
| $x$ | Nose |
| $y$ | Heel edge |
| $z$ | Up, out of the base |

Edge angle $\theta > 0$: rolled about $+x$ so the **toe edge** ($-y$) digs in. Toe edge turns clockwise (riding nose
first), heel edge counter-clockwise.

## 3. Bindings

| Binding | Foot | Position $x$ | Angle $\alpha$ | Toe direction |
|---|---|---|---|---|
| Front | Left | $s + b/2 = +0.265$ m | +54° | yaw $\alpha - 90° = -36°$ |
| Back | Right | $s - b/2 = -0.295$ m | +45° | yaw $-45°$ |

**Regular, posi-posi stance:** left foot forward, and both bindings angled toward the nose (+54° / +45°), the forward
stance used for carving. Not duck (front +, back −). Stance width $b = 0.56$ m, setback $s = -15$ mm (toward the
tail). The picture shows the nose on the left.

`config.yaml` holds the same values (`binding_angles_deg=[54, 45]`, `stance_width=0.56`); the stance file is
`stance_54_45_squat4.yaml` from `scripts/make_stance.py`.

Both feet are welded to the board: the front binding is a tree joint, the back one a loop-closing constraint (see
[humanoid model §3](HUMANOID_MODEL.md#3-board-binding)).

## 4. Two-half board: torsion, camber, flex, progressive sidecut

**Opt-in, off in the demo run** (the demo rides the rigid board of §1). `--set board_flex=true`: the board becomes **two rigid halves** (nose, tail), each a box of half the length, joined at
the centre by a spring joint with two DoFs. The policy can't drive it: it bends and twists only under the rider's
load.

| DoF | Spring | Rest | Effect |
|---|---|---|---|
| `board_flex:0` roll | Torsion, 5 N·m/° | 0 | Nose and tail can edge differently |
| `board_flex:1` pitch | Bend, set so $N_c$ presses it flat | $-\varphi_0$ (camber) | Unloaded, only the tips touch |

$$\varphi_0 = \frac{4h}{L} \quad (h = 5\text{ mm camber}) \qquad k_{\text{bend}} = \frac{N_c\,L}{4\,\varphi_0} \quad (N_c = 330\text{ N} \approx \tfrac{1}{2}\,mg)$$

The two angled boxes are a two-segment **camber profile**: the collision shape itself bends, instead of one flat
8 mm box.

Each half glides and grips on **its own edge angle $\theta_i$ and load $N_i$** (from its own contact sensor). Steering
is one arc for the whole board:

$$\kappa = \underbrace{\frac{c}{R(s)\cos\bar\theta}}_{\text{sidecut}} + \underbrace{(N - N_c)_+\,\frac{L}{6\,EI}\,\sin\lvert\bar\theta\rvert}_{\text{flex}}$$

| Symbol | Meaning |
|---|---|
| $\bar\theta = \sum_i w_i\theta_i$ | Load-weighted edge angle, $w_i = N_i / N$ (**torsion**) |
| $s = w_{\text{nose}} - w_{\text{tail}}$ | Centre of pressure, −1 (all tail) … +1 (all nose) |
| $R(s) = R_{sc} + (R_{\text{tip}} - R_{sc})\,s^2$ | **Progressive sidecut**: 7.6 m centred, 10.7 m at a tip |
| $c = 1 + q_{\text{bend}}/\varphi_0 \in [0, 1]$ | **Camber** pressed flat: 0 at rest (tips only, no sidecut), 1 flat (from the joint angle) |
| $EI = 1500$ N·m² | **Flex**: load beyond $N_c$ bends the board into a tighter arc |

### Validation (`scripts/validate_board.py`)

![Two-half board validation: flex, progressive sidecut and torsion, sim vs formula](snow/board_validation.png)

| Check | Sim / formula |
|---|---|
| **Flex**: radius vs load (θ = 40°) | 5.85 / 5.82 m at 0.4 mg, 3.99 / 3.95 m at 2 mg |
| **Progressive sidecut**: radius vs centre of pressure (θ = 20°) | 7.15 / 7.14 m centred, 10.06 / 10.05 m all on one half |
| **Torsion**: radius vs tail edge (nose 35°) | 6.24 / 6.23 m untwisted, 7.03 / 7.02 m with the tail at 10° |

In the rider, standing still: body weight (664 N) presses the camber flat ($c$ = 0.99), with 56–59% of the load on the
nose (the front knee is bent more). Measured with an earlier stance, not re-checked on the current one.

## Still not modelled

| Gap | Effect |
|---|---|
| Continuous flex | Two segments, one hinge: no bending within a half |
| Reverse camber on snow | A rigid half can't sink into the snow: flex beyond flat is analytic, not geometric |
| Snow deformation | Hardpack only: no edge penetration that grows with load |
