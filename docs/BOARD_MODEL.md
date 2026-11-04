# Board model

The board is one rigid link authored onto the humanoid at spawn (`envs/rider.py::_author_board`), with dimensions
from a real 152 cm all-mountain board. PhysX sees a thin box that touches frictionless snow; everything the board's
*shape* does on snow (sidecut, grip) lives in the [snow model](SNOW_MODEL.md). Knobs: `config.yaml` → `board`,
`rider`. SI units.

![Snowboard top view to scale: sidecut, effective edge, bindings at +42 and +27 degrees](rider/board_top.svg)

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
| Front | Left | $s + b/2 = +0.245$ m | +42° | yaw $\alpha - 90° = -48°$ |
| Back | Right | $s - b/2 = -0.275$ m | +27° | yaw $-63°$ |

**Regular, posi-posi stance:** left foot forward, and both bindings angled toward the nose (+42° / +27°), the forward
stance used for carving. Not duck (front +, back −). Stance width $b = 0.52$ m, setback $s = -15$ mm (toward the
tail). The picture shows the nose on the left.

Both feet are welded to the board: the front binding is a tree joint, the back one a loop-closing constraint (see
[humanoid model §3](HUMANOID_MODEL.md#3-board-binding)).

## Not modelled (yet)

| Gap | Effect | Status |
|---|---|---|
| **Camber** | An unloaded board touches only near its tips; light load holds less edge | Planned |
| **Flex** | Pressing the board bends it into a tighter arc than the sidecut | Planned |
| Torsion | Nose and tail can't edge differently | — |
| Progressive sidecut | Fore/aft weight shift doesn't change the arc | — |
| Real camber profile in the collision box | One flat 8 mm box | — |
