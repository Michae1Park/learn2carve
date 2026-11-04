# Snow model

> **TL;DR** PhysX can't carve: its friction is the same in every direction. `envs/snow.py` replaces it with
> three forces taken from how a real board works: **glide**, **sidecut steering** and **edge grip**. Driven alone,
> the model hits textbook numbers: **turn radius within 3%**, **terminal speed within 0.2 m/s**.

![Snow model validation: turn radius vs edge angle, and terminal speed vs slope, sim vs formula](snow/validation.png)

## The idea in one picture

```
           edge angle θ  (the rider tips the board; the policy's only lever)
                 │
     ┌───────────┼─────────────────────────────┐
     ▼           ▼                             ▼
  1 GLIDE     2 SIDECUT                     3 GRIP
  along the   board yaws onto an arc        sideways spring at the biting edge
  board       of radius R·cos θ             holds the arc ... up to μ(θ)·N
     │           │                             │
     └──── carve: arc held, zero side slip ────┘
           skid:  arc needs more grip than μ(θ)·N → edge slides
```

The policy never steers the board. It only moves its body. Tipping the board puts it on edge, and the snow
does the rest, as with a real board.

## The three forces

| # | Force | Formula | What it does |
|---|---|---|---|
| 1 | **Glide** | `F = −μ_glide · N` along the board, `−½ρ C_dA \|v\| v` air drag on the body | Sets top speed |
| 2 | **Sidecut** | yaw rate → `v / (R_sc · cos θ)`, torque capped at `μ · N · L/4` | Tipped board turns on its arc |
| 3 | **Grip** | sideways spring-damper at the edge, capped at `μ(θ) · N` | Holds the carve. Past the cap: skid |

Grip `μ(θ)` blends from **0.15 flat** (skids easily) to **1.5 on edge** (bites) over the first 15°.

## Validation (`scripts/validate_snow.py`)

A planar rigid board (rider + board mass) is driven by the snow model alone, with no humanoid and no Isaac Sim. It's
then compared against closed-form physics.

| Check | Formula | Sim vs formula |
|---|---|---|
| **Slow turn follows the sidecut** | `R = R_sc · cos θ` | 7.5 / 7.5 m at 10°, 5.4 / 5.4 m at 45° (6 m/s) |
| **Fast turn opens up at the grip limit** | `R = v² / (μ g)` | 13.7 / 13.3 m at 14 m/s, any edge angle ≥ 15° |
| **Terminal speed** | `v = √(2mg(sin α − μ cos α) / ρC_dA)` | 18.2 / 18.2 m/s on 10°, 23.6 / 23.6 m/s on 15° |

**Reading the left plot:**
- **Blue (6 m/s):** slow enough that the edge never slips. Radius shrinks with edge angle, exactly as the sidecut says.
- **Red (14 m/s):** too fast for any arc the board can hold. Radius is set by grip, not by edge angle.
- **Green (10 m/s):** follows the sidecut up to ~25°, then hits the grip limit. This is the **carve → skid transition**.

Unit tests (`tests/test_snow.py`) pin the same behaviour, plus signs and edge cases (flat board doesn't steer,
airborne board has no grip).

## Parameters (`config.yaml` → `snow`, `board`)

| Parameter | Value | Source |
|---|---|---|
| Sidecut radius `R_sc` | 7.6 m | Real board: 289 / 246 / 289 mm widths over a 1.14 m edge |
| Glide `μ_glide` | 0.05 | Waxed base on hardpack: 0.03–0.08 |
| Grip `μ_flat` → `μ_edge` | 0.15 → 1.5 | Tuned: flat skids, edged holds ~1.5 g |
| Air drag `C_dA` | 0.5 m² | Upright rider |
| Mass | 67.7 kg | 65 kg rider + 2.7 kg board |

## What it doesn't model

| Left out | Effect |
|---|---|
| Progressive sidecut (10.7 / 6.8 / 10.7 m) | One effective radius. Fore/aft weight shift doesn't change the arc |
| Board flex, camber | No "pressing" the board into a tighter arc |
| Snow conditions (powder, ice, ruts) | One groomed hardpack |
| Real terrain | Physics is one tilted plane. The mountain is visual only |

## Implementation notes

- **Grip is a spring, not a hard constraint.** Cancelling the slip in one step (`m/dt`) shakes the light board and
  topples the rider. The spring-damper is sized for the ~7 kg board end at 120 Hz.
- **A flat board resists pivoting.** Without that, it spins up by itself and throws the rider.
- **Drag acts on the body, not the board.** Through the feet it would pitch the rider forward.
- **Every quantity is in the slope frame.** The model only needs the snow normal, so tilted terrain and tilted
  gravity give identical results (D-008).
