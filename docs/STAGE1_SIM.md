# Stage 1 — Sim: humanoid + board on snow

## Goal

The humanoid_amp env, modified with a sloped plane, a board fixed to both feet, and a snow model where
**tipping the board on edge makes it turn**.

## Commands

```bash
python scripts/sim_check.py                                  # humanoid in stance on the board, sliding (no policy)
python envs/snow.py --test                                   # scripted board: edge angle → turn
```

## Changes to the forked env

All dimensions come from `config.yaml` → `rider:` / `board:` (SI units).

| Change | Detail |
|---|---|
| Ground | Plane tilted 15° |
| Rider | AMP humanoid scaled to 1.65 m, 65 kg; regular stance |
| Board | Box 1.52 × 0.246 × 0.008 m, 2.7 kg; collision is a plain box (sidecut lives in the snow model) |
| Bindings | Fixed joints, 0.52 m apart, centred 15 mm toward the tail, feet rotated 42° / 27° |
| Snow | PhysX friction ≈ 0 + `snow.py` forces each step (below) |
| Reset | Start on the slope in a fixed snowboard stance (switches to clip frames once Stage 2 exists) |

## Snow model (`envs/snow.py`, groomed hardpack)

Three terms, applied to the board as an external force/torque each physics step:

```
θ      = edge angle (board roll relative to the slope surface)
N      = normal force on the board
v_long = velocity along the board,   v_lat = velocity across it

1. Glide     F_long = −μ_glide · N · sign(v_long)  −  ½ ρ C_dA |v| v
2. Sidecut   ω_target = v_long / (R_eff · cos θ) · sign(θ)      (0 when |θ| < min_edge_deg)
             τ_yaw    = k_yaw · (ω_target − ω_yaw)
3. Grip      F_lat = −(k·z + c·v_lat), z = ∫v_lat (stick-slip spring), clamped to |F_lat| ≤ μ_edge · N
             → holds the line (carve) until the turn demands more grip, then slips (skid / wash-out)
```

The policy tips the board and keeps balance. The snow model turns the edge angle into an arc. Flat
base: no steering, little grip → skid. On edge: arc of radius `R_eff · cos θ` while grip holds.

`R_eff = 7.6 m` is a single-radius stand-in for the 10.7 / 6.8 / 10.7 m progressive sidecut
(derived from the 289 / 246 mm widths over the 1.14 m effective edge). Modelling the progressive
radius is a post-MVP refinement.

## Done when

- [ ] Scripted board + 65 kg dummy mass, flat base on 15° → accelerates to a plausible top speed.
- [ ] Scripted edge angle θ ∈ {10, 20, 30, 40}° at low speed → turn radius ≈ 7.6 · cos θ m.
- [ ] Same θ at high speed → board washes out (grip limit reached) instead of turning tighter.
- [x] Humanoid in stance, held by PD only, slides straight down a gentle slope for a few seconds
      without the board/feet joints exploding. (~3 s on 3–5°, ~4 s on 15°; it then topples, it doesn't explode:
      braking pitches a forward-stance rider toeward, and catching that is the policy's job.)

## Gotchas

- Fixing both feet to one board creates a closed kinematic loop. If it explodes, attach one foot via
  the articulation and the second with a separate fixed joint, or lower that joint's stiffness.
- The upstream solver iterations (4 position / 0 velocity) can't hold the two-binding loop against snow
  grip: the rider folds at the ankles and tips over even standing still. `rider_cfg` uses 32 / 4
  (16 / 4 still sags ~0.05 rad at the hips/ankles; stiffer gains make it worse, not better).
- A flat board needs yaw resistance too (grip spread along the edge resists pivoting). Without it the
  board spins up on its own and the turn throws the rider over. Capped at μ_lat · N · L / 4.
- Air drag goes on the rider's body, not the board: at speed it balances gravity, and through the feet
  it would pitch the rider forward.
- Grip (term 3) acts on the board alone. Sizing it to cancel the whole rider's sideslip per step
  (≈ m_system / dt) is unstable for the light board end: it shakes the board sideways at the grip limit
  and topples a standing rider within ~1 s. Use a stick-slip spring-damper sized for the board end.
- Use the board's *own* frame for θ and v_lat, not world axes. The slope is tilted.
- 42° / 27° is a forward (carving) stance, not duck. Front-foot toe overhang is small on a 246 mm waist,
  but check that the boot collision boxes don't touch the snow at high edge angles.
- The humanoid_28 USD authors knee/elbow (revolute) drives as `drive:X`; PhysX only reads `drive:angular`
  on revolute joints, so they load with **no motor** and the knees fold within ~1 s.
  `envs/rider.py::_fix_hinge_drives` copies X → angular at spawn.
