# Stage 3 — Train: imitate, then RL

## Goal

One policy that rides down the slope **looking like the clip**.

## Commands

```bash
python scripts/train.py --phase imitate                                  # A: style reward only
python scripts/train.py --phase rl --init logs/imitate/best.pt          # B: style + task reward
python scripts/play.py  logs/rl/best.pt --video
```

## How it works

```
policy π(a | s)  ── joint position targets ──► PD ──► physics
       ▲
       │ PPO
reward = w_style · r_style  +  w_task · r_task
         │                     │
         AMP discriminator:    + uprightness
         "does this motion     + downhill speed (capped)
          look like the clip?" − fall (terminate)
```

| Phase | `w_style` | `w_task` | Result |
|---|---|---|---|
| A imitate | 1.0 | 0 (except fall termination) | Initial snowboarder: moves like the demo, may be fragile |
| B RL | 0.5 | 0.5 | Final policy: keeps the style, stays up |

Phase A is the "imitation learning" box in the diagram, and phase B is the "RL" box. Both are the same
training script with different weights, so there's no separate behavior-cloning code (see D-002).

**Reference state initialization:** start episodes at random frames of the clip. This is cheap and the
biggest single help for imitation.

**Pelvis assist (D-006):** an upward/stabilizing force on the pelvis that fades from
`train.assist.max_force` to 0 over `anneal_iters`. Final evaluation runs with the assist off.

## Done when

- [ ] With the assist off, the policy rides 20 s down the slope without falling, > 90 % of episodes.
- [ ] Visible alternating toe/heel edges.

## Gotchas

- A 5–10 s clip is short. Loop it (carving is periodic) and mirror it for more data.
- If the discriminator wins too easily (style reward → 0), lower its learning rate or add a gradient
  penalty. The skrl AMP config already has knobs for both.
