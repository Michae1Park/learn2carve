# Stage 4 — Show: did the demo help?

## Goal

A result someone can watch, plus one comparison that shows LfD + RL beats RL alone.

## Commands

```bash
python scripts/train.py --phase rl --no-style       # baseline: task reward only, from scratch
python scripts/play.py logs/<run>/best.pt --video   # for each run
```

## Comparison

| Run | Expectation |
|---|---|
| RL only (task reward) | Stays up, but moves in a weird, un-human way (classic RL gait) |
| Imitate only (phase A) | Looks like the clip, falls more |
| Imitate → RL (phase B) | Looks like the clip **and** stays up |

Report: falls per episode, mean downhill speed, and the side-by-side video. The video is the real
deliverable.

## Done when

- [ ] 3-panel side-by-side video.
- [ ] Small table with the three runs.
