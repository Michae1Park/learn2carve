# Stage 2 — Clip: YouTube → reference motion

## Goal

One 5–10 s carving clip, converted into a motion file that Isaac Lab's AMP motion loader can read for our
humanoid.

## Commands

```bash
python motion/extract.py  data/videos/carve.mp4 --out data/motions/carve_smpl.npz   # video → SMPL
python motion/retarget.py data/motions/carve_smpl.npz --out data/motions/carve.npz  # SMPL → humanoid
python scripts/play_motion.py data/motions/carve.npz                                # check by eye
```

## How it works

```
video ──► world-grounded 3D human (GVHMR) ──► SMPL pose per frame (gravity-aligned world frame)
                                                   │
                                                   ▼
                     retarget: copy joint rotations onto the matching humanoid joints,
                     scale root height, smooth (low-pass), compute velocities by finite difference
                                                   │
                                                   ▼
                     carve.npz  (same fields as the humanoid_amp motion files:
                                 dof positions/velocities, body positions/rotations/velocities, fps)
```

## Picking the clip

This choice matters more than anything else in the stage:

- Follow-cam or side view, one rider, whole body visible, no cuts.
- Clean linked carves (toe → heel → toe) on a groomed slope.
- Minimal snow spray and baggy-clothing occlusion.

## Done when

- [ ] `play_motion.py --sim` plays the clip on the Stage 1 slope with the feet on the board.
- [ ] `play_motion.py` shows the humanoid doing recognizable toe/heel carves.
- [ ] Feet stay a constant distance apart (they're on a board). If they don't, the pose estimate is bad.

## Gotchas

- **Fallback if extraction is too noisy:** keyframe the same carve in Blender on an SMPL body, or use a
  mocap clip. The rest of the pipeline doesn't change. Don't spend more than a day fighting the video.
- The board isn't in the motion. The Stage 1 sim adds it by fixing it to the feet.
- Regular vs. goofy: mirror the clip if the rider's stance doesn't match the sim humanoid.
