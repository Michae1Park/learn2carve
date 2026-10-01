# learn2carve

Teach a simulated snowboarder to **carve** in Isaac Sim from a YouTube demonstration (LfD), then use RL
to make it stay up on the slope.

Status: **MVP design**. No code yet.

## Pipeline

```
YouTube carving clip
        │  Stage 2: video → 3D human motion → retarget to sim humanoid
        ▼
Reference motion (.npz)
        │
        ▼
┌──────────────── Isaac Sim slope + board (Stage 1) ────────────────┐
│                                                                   │
│  Stage 3, phase A  imitation   style reward only       → initial snowboarder
│  Stage 3, phase B  RL          style + task reward     → final policy
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
Stage 4: video of the avatar carving + "with vs. without demo" comparison
```

The simulation is built first (Stage 1) so it can be tested before a clip exists.

## Stages

| Stage | Question | Output | Doc |
|---|---|---|---|
| **1 Sim** | "Can the humanoid ride a board down a slope?" | Isaac Lab env: slope + board + snow force | [STAGE1_SIM](docs/STAGE1_SIM.md) |
| **2 Clip** | "What does a human carve look like on our humanoid?" | `data/motions/carve.npz`, playable in the sim | [STAGE2_CLIP](docs/STAGE2_CLIP.md) |
| **3 Train** | "Imitate, then RL" | Policy checkpoint | [STAGE3_TRAIN](docs/STAGE3_TRAIN.md) |
| **4 Show** | "Did the demo help?" | Video + one plot | [STAGE4_SHOW](docs/STAGE4_SHOW.md) |

Design rationale: [docs/DECISIONS.md](docs/DECISIONS.md).

## Layout

```
learn2carve/
├── config.yaml
├── docs/
├── data/
│   ├── videos/             # source clip (not committed)
│   └── motions/            # carve.npz
├── motion/
│   ├── extract.py          # video → SMPL motion   (Stage 2)
│   └── retarget.py         # SMPL → humanoid .npz  (Stage 2)
├── envs/
│   ├── carve_env.py        # forked from Isaac Lab humanoid_amp (Stages 1, 3)
│   ├── snow.py             # board–snow force       (Stage 1)
│   └── assets/board.usd
└── scripts/
    ├── sim_check.py        # stance rider sliding on the slope (Stage 1)
    ├── play_motion.py      # play the reference clip kinematically (Stage 2)
    ├── train.py            # skrl AMP
    └── play.py             # roll out policy + record video
```

## Stack

Isaac Sim + Isaac Lab, starting from Isaac Lab's **humanoid AMP** example (direct workflow, `skrl`). It
already contains a humanoid, a motion loader, an AMP discriminator ("style reward") and PPO, so most of
the RL side is a fork rather than new code. Video → 3D pose uses an off-the-shelf world-grounded human
mesh recovery model (GVHMR or similar).

## Out of scope for the MVP

Style blending, terrain curriculum, moguls/jumps, procedural mountains, domain randomization, multiple
styles. Each of these can be added later as another stage on the same pipeline.
