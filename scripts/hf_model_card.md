---
library_name: skrl
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - isaac-lab
  - isaac-sim
  - humanoid
  - unitree-g1
  - snowboarding
  - ppo
---

# learn2carve: saved runs

Policy checkpoints for [learn2carve](https://github.com/Michae1Park/learn2carve), PPO policies (skrl, Isaac Lab) that
teach a stock Unitree G1 humanoid to carve linked snowboard turns down a 100 m course in Isaac Sim.

The code, the environment and the training notes live in the GitHub repo, which keeps only the final policy's weights.
This repo holds every saved run, at the same paths as in the GitHub repo, so a folder from here drops straight into a
clone.

## Final policy

`saved/g1/chain/07_stage7_g1_7065_finish60`: stock Unitree G1 on 70°/65° bindings, edges switched at 60° across the
hill.

| Eval (120 s at the full command) | |
|---|---|
| Runs that reach the end of the 100 m course | 6 / 6, no falls |
| Run time | 17.3–18.5 s |
| Linked turns per run | 6 |
| Edge, mean | 57–59° |
| Slip | 1.5° |

## Contents

| Folder | What |
|---|---|
| `saved/g1/chain/01`–`07` | The G1 training chain: the last run of each stage, each warm-started from the one before |
| `saved/01`–`04` | The earlier AMP rider, stages 2–4 |

Each run folder holds `checkpoints/best_agent.pt`, `run.yaml` (the full env config it trained with), `physical.yaml`,
`ppo.yaml`, the stance file and a README with its training metrics.

## Use

```
git clone https://github.com/Michae1Park/learn2carve && cd learn2carve
scripts/install.sh
hf download <this repo> --include "saved/g1/chain/05_stage5_g1_7065_edge65/*" --local-dir .
scripts/py playground/snowboard/play.py saved/g1/chain/05_stage5_g1_7065_edge65
```

See `docs/G1_TRAINING_NOTES.md` in the GitHub repo for the stages, rewards and what each run learned.
