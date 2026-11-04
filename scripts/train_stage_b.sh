#!/usr/bin/env bash
# Stage B (warm start): light style on top of Stage A; open upper body only loosely. Usage:
#   scripts/train_stage_b.sh --init_from <stage A run> [more key=value overrides]
cd "$(dirname "$0")/.." && exec scripts/train_stage_a.sh rew_facing=0.3 rew_low=0.5 rew_hand=0.3 rew_hip=0.3 "$@"
