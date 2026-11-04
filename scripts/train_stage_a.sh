#!/usr/bin/env bash
# Stage A (from scratch): learn to flip edges only. Posture/style terms off, edge change boosted. Extra args go to train.py.
cd "$(dirname "$0")/.." && exec scripts/train_best.sh speed_cap=8.0 "speed_cmd=[5,7]" stance_tag=squat3 guide_fade_below_deg=0 \
  rew_low=0 rew_bow=0 rew_facing=0 rew_hand=0 rew_hip=0 rew_height=0 rew_hip_tail=0 rew_arm=0 rew_balance=0 \
  rew_edge_switch=50 "$@"
