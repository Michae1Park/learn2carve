#!/usr/bin/env bash
# Stage A (from scratch): learn a thin S at high edge with edge changes over the base. Posture terms off, gates and
# penalties loosened. Extra args go to train.py (later --set values override these).
cd "$(dirname "$0")/.." && exec scripts/train_best.sh speed_cap=8.0 "speed_cmd=[5,7]" stance_tag=squat3 guide_fade_below_deg=0 \
  rew_low=0 rew_bow=0 rew_facing=0 rew_hand=0 rew_hip=0 rew_height=0 rew_hip_tail=0 rew_arm=0 rew_balance=0 \
  rew_edge_switch=30 switch_flat_deg=10 rew_wrong_edge=-1 rew_edge_hold=-2 edge_hold_after_s=5 \
  rew_traverse=-1 traverse_after_s=5 rew_deep_edge=0 rew_edge_rate=-1 rew_straight=-1 rew_uphill=-2 \
  sigma_heading_deg=45 "$@"
