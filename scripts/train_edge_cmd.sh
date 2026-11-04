#!/usr/bin/env bash
# Edge command (from scratch): the env commands a signed edge angle that rolls heel -> base -> toe on a timer; the
# policy is rewarded for tracking it (curriculum: ±20° / 4.5 s up to ±50° / 2.6 s). The S-gate rewards are off.
# Extra args go to train.py.
cd "$(dirname "$0")/.." && exec scripts/train_best.sh speed_cap=8.0 "speed_cmd=[5,7]" stance_tag=squat3 \
  edge_cmd=true rew_guide=0 guide_steps=0 \
  rew_low=0 rew_bow=0 rew_facing=0 rew_hand=0 rew_hip=0 rew_height=0 rew_hip_tail=0 rew_arm=0 rew_balance=0 \
  rew_edge_switch=0 rew_edge_hold=0 rew_wrong_edge=0 rew_release=0 rew_traverse=0 rew_swing=0 rew_deep_edge=0 \
  rew_edge_rate=-1 rew_straight=-1 rew_uphill=-2 rew_carve=2 episode_length_s=20 "$@"
