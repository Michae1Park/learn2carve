#!/usr/bin/env bash
# Edge command (from scratch): the env commands a signed edge angle that rolls heel -> base -> toe on a timer (a turn
# every ~2 s; curriculum ±20° up to ±50°); the policy is rewarded for tracking it while carving cleanly.
# Only what the carve needs is rewarded: edge changes and high edge (edge_cmd), frequent turns (turn), a thin S line
# (carve: clean, low slip), plus the fall penalty and the nose-first rule (no alive reward: surviving a straight run paid more than turning). Speed, posture and everything else is left to
# learning (physics: a high edge only balances at speed, sin θ = v² / (g · 7.6 m)). Extra args go to train.py.
# Weights: edge tracking dominates (30/s), then not falling (-50), carve 1/s and turn 2 far behind.
cd "$(dirname "$0")/.." && exec scripts/train_best.sh stance_tag=squat3 edge_cmd=true rew_guide=0 guide_steps=0 \
  rew_alive=0 rew_speed=0 rew_speed_cap=0 rew_straight=0 rew_uphill=0 rew_edge_rate=0 \
  rew_low=0 rew_bow=0 rew_facing=0 rew_hand=0 rew_hip=0 rew_height=0 rew_hip_tail=0 rew_arm=0 rew_balance=0 \
  rew_edge_switch=0 rew_edge_hold=0 rew_wrong_edge=0 rew_release=0 rew_traverse=0 rew_swing=0 rew_deep_edge=0 \
  rew_edge_cmd=30 rew_fall=-50 rew_carve=1 rew_turn=2 episode_length_s=20 "$@"
