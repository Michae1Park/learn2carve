#!/usr/bin/env bash
# Stage 1, carve controller: every episode starts already in a balanced carve (board on edge θ, sin θ = v² / (g R),
# heel or toe, 5-7.5 m/s -> 20-49°) and the command holds that side at the edge that balances the current speed; no edge changes. Learn to hold a carve:
# edge tracking + pendulum pose (legs in, COM less: angulation) + not falling. Synergy actions, 7° slope, wide ankles.
# A held carve circles back up the hill, so episodes are short (4 s). Extra args go to train.py.
cd "$(dirname "$0")/.." && exec scripts/train_edge_cmd.sh slope_deg=7 "ankle_range_deg={x: 45, y: 80, z: 55}" \
  stance_tag=squat4 rider_to_height=true stance_width=0.56 knee_rotation=true "passive_joints=[knee:2]" arm_stiffness_scale=0.3 arm_action_scale=0.5 torso_twist_deg=90 \
  synergy=true carve_start=true edge_cmd_hold=true "start_speed_turns=[5,7.5]" \
  rew_pose_ref=10 rew_fall=-10 episode_length_s=4 "$@"
