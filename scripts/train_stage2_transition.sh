#!/usr/bin/env bash
# Stage 2, edge changes (from scratch): start in a balanced carve (80% toe) with the timed edge command at the peak of
# that side, then heel <-> flat <-> toe on a timer (±15 -> 30°, period 6 -> 4 s), 10 s episodes on a 7° slope; half
# the episodes restart from recorded toe-carve states. Reward: playground/snowboard/rewards/stage2.yaml.
# Extra args go to train.py, e.g. --init_from <run> --timesteps 12000 --set rsi_frac=0.3
cd "$(dirname "$0")/.." && exec scripts/py playground/snowboard/train.py --livestream 2 \
  --rewards playground/snowboard/rewards/stage2.yaml --set \
  slope_deg=7 "binding_angles_deg=[54,45]" pelvis_mass_frac=0.2 \
  stance_tag=squat4 rider_to_height=true stance_width=0.56 knee_rotation=true "passive_joints=[knee:2]" \
  "ankle_range_deg={x: 45, y: 80, z: 55}" arm_stiffness_scale=0.3 arm_action_scale=0.5 torso_twist_deg=90 \
  stiff_boots=true "boot_range_deg={y: [-5, 15], x: [-5, 5], z: [-5, 5]}" \
  synergy=true "synergy_toe_rad={left_knee: 0.4, right_knee: 0.4, left_hip_y: -0.3, right_hip_y: -0.3}" \
  synergy_smooth=0.35 joint_smooth=0.35 \
  episode_length_s=10 carve_start=true carve_start_toe_frac=0.8 "start_speed_turns=[4,5.5]" lying_deg=95 \
  "edge_cmd_amp_deg=[15,30]" "edge_cmd_period_s=[6.0,4.0]" rsi_frac=0.5 "$@"
