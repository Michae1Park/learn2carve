#!/usr/bin/env bash
# Stage 2, transitions: start in a balanced carve (as stage 1) with the timed edge command at the peak of that side,
# so the first edge change comes ~1 s in; then heel <-> base <-> toe every ~2 s. Warm start from stage 1:
#   scripts/train_stage2_transition.sh --init_from logs/playground/snowboard/<stage 1 run> --timesteps 12000
cd "$(dirname "$0")/.." && exec scripts/train_edge_cmd.sh slope_deg=7 "ankle_range_deg={x: 45, y: 80, z: 55}" \
  synergy=true carve_start=true edge_cmd_hold=false "start_speed_turns=[5,7.5]" \
  rew_pose_ref=10 rew_fall=-10 episode_length_s=10 "edge_cmd_period_s=[6.0,4.0]" edge_cmd_sigma_deg=30 "start_speed_turns=[4,5.5]" rew_edge_side=20 "edge_cmd_amp_deg=[15,30]" lying_deg=95 carve_start_toe_frac=0.8 rew_tilt=-10 rew_rise=30 rsi_frac=0.5 "$@"
