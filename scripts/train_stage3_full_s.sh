#!/usr/bin/env bash
# Stage 3, full S: start flat at a moderate speed (no carve start), 20 s runs, the timed edge command heel <-> toe at
# higher amplitude (25 -> 45 deg) on a steeper slope (9 deg), so the speed that balances a 40 deg+ edge is reachable
# (sin θ = v² / (g · 7.6 m): 40 deg needs ~7 m/s). Same rider and rewards as stage 2. Warm start from stage 2:
#   scripts/train_stage3_full_s.sh --init_from logs/playground/snowboard/<stage 2 run> --timesteps 16000
cd "$(dirname "$0")/.." && exec scripts/train_stage2_transition.sh carve_start=false episode_length_s=20 slope_deg=9 \
  "edge_cmd_amp_deg=[25,45]" "edge_cmd_period_s=[5.0,4.0]" rsi_frac=0.3 rew_grab=10 "$@"
