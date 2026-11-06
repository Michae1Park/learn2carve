#!/usr/bin/env bash
# Stage 3, full S: the stage 2 rider, started flat (no carve start), 20 s runs on a 9° slope at ±25 -> 45°, so the
# speed that balances a 40°+ edge is reachable (sin θ = v² / (g · 7.6 m): 40° needs ~7 m/s). Edges switch on heading:
# when the turn reaches 50° across the hill (shifted 2°/m toward the piste centre so the S stays between the fences),
# after 1 s at least and 4 s at most. Reward: playground/snowboard/rewards/stage3.yaml. Warm start from stage 2:
#   scripts/train_stage3_full_s.sh --init_from logs/playground/snowboard/<stage 2 run> --timesteps 16000
cd "$(dirname "$0")/.." && exec scripts/train_stage2_transition.sh \
  --rewards playground/snowboard/rewards/stage3.yaml --set carve_start=false episode_length_s=20 slope_deg=9 \
  "edge_cmd_amp_deg=[25,45]" "edge_cmd_period_s=[5.0,4.0]" rsi_frac=0.3 \
  edge_cmd_mode=heading edge_cmd_max_s=4 edge_cmd_center_deg_per_m=2 "$@"
