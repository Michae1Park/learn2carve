#!/usr/bin/env bash
# Stage 2, edge changes (from scratch): start in a balanced carve (80% toe) with the timed edge command at the peak of
# that side, then heel <-> flat <-> toe on a timer (±15 -> 30°, period 6 -> 4 s), 10 s episodes on a 7° slope; half
# the episodes restart from recorded toe-carve states. Rider: SnowboardEnvCfg defaults. Reward: rewards/stage2.yaml.
# Extra args go to train.py, e.g. --init_from <run> --timesteps 12000 --set rsi_frac=0.3
cd "$(dirname "$0")/.." && exec scripts/py playground/snowboard/train.py --livestream 2 \
  --rewards playground/snowboard/rewards/stage2.yaml --set \
  slope_deg=7 episode_length_s=10 carve_start=true carve_start_toe_frac=0.8 "start_speed_turns=[4,5.5]" \
  "edge_cmd_amp_deg=[15,30]" "edge_cmd_period_s=[6.0,4.0]" rsi_frac=0.5 "$@"
