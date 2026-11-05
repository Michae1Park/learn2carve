#!/usr/bin/env bash
# The base carve setup the stage scripts build on. Extra args go to train.py, e.g.
#   scripts/train_best.sh --seed 2
cd "$(dirname "$0")/.." && exec scripts/py playground/snowboard/train.py --task carve --livestream 2 \
  --set edge_target_deg=40 "binding_angles_deg=[54,45]" stance_tag=squat speed_cap=9.0 "speed_cmd=[6,8]" \
  rew_guide=10 pelvis_mass_frac=0.2 "$@"
