# Stage 3, centred S turns
Recipe: `scripts/train_stage3_full_s.sh` (heading-switched edge command, the switch heading shifted 2°/m toward the
piste centre, 4 s backstop), warm-started from `saved/02_stage3_heading`.

Training (end): edge error about 5°, command at its hardest level (±45°), falls 0%, fence 2%, episodes 19.6 of 20 s,
1.7 turns per 10 s.

Eval (`play.py`, 90 s at the full command, `trail.png` and `ride.csv` are its longest run): 5 of 5 finished runs reach
the end of the 100 m course, 3–4 linked turns each, edge changes every ~3.8 s, toe edge mean 42° (max 52°), heel edge
mean 38° (max 48°), slip 2.5°, track within y −12.7 to +7.9 m (fence ±20 m). Turns overshoot: heading at the switch
reaches 56–91° across the hill.
