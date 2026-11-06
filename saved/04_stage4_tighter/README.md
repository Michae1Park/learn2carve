# Stage 4, tighter turns
Recipe: `scripts/train_stage3_full_s.sh` with `edge_cmd_heading_deg=40 edge_cmd_center_max_deg=10` (switch the edge at
40° across the hill instead of 50°, centring shift capped at 10°), warm-started from `saved/03_stage3_centred`,
16000 steps. Checkpoint is the run's final one.

Eval (`play.py`, 90 s at the full command, `trail.png` and `ride.csv` are its longest run): 5 of 5 finished runs reach
the end of the 100 m course in 15–17.5 s, 4–5 linked turns each (03: 3–4), edge mean 39–40°, slip 3°, mean heading
35–39° (03: 38–42°), track within ±15 m (fence ±20 m). The overshoot of 03 is gone: no flat cross-hill segments. Left:
small skid smudges at each edge change.
