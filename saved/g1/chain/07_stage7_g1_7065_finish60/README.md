# 07 stage7_g1_7065_finish60
Edges switch at 60° across the hill. Warm-started (through the stage's earlier runs) from chain/06_stage6_g1_7065_edge75.

Training, last 20% of the run: turns_per_10s 3.42 | edge_abs_deg 57.23 | fell 0.06 | speed 8.67 | heading_abs_deg 42.75

The final G1 carver: stock Unitree G1 (Isaac Lab drives, arms at kp 300 / kd 15) on 70/65 bindings, 0.42 m stance,
hips 0.10 m toward the tail, pelvis hinged 50° forward at 0.60 m hip height (`stance_70_65_g1`). Play with
`--set robot=g1`.

Eval (`play.py`, 120 s at the full command, `ride.csv`): 6 of 6 finished runs reach the end of the 100 m course in
17.3–18.5 s, no falls, 6 linked turns each (AMP saved/04: 4–5), edge mean 57–59°, slip 1.5°, mean heading 42–44°,
track within ±11 m.
