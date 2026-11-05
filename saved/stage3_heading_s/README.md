# Stage 3, heading-switched S turns
Recipe: `scripts/train_stage3_full_s.sh` (flat start, 20 s runs, 9° slope, edge command ±25 → 45°, edges switch when the
turn reaches 50° across the hill or after 3 s), warm-started from the stage 2 policy on the rebuilt rider (1.65 m,
0.56 m stance, passive knee twist, softer arms, boots, toe-side squat, hand grab, action smoothing).

Training (end): edge error 6.2°, command at its hardest level (±45°), falls 2%, fence 9%, episodes 18.7 of 20 s.
Inside hand to its grab point during turns: heel turn 0.45 m, toe turn 0.63 m.

Eval (`play.py`, 90 s, `trail.png` and `ride.csv` are its longest run): edge changes every ~3 s (the timer backstop,
turns rarely reach 50° in time), toe edge mean 34° (max 56°), heel edge mean 32° (max 51°), slip 4°, heading −69° to
+62°. Play runs end at the fence or in a fall more often than in training (11 runs: 3 falls, 7 fence).
