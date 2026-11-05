# Stage 2 on the rebuilt rider (run 10-34-21)
Recipe: `scripts/train_stage2_transition.sh --timesteps 16000` at 336f08e, from scratch: rider 1.65 m, 0.56 m stance
(squat4), passive knee twist, softer arms, torso twist ±90°. Falls 8%, episodes 9.2 of 10 s, edge error 18°.
Eval: toe -> heel -> toe -> heel -> toe -> heel in one 11.6 s run. Switch test from a low toe carve: the policy reaches
the heel 75% with 0% falls (the old rider: 0%).
