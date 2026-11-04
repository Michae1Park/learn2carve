# Best carve so far (2026-11-04)

Seed 1, 12 000 steps (~17 min on one L40S). First run whose edge angle *rose* during training instead of fading flat.
Code state: commit `359073c` (hand reward, straight penalty, cap −10) plus the later fall-height / low-band change `4cd82a0`.

| Part | Value |
|---|---|
| Rider | baseline 28-DoF AMP humanoid, bindings 54° / 45° |
| Start pose | `envs/assets/stance_54_45_squat.yaml`: hip 0.62 m, torso twisted ~48° open to the nose, hips back, torso leaning ~36°, knees ~85° (built with `scripts/make_stance.py --angles 54 45 --hip_height 0.62 --open_deg 40 --pitch 35 --back 0.10 --twist 0.87 --tag squat`) |
| Task | carve, speed cap 9 m/s, command 6–8 m/s, `rew_speed_cap` −10, `rew_straight` −3 |
| Style rewards | `rew_facing` 1, `rew_hip` (head-to-knee L fold) 1, `rew_hand` 1 (lowest hand near the snow while edged), low ramp from the stance |
| Guide | `rew_guide` 10, never faded (`guide_fade_below_deg=0`) |

Eval (`scripts/eval_runs.py`, 128 episodes, deterministic):

| fell | fence | turns / 10 s | edge (turning) | pelvis | facing | hip flex | slip | seconds |
|---|---|---|---|---|---|---|---|---|
| 4.7% | 6.2% | 0.033 | **44°** | 0.275 m | 0.90 | 70° | 17° | 11.3 |

Open problem: it rides one long traverse instead of linked turns (turns ≈ 0).

Use it:
```bash
scripts/train_best.sh                                   # retrain the recipe
scripts/train_best.sh --init_from saved/best_carve      # warm start from this policy
scripts/py playground/snowboard/play.py saved/best_carve --livestream 2
scripts/py scripts/eval_runs.py saved/best_carve
```
