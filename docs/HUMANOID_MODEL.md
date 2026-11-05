# Humanoid model

The rider is Isaac Lab's **28-DoF AMP humanoid** (`humanoid_28.usd`, capsule bodies), scaled to a 65 kg, 1.65 m rider and
bound to the board as one articulation (`envs/rider.py`, `envs/carve_env.py`). PhysX simulates it; the policy only
sets joint targets. Knobs: `config.yaml` → `rider`, plus the `CarveEnvCfg` flags of §6, which the demo run sets from its training scripts. SI units; angles in ° unless marked rad.

![Humanoid kinematic tree: pelvis root, abdomen, torso, neck, head, arms and legs, both feet bound to the board](rider/humanoid_tree.svg)

## 1. Bodies

| Body | Mass [kg] | Body | Mass [kg] |
|---|---|---|---|
| Pelvis (root) | 13.00 | Thigh ×2 | 5.90 |
| Torso | 12.32 | Shin ×2 | 4.92 |
| Head | 5.98 | Foot ×2 | 1.46 |
| Upper arm ×2 | 2.54 | Forearm ×2 | 1.57 |
| Hand ×2 | 0.45 | **Board** | 2.70 |

**Total 67.7 kg.** The pelvis carries **20%** of the rider's mass (`pelvis_mass_frac=0.2`; stock 12.6%, 8.17 kg), the
other bodies share the rest. Without it every body is the stock share × 1.82.

The stock humanoid is ~36 kg and 1.612 m. The demo scales its size ×1.024 to 1.65 m (`rider_to_height=true`: link
offsets, joint anchors and shapes), then its masses, inertias and joint gains by one mass factor, so the dynamics keep
their shape at the new weight:

$$k = \frac{m_{\text{rider}}}{\sum_i m_i} = \frac{65}{35.7} = 1.82 \qquad m_i \leftarrow k\,m_i,\quad I_i \leftarrow k\,I_i,\quad (K_p, K_d) \leftarrow k\,(K_p, K_d)$$

## 2. Joints and drives

Spherical joints are three revolute axes ($x, y, z$); knees and elbows are hinges. Every joint is a stiff PD
position drive (implicit in PhysX):

$$\tau_j = K_p\,(q_j^* - q_j) - K_d\,\dot q_j$$

| Joint | DoF | Range (x / y / z) | $K_p$ | $K_d$ |
|---|---|---|---|---|
| Abdomen | 3 | ±60 / −60…90 / ±50 (±90 in the demo) | 62 537 | 6 254 |
| Neck | 3 | ±50 / −40…60 / ±45 | 5 211 | 521 |
| Shoulder ×2 | 3 | −180…45 / −180…60 / ±90 | 20 846 | 2 085 |
| Elbow ×2 | 1 | −160…0 | 15 634 | 1 563 |
| Hip ×2 | 3 | −60…15 / −140…60 / −60…35 | 31 268 | 3 127 |
| Knee ×2 | 1 | 0…160 | 31 268 | 3 127 |
| Ankle ×2 | 3 | ±30 / ±55 / ±40 | 20 846 | 2 085 |

Gains are as reported by Isaac Lab after scaling (stock drives, the baseline; the demo changes some, see §6).
Joint velocity is capped at 100 rad/s.

## 3. Board binding

Both feet are welded to the board. A tree can't hold a loop, so the two bindings are split:

| Binding | Joint | Role |
|---|---|---|
| Front (left foot) | Fixed joint, **in the articulation tree** | Board is a child link of the left foot |
| Back (right foot) | Fixed joint, **excluded from the articulation** | Loop-closing constraint, solved iteratively |

The loop needs a converged solver: **32 position / 4 velocity iterations** at 120 Hz.

| Solver | Result |
|---|---|
| 4 / 0 (upstream) | Rider folds at the ankles, tips over standing still |
| 16 / 4 | Hips and ankles still sag ~0.05 rad |
| **32 / 4** | Within 0.02 rad, ~22% slower than 16 / 4 |

## 4. Stance (`scripts/make_stance.py`)

The default pose $q^{\text{stance}}$ is *found by simulation*, not hand-set: the pelvis and board are welded to the
world at the target poses, the binding constraints pull the feet onto the board, and the drives settle. The result
closes both loops exactly and is saved to `envs/assets/stance.yaml`.

| Setting | Value |
|---|---|
| Stance file | `stance_54_45_squat4.yaml` (`stance_tag=squat4`) |
| Stance | Regular (left foot forward), bindings +54° / +45°, 0.56 m apart |
| Command | `make_stance.py --angles 54 45 --hip_height 0.635 --open_deg 40 --pitch 35 --tail 0.123 --guess_from squat3 --to_height --stance_width 0.56` |
| Hip height | 0.635 m above the board top (a deep squat; baseline was 0.80 m) |
| Pelvis open | 40° from the toe edge toward the nose, hinged forward 35°, shifted 0.123 m toward the tail |
| Chest twist (`abdomen_z`) | +0.83 rad (47°) toward the nose |
| Knees (front / back) | 79° / 90° bent |

Stance files are named by binding angles, `stance_<front>_<back>[_<tag>].yaml`. `stance_54_45.yaml` is
`config.yaml`'s default; the unsuffixed `stance.yaml`/`stance_squat.yaml` belong to the old 42°/27° stance and are no longer read.

## 5. Actions → joint targets

The demo policy has **30 actions**: 2 synergy actions + 28 joint residuals (26 joints + 2 knee flexions; the knee
twist and the boots' stance are left to their drives). Targets add up as

$$q^* = \operatorname{clip}\!\left(q^{\text{stance}} + 0.3 \cdot 0.5 \cdot \tfrac{1}{2}(q^{\max} - q^{\min}) \cdot a_j + \textstyle\sum_k s_k\,\rho_k + \text{toe}(s_0),\ q^{\min},\ q^{\max}\right)$$

| Term | Value |
|---|---|
| Joint residual $a_j \in [-1, 1]$ | $\pm15\%$ of each joint's half-range (action scale 0.5 × synergy residual 0.3); **arms ±50%** (`arm_action_scale=0.5`, no residual factor) |
| Edge synergy $s_0$ | Both ankle flex +0.5 rad and both knees +0.6 rad per unit: one number rolls the board onto its edge (+1 toe) |
| Lean synergy $s_1$ | Abdomen side-bend 0.9 rad per unit (counter-angulation) |
| Toe extra | For $s_0 > 0$ also knees +0.4 rad and hips pitch −0.3 rad per unit: the toe edge comes from a squat |
| Smoothing | Both the synergy and joint actions go through an EMA, weight 0.35 on the new value (~0.1 s at 30 Hz), to stop chatter |

Without synergy (baseline) the 28 joint actions move $\pm 50\%$ of the half-range directly.

## 6. Rider options (`CarveEnvCfg` flags, `--set`)

The defaults are the baseline above. The demo run (`scripts/train_stage2_transition.sh`, `train_stage3_full_s.sh`) sets
the **bold** rows; the rest are available but off.

| Flag | Change |
|---|---|
| **`rider_to_height=true`** | Humanoid scaled ×1.024 to the 1.65 m of `config.yaml` |
| **`stance_width=0.56`** | Bindings 0.56 m apart (config default 0.56 as well) |
| **`binding_angles_deg=[54,45]`** | Forward stance, the config default; hips face more toward the nose |
| **`pelvis_mass_frac=0.2`** | **Heavy root**: the pelvis carries 20% of the mass (stock 12.6%), the rest scale down to keep 65 kg |
| **`knee_rotation=true`**, **`passive_joints=[knee:2]`** | **Knee twist.** Each knee becomes 2 DoF: `knee:0` flexion (0–160°) + `knee:2` axial twist about the shin (±20°, 0.3× the flexion stiffness). The twist is **passive**: held by its spring and damper, observed, not in the action space |
| **`ankle_range_deg={x:45,y:80,z:55}`** | Ankle limits widened from ±30 / ±55 / ±40° (the squat stance sat at the stock limits) |
| **`stiff_boots=true`**, **`boot_range_deg={y:[-5,15], x:[-5,5], z:[-5,5]}`** | **Snowboard boots.** Ankles held within roll ±5°, twist ±5° and flex −5…+15° of their stance angle, drives 3× stiffer: the edge has to come from the hips and knees. The flex range is asymmetric so the shin can lean into the toe side |
| **`torso_twist_deg=90`** | `abdomen_z` range ±90° (stock ±50°; the stance already twists +47°, which left no room to turn the chest past the nose for a heel turn) |
| **`arm_stiffness_scale=0.3`**, **`arm_action_scale=0.5`** | Softer arms (shoulder/elbow gains ×0.3) with a larger action range, so they can reach for the edge |
| **`synergy=true`**, `synergy_toe_rad`, `synergy_smooth=0.35`, `joint_smooth=0.35` | Edge and lean actions (§5) |
| `thoracic=true` | **Second spine joint.** The torso splits into a lumbar segment (4.4 kg) and a chest (12.6 kg) joined by a 2-DoF thoracic joint, `thorax:0` roll (±40°) and `thorax:1` pitch (−40…60°). Neck and shoulders move to the chest. **30 DoF** |
| `compliant_drives=true` | **Per-group stiffness** instead of one uniform, very stiff drive (damping follows, keeping each joint's damping ratio) |
| `neck_stiffness_scale=0.5` | Softer neck: the head moves instead of being held rigid |
| `spine_damping_scale=2` | Extra damping on abdomen + thorax: no torso jerks |
| `physics_hz=240`, `solver_iterations=[16,4]` | Physics at 240 Hz (decimation 8), fewer solver iterations |
| `leg_action_scale=0.5` | Hips and knees get ±50% of their half-range per unit action (others ±25%) |
| `board_flex=true` | Two-half board, see [board model §4](BOARD_MODEL.md#4-two-half-board-torsion-camber-flex-progressive-sidecut) |

**Spine:**

```
pelvis ─abdomen (3)─ lumbar ─thorax (2)─ chest ─┬─ neck (3) ─ head
                                                └─ shoulders (3) ─ arms
```

**Compliant drives** (stiffness × scale, on top of the mass scaling):

| Group | Hip | Knee | Ankle | Abdomen | Thorax | Shoulder | Elbow | Neck |
|---|---|---|---|---|---|---|---|---|
| Scale | 1.0 | 1.0 | 0.7 | 0.35 | 0.35 | 0.3 | 0.3 | 0.4 (× 0.5 neck scale) |

Legs carry the body and stay stiff; the spine, arms and neck give, as a human's do.

### Checks (`scripts/rider_check.py`: stance held 3 s, zero action)

Measured on the earlier rider (baseline stance, before the rebuild); not re-run on the demo rider.

| Model | Worst sag | Falls | Step time |
|---|---|---|---|
| Baseline: 120 Hz, 32 / 4 | 0.014 rad | 0% | 1.0× |
| 240 Hz, 8 / 4 | 0.055 | 0% | 1.1× |
| 240 Hz, 12 / 4 | 0.023 | 0% | 1.2× |
| **240 Hz, 16 / 4** | **0.013** | 0% | — |
| 120 Hz, 12 / 4 | 0.044 | **100%** | 0.6× |
| **Full new rider + two-half board**, 240 Hz, 16 / 4 | **0.017** (board bend) | 0% | 2.0× (training throughput) |

- **Velocity iterations don't help** (12 / 4 and 12 / 6 are identical).
- **The split torso first sagged 0.15 rad and fell.** The chest sphere overlaps the pelvis's upper-waist sphere by
  ~3 cm; once the lumbar segment sat between them they were no longer neighbours, so PhysX pushed them apart every
  step. The pair is now filtered.

## Known gaps

| Gap | Effect |
|---|---|
| **No torque limits** (effort limit = float max) | Joints are stronger than a human's: balance is easier than it should be |
| No wrists, no toes | Fine in boots; hands can't counter-balance finely |
| CoM not observed | The policy must infer its balance from body pose |
