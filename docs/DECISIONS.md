# Decision & issue log

## Decisions

| ID | Date | Decision | Status |
|---|---|---|---|
| D-001 | 2026-10-01 | MVP = one clip, one slope, one style | Accepted |
| D-002 | 2026-10-01 | Imitation via AMP style reward, not behavior cloning | Accepted |
| D-003 | 2026-10-01 | Fork Isaac Lab's humanoid AMP example | Accepted |
| D-004 | 2026-10-01 | Coarse snow model: glide + sidecut steering + edge grip limit | Accepted |
| D-005 | 2026-10-01 | Build the sim before the clip | Accepted |
| D-006 | 2026-10-01 | Annealed pelvis assist force during training | Accepted |
| D-007 | 2026-10-02 | RL-only snowboard playground (side slip, skid, carve) before the clip | Accepted |
| D-008 | 2026-10-02 | Mountain = tilted ground + real gravity, env works in a slope frame | Accepted |

### D-001
Smallest thing that demonstrates "LfD + RL teaches a carve". Multiple styles, terrain curriculum,
procedural mountains and randomization are deferred until the MVP works.

### D-002
YouTube gives poses, not actions, so behavior cloning would first need an action-labelling model. AMP
learns directly from poses. The discriminator becomes the style reward, and "imitate → RL" is the same
training run with the task reward switched on.
*Revisit if* AMP can't learn from a single short clip. Then switch to a DeepMimic-style tracking
reward (match the clip frame by frame), which is more reliable with one clip.

### D-003
Isaac Lab ships a humanoid AMP env (direct workflow, skrl) with motion loading, discriminator and PPO.
Forking it means the only new pieces are the board, the slope, the snow force and the retargeting script.
Check the current Isaac Lab release for the exact task name/path before forking.

### D-004
PhysX has no snow model, and isotropic friction won't carve. The first draft (lateral damping that
grows with edge angle) only stops sideways sliding and doesn't steer the board. The model now has three
terms: glide friction + drag, a sidecut yaw term (arc radius `R·cos θ`), and a lateral grip limit
`μ_edge · N` that turns into a skid when exceeded. Board geometry comes from the real board spec
(`config.yaml`). Snow coefficients are typical hardpack values, tuned by two checks: turn radius vs.
edge angle, and top speed on 15°.
*Revisit if* the demo looks wrong in a way the policy can't fix. Then look at ski–snow contact
literature or model the progressive sidecut.

### D-005
The sim (slope, board, snow force) can be built and tested without any motion data, and it is the
riskiest part. The clip pipeline then has a concrete target to retarget onto.

### D-006
The snow model is approximate and the YouTube motion won't match our slope, speed or turn radius exactly,
so early training can stall at "falls immediately". A small force on the pelvis (residual force control)
helps keep the rider up and is annealed to 0 over training (`config.yaml` → `train.assist`).

### D-007
Before any demo exists, check that plain PPO on the Stage 1 env can ride at all (`playground/snowboard/`). All
three tasks are one env with two time-varying targets: path heading relative to the fall line and slip angle
(board axis vs. travel direction). Side slip = straight down at -90° slip, skid = sine heading with slip peaking
mid-turn, carve = sine heading at zero slip. No AMP, no assist force: the reward alone gets a side slip and a skidded S. A carve
holds a thin S for ~10 s and then speeds up until the turns flatten (results in `playground/snowboard/README.md`).
Policies from here are a baseline for Stage 4's "with vs. without demo" comparison.

### D-008
Tilted gravity on flat ground (D-004 era) looks wrong in the viewer: the rider leans and slides on a level floor.
The ground plane is now tilted instead (`CarveEnvCfg.mountain`), gravity points down, and `envs/mountain.py` adds
visual-only scenery. To keep the MDP identical, the env converts every read into a slope frame (x downhill, z =
snow normal) and every write back to the world, so observations, rewards and the snow model don't change and
policies transfer between the two modes. Stage 1 checks keep the flat mode (`mountain = False`).

## Issues

_None yet._

## Roadmap

| Stage | Status |
|---|---|
| 1 Sim | Not started |
| 2 Clip | Not started |
| 3 Train | Not started |
| 4 Show | Not started |
