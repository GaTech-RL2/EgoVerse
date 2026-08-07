# DP small-circle sim_v2 evaluation protocol

Recorded on 2026-08-02 for the `dp_c4500_v2` policy evaluation on Skynet.

## Immutable inputs

- Simulator root: `/coc/flash7/paphiwetsa3/projects/sim_v2`
- Simulator package: `/coc/flash7/paphiwetsa3/projects/sim_v2/Tsimulation`
- Obstacle definitions: `/coc/flash7/paphiwetsa3/projects/sim_v2/Tsimulation/pushshapes/obstacles.py`
- Obstacle-definition SHA-256: `6a57e5c09ef76f12f34c48c1a8588fbbc3e308bff234d1d320783ee0aa280a2c`
- Curated obstacle-generation seeds: `/coc/flash7/paphiwetsa3/projects/sim_v2/recollect/eval_hard_seeds_obj_gen_v2_150.json`
- Seed-file SHA-256: `dce048a612e13667c8aa831b5f40689dc1b3c45cbc95236df737198d781e2dfc`
- Checkpoint: `/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-2/logs/dp_c4500_v2/checkpoints/epoch=0020-val_loss=0.0196.ckpt`
- Policy weights: EMA (`weights=auto`, following the checkpoint configuration)

The evaluator must export both of the following so it imports the new simulator and not the older copy in `EgoVerse-gmm-2`:

```bash
export EGOVERSE_ROOT=/coc/flash7/paphiwetsa3/projects/sim_v2
export PYTHONPATH=/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-2/external/diffusion_policy:/coc/flash7/paphiwetsa3/projects/sim_v2
```

## Obstacle-generation protocol

- Pusher embodiment: `circle_small`
- Solid-pusher physics: enabled (`PushShapesEnv(..., solid_pusher=True)`), matching data collection with `--solid-pusher`
- Object: `T`
- Obstacle levels: 1 through 30 from `sim_v2`
- Episodes: 5 curated seeds per level, 150 total
- Goal mode: `seeded`
- Observation history: 2 steps
- Receding-horizon actions executed per prediction: 8
- Maximum environment steps: 1,800
- Image size: 96x96 RGB
- Agent-position dimension: 2
- Per-episode score: maximum object-goal coverage (IoU) reached during rollout
- Success threshold: peak coverage >= 0.80
- Videos: all 5 episodes per level, MP4 at 10 FPS
- Sharding: one obstacle level per Slurm array task (30 tasks)
- Slurm job: `3638463`
- Launcher: `/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-2/dp_eval_small_circle_obs_gen_v2_solid.sbatch`
- Result directories: `/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-2/logs/dp_sc_v2_solid_obsgen_lXX`
- Videos: `/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-2/logs/dp_sc_v2_solid_obsgen_lXX/media/*.mp4`

## In-domain protocol

- Pusher embodiment: `circle_small`
- Solid-pusher physics: enabled, matching data collection
- Object: `T`
- Obstacle level: 0 (empty arena from `sim_v2`)
- Seeds: integers 0 through 39
- Episodes: 40 total
- Other rollout, score, success, and video settings: identical to the obstacle-generation protocol
- Sharding: eight Slurm array tasks, five seeds per task
- Slurm job: `3638475`
- Launcher: `/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-2/dp_eval_small_circle_indomain_v2_solid_sharded.sbatch`
- Seed shard files: `/coc/flash7/paphiwetsa3/projects/sim_v2/recollect/in_domain_seeds_00_04.json` through `in_domain_seeds_35_39.json`
- Result directories: `/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-2/logs/dp_sc_v2_solid_indom_sAA_BB`
- Videos: `/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-2/logs/dp_sc_v2_solid_indom_sAA_BB/media/*.mp4`

## Curated obstacle-generation seeds

| Level | Seeds |
|---:|---|
| 1 | 3238, 2351, 2558, 2963, 3450 |
| 2 | 2399, 2852, 3199, 2678, 3518 |
| 3 | 3671, 3704, 2209, 2748, 3525 |
| 4 | 3862, 3662, 2456, 2980, 3165 |
| 5 | 2587, 3092, 2767, 2940, 3854 |
| 6 | 2193, 2539, 2991, 2248, 2999 |
| 7 | 2598, 3164, 2979, 3827, 3573 |
| 8 | 2654, 3890, 2220, 3746, 3274 |
| 9 | 2644, 2462, 3757, 2970, 3149 |
| 10 | 3995, 3779, 2213, 2581, 3621 |
| 11 | 2939, 2877, 2777, 2730, 3847 |
| 12 | 3150, 2615, 3176, 3563, 3153 |
| 13 | 2993, 3392, 3514, 2329, 3642 |
| 14 | 3112, 2060, 2576, 2656, 2481 |
| 15 | 2598, 3426, 2821, 3901, 2833 |
| 16 | 3622, 3656, 2764, 3662, 2430 |
| 17 | 2135, 2016, 3603, 2974, 2581 |
| 18 | 3013, 2854, 2990, 2351, 3227 |
| 19 | 3356, 3031, 3561, 2056, 2030 |
| 20 | 2183, 3760, 3993, 2864, 3886 |
| 21 | 3956, 3995, 3079, 3553, 2425 |
| 22 | 3592, 3440, 3866, 2891, 2261 |
| 23 | 3967, 2977, 2972, 2890, 2359 |
| 24 | 2624, 2491, 2988, 3838, 3785 |
| 25 | 2401, 3392, 2161, 3642, 2044 |
| 26 | 3945, 2724, 3800, 3360, 3955 |
| 27 | 2416, 3381, 3687, 2955, 3215 |
| 28 | 2939, 2547, 2777, 2049, 3232 |
| 29 | 3566, 2049, 3709, 2834, 3365 |
| 30 | 2695, 2639, 2000, 2126, 2508 |

## Important provenance note

Earlier videos under `dp_sc_obsgen_eval_l*` imported the older `EgoVerse-gmm-2/Tsimulation` obstacle definitions. The first `dp_sc_v2_obsgen_l*` and `dp_sc_v2_indom_s*` runs imported `sim_v2` but accidentally left `solid_pusher=False`. Neither group is valid for this protocol. Only outputs beginning with `dp_sc_v2_solid_` use both the new obstacle definitions and the collection-matched solid-pusher physics.
