# Arc-tok rotation-fixed training runs

## What this is

Nine training runs on the `aniketh/arc` branch that exercise the
**rotation-always** arc-length tokenizer (commit `a412dcfe`). Each run
matches an earlier `_h100_bs128_lr3e4_val30_ckpt30_D40M100_amatch_*` run
verbatim on training setup — the only difference is that the model now
outputs 14 dims per token (xyz + ypr + gripper per arm) instead of 8
(xyz + gripper per arm). Rotation is unconditionally supervised.

All runs log to wandb project `rl2-group/arc` with tag `rotation-fixed`.

## Prerequisites

1. On branch `aniketh/arc` at commit `a412dcfe` or later (this is where
   `TokenizeBimanualArcLengthCartesian` outputs `(M+1, 14)` including
   SLERPed ypr).
2. `.venv` at `/storage/project/r-dxu345-0/acheluva3/Egoverse-arc/.venv`
   with dependencies from `requirements.txt`.
3. Dataset synced locally at `/storage/project/r-dxu345-0/shared/arc_tests`
   for the SQL filters used by each data config.
4. Wandb credentials at `~/.netrc` (already set up).
5. Valid billing minutes on `gts-dxu345-rl2` (checked via `sshare`).

## The 9 runs

| Tag | Data config | Model config | Kind |
|---|---|---|---|
| arctok_cotrain | `arc_tests_cotrain_arctok` | `hpt_cotrain_mecka_flow_shared_head_arc_D40_M100` | arc-tok |
| arctok_mixture | `mecka_folding_eva_fold_cotrain_arctok` | `hpt_cotrain_mecka_flow_shared_head_arc_D40_M100` | arc-tok |
| arctok_mecka_folding | `mecka_folding_clothes_arctok` | `hpt_cotrain_mecka_flow_shared_head_arc_D40_M100` | arc-tok |
| veldec | `mecka_folding_eva_fold_cotrain_arctok` | `hpt_cotrain_mecka_flow_shared_head_arc_D40_M100_veldec` | arc-tok + MLP vel head |
| velreadout | `mecka_folding_eva_fold_cotrain_arctok` | `hpt_cotrain_mecka_flow_shared_head_arc_D40_M100_velreadout` | arc-tok + Linear vel readout |
| eva_only_arctok | `eva_only_fold_arctok` | `hpt_cotrain_mecka_flow_shared_head_arc_D40_M100` | arc-tok (robot only) |
| baseline_cotrain | `arc_tests_cotrain` | `hpt_cotrain_mecka_flow_shared_head` | non-arc baseline |
| baseline_mixture | `mecka_folding_eva_fold_cotrain` | `hpt_cotrain_mecka_flow_shared_head` | non-arc baseline |
| baseline_mecka_folding | `mecka_folding_clothes` | `hpt_cotrain_mecka_flow_shared_head` | non-arc baseline |

## Common setup (all 9 runs)

```
trainer.strategy=ddp_find_unused_parameters_true
trainer.check_val_every_n_epoch=30
callbacks.model_checkpoint.every_n_epochs=30
trainer.max_epochs=3000
trainer.min_epochs=3000
paths.dataset_dir=/storage/project/r-dxu345-0/shared/arc_tests
launch_params.nodes=1
launch_params.gpus_per_node=1
+evaluator.arc_match_distance=0.4
+evaluator.arc_match_points=15
+logger.wandb.tags=[rotation-fixed]
```

Batch size (128), LR (3e-4), val every 30 epochs, ckpt every 30 epochs
are inherited from the model config defaults.

## Arc-tok specifics (6 runs)

Add these overrides to every arc-tok launch:

```
arc_tokenizer.min_distance_unit=0.4
arc_tokenizer.resampled_vector_length=100
evaluator=eval_arctok_D40_M100
```

`D=0.4 m`, `M=100 waypoints` → each sample is `(101, 14)`. Rotation is
SLERPed across waypoints; velocity row (last row) holds per-axis mean
rate for xyz + ypr + gripper.

## Non-arc baseline specifics (3 runs)

```
data.train_datasets.human_bimanual.resolver.transform_list.stride=1
data.valid_datasets.human_bimanual.resolver.transform_list.stride=1
evaluator=hpt/base
```

Baselines predict raw time-indexed `(100, 14)` actions directly (no
tokenization). The human stride override matches the arc-tok pipeline's
implicit stride=1 so training samples are comparable.

## Per-run tweaks

The two `mecka_folding_clothes*` runs (arctok_mecka_folding,
baseline_mecka_folding) additionally pass:

```
+evaluator.limit_val_batches=40
```

so val doesn't dominate wall clock on that smaller dataset.

## Description strings (wandb id template)

Wandb id resolves as `${name}_${description}_${now:%Y-%m-%d_%H-%M-%S}`,
so pick descriptions carefully to keep runs identifiable:

- arctok runs: `description=rotation_fixed_D40M100_amatch`
- veldec: `description=rotation_fixed_D40M100_amatch_veldec`
- velreadout: `description=rotation_fixed_D40M100_amatch_velreadout`
- baseline runs: `description=rotation_fixed_D40M100_amatch`

## Launcher script

The full 9-run launcher lives at `/tmp/relaunch_9_rotation_fixed.sh`
(L40S variant) and `/tmp/relaunch_9_rotation_fixed_bw.sh` (Blackwell
variant). Each is a bash script that fires 9 `nohup python
egomimic/trainHydra.py … -m` invocations in parallel. Submitit takes
each one, packages the job, and submits to the target partition.

To target a different partition, swap the `hydra/launcher=...` value in
`COMMON`:

- `hydra/launcher=submitit_pace_h100` → gpu-h100
- `hydra/launcher=submitit_pace_l40s` → gpu-l40s
- `hydra/launcher=submitit_pace_a100` → gpu-a100
- `hydra/launcher=submitit_pace_blackwell` → gpu-rtxpro-blackwell

## Fresh vs resume

**All 9 runs are FRESH** (no `+ckpt_path`, no `+logger.wandb.id`, no
`+logger.wandb.resume`). The old 8-dim arc-tok checkpoints are
architecturally incompatible with the new 14-dim output layer, so
resume from `last.ckpt` would fail with a shape mismatch on the head
projection weights. Non-arc baselines could technically resume their
old ckpts (still 14-dim natively) but are relaunched fresh for
apples-to-apples comparison with the arc-tok variants.

## Log dirs

Hydra writes to `./logs/<name>/rotation_fixed_D40M100_amatch_<ts>/0/`.
Symlinks pointing at all 9 real run dirs live at
`logs/rotation-fix/<name>__<description_ts>` for easier navigation.

## Known operational issues

1. **Billing quota kills**: the `gts-dxu345-rl2` account occasionally
   trips `AssocGrpBillingMinutes`, which sends `SIGUSR2` mid-run.
   Submitit's signal handler saves + exits cleanly, so slurm reports
   `COMPLETED` even though training barely started. Check
   wandb `_runtime` (should be many minutes) vs slurm `Elapsed` (may
   show hours) to distinguish real training from norm_stats + kill.
2. **Norm-stats cold start**: fresh runs recompute normalization stats
   on 100% of the training set (~20-30 min on eva_only_fold_arctok,
   longer on mixtures). To skip this, either pass
   `+norm_stats.precomputed_norm_path=<abs_path_to_norm_stats.json>`
   from a previous run, or shrink the sample with
   `norm_stats.sample_frac=0.2` (20% subsample computes in ~5 min).
3. **Partition availability** (checked via `sinfo`, `squeue`): at time
   of writing all GPU partitions are fully allocated. Best queue-turnover
   options: `gpu-v100` and `gpu-rtx6000`. Worst: `gpu-l40s`.

## Verification checklist before firing

```bash
# 1. Confirm branch + commit
git rev-parse --abbrev-ref HEAD    # aniketh/arc
git log --oneline -1               # a412dcfe (or later)

# 2. Hydra compose smoke test
source .venv/bin/activate
python -c "
from hydra import compose, initialize
with initialize(config_path='egomimic/hydra_configs', version_base=None):
    cfg = compose(config_name='train_zarr_cartesian', overrides=[
        'model=hpt_cotrain_mecka_flow_shared_head_arc_D40_M100',
        'data=mecka_folding_eva_fold_cotrain_arctok',
        'evaluator=eval_arctok_D40_M100', 'paths.dataset_dir=/tmp',
    ])
    print(cfg.model.robomimic_model.head_specs.shared.fm_policy.model.act_dim)
"
# expect: 14

# 3. Billing quota
sshare -A gts-dxu345-rl2 -u acheluva3 -o Account,User,EffectvUsage
# want EffectvUsage < 0.90 to avoid quota kills

# 4. Fire the launcher (choose partition first)
bash /tmp/relaunch_9_rotation_fixed_bw.sh
```

Confirm submission by running `squeue -u acheluva3 -o "%A %j %T %R %P"`
and filtering StdOut paths for `rotation_fixed_`.

## Sanity check after runs start

- Each run's `logs/.../0/wandb/run-*/logs/debug.log` should contain a
  line "Syncing run <name>" indicating wandb init succeeded.
- On successful training, wandb's `Train/Loss` starts logging within a
  few minutes of `Initializing distributed`. If minutes go by with no
  `Train/Loss` metric on wandb, the run is stuck in norm_stats or has
  been quota-killed pre-training.
