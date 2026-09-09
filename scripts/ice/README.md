# ICE training tools

These tools provide a portable, fail-closed path for one-GPU Planar V2 BC
training on PACE ICE. Site paths, source identity, dataset identity, output,
W&B identity, GPU type, and checkpoint validation are inputs; none are embedded
in the launcher.

## Prepare an immutable source/runtime

Use a clean, detached or branch checkout on the current scratch path reported
by `pace-whoami`. Capture the environment after installing from the repository
lock:

```bash
python scripts/ice/capture_runtime_lock.py \
  --repo /absolute/repo \
  --expected-head 0123456789abcdef0123456789abcdef01234567 \
  --lock-input /absolute/repo/uv.lock \
  --output /absolute/run-bundles/runtime-lock.json
```

The command refuses a dirty checkout and refuses to overwrite a lock.

## Required launcher inputs

Export these before `sbatch --export=ALL`:

- `ICE_REPO`, `ICE_EXPECTED_HEAD`, and `ICE_PYTHON`
- `ICE_TOOL_DIR` (optional only when the tools are at
  `$ICE_REPO/scripts/ice`)
- `ICE_CHECKPOINT_VALIDATOR`, an absolute executable path to
  `validate_lightning_checkpoint.py`
- `ICE_EXPECTED_SPLIT_MANIFEST_SHA256`, matching the exact committed Planar
  U-Socket split manifest (override `ICE_SPLIT_MANIFEST` only with identical
  bytes at another absolute path). Its current committed SHA-256 is
  `3683e3461596eef8df2432fa865779b3c77b2a2057dabd0fea125595729cf313`.
- `ICE_DATASET_DIR` and a new, absolute `ICE_OUTPUT_DIR`
- `ICE_EXPERIMENT`, exactly one of
  `pusht/planar_v2_usocket_direct_bc`,
  `pusht/planar_v2_usocket_arc_bc`, or
  `pusht/planar_v2_usocket_arc_hybrid_D40_M16_R24deg_bc`
- `ICE_WANDB_ENTITY`, `ICE_WANDB_PROJECT`, and a unique
  `ICE_WANDB_RUN_ID`
- one GPU selected by the launcher's `H100|H200` Slurm constraint
- `ICE_MAX_STEPS`, `ICE_VAL_CHECK_INTERVAL`,
  `ICE_LIMIT_TRAIN_BATCHES`, `ICE_LIMIT_VAL_BATCHES`,
  `ICE_CHECKPOINT_EVERY_N_STEPS`, `ICE_TRAIN_BATCH_SIZE`, and
  `ICE_VALID_BATCH_SIZE`

Optional settings include `ICE_WANDB_NAME`, `ICE_WANDB_GROUP`,
`ICE_WANDB_TAGS`, `ICE_NORM_STATS_PATH`, `ICE_SEED`, and
`ICE_MAX_RESTARTS`. Authentication remains in the user's normal W&B
environment; never place credentials in a submission script.

On the first attempt, omitting `ICE_NORM_STATS_PATH` computes fresh statistics
from the pinned training split and writes
`$ICE_OUTPUT_DIR/norm_stats/norm_stats.json`. Every Slurm restart must reuse
that exact file; it will not recompute or overwrite normalization. Supplying a
precomputed file or directory likewise sets the cache destination to `null`.

The safe default is `ICE_LAUNCH_MODE=preflight`. It validates source and input
paths, resolves the full Hydra config, checks the one-GPU/runner-owned contract,
re-enumerates the physical dataset through the training resolver, reproduces
the seed-42 1% episode split with zero ID/path overlap and complete union, and
dry-runs the generic checkpoint runner without creating the requested run
directory. `ICE_LAUNCH_MODE=dry-run` is an alias. Set
`ICE_LAUNCH_MODE=run` only after reviewing the emitted preflight record.
The validator records new resolved-path hashes on ICE; it does not compare the
manifest's Skynet-specific absolute-path hashes. Episode IDs and their hashes
must match exactly.

Probe the exact real allocation before submission, for example:

```bash
sbatch --test-only --account=coc --partition=ice-gpu --qos=coc-ice \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G \
  --time=16:00:00 --gres=gpu:1 --constraint='H100|H200' \
  scripts/ice/launch_planar_bc.sbatch
```

Then submit exactly one selected allocation. The batch job records the Slurm
allocation and full `nvidia-smi` report, rejects nonzero uncorrectable ECC or a
pending repair/reset, performs a real BF16 forward/backward pass, and completes
a one-rank NCCL all-reduce before training. Training uses world size one,
`strategy=auto`, immutable
epoch/optimizer-step checkpoints, and the generic requeue runner as the sole
owner of `scontrol requeue`.

The runner validates every candidate checkpoint, forwards `SIGUSR2` to request
a fresh full-state checkpoint at the `USR1` boundary, resumes the same W&B ID,
and writes `COMPLETE.json` only after successful training and terminal
checkpoint validation. It never opts into unvalidated checkpoints. For a long
run, launch the separate CPU checkpoint mirror described by
`ice_checkpoint_mirror.sbatch`; do not put mirroring inside the GPU job.

## Multi-node checkpoint mirror pool

For sustained or multi-run training, use `ice_checkpoint_mirror_pool.sbatch`
instead of the single-node wrapper. Submit it as a bounded Slurm array, with one
exclusive `ice-cpu` node per array element:

```bash
export ICE_MIRROR_POOL_SIZE=4
export ICE_MIRROR_POOL_SCRIPT=/absolute/repo/scripts/ice/ice_checkpoint_mirror_pool.py
export ICE_MIRROR_PYTHON=/absolute/environment/bin/python
sbatch --array=0-3%4 scripts/ice/ice_checkpoint_mirror_pool.sbatch
```

All workers must receive the same manifest, state directory, scratch root,
quota, destination host, and SSH identity. The shared state directory must be
on ICE storage visible from every allocated node. Workers rotate their scan
order and take a nonblocking per-checkpoint lock, so validation, hashing, and
upload of different checkpoints proceed concurrently without duplicate
transfers. Short state and event locks prevent lost updates.

`ICE_MIRROR_PYTHON` is mandatory and must be the absolute interpreter that can
import every validator dependency. The wrappers never infer Python from
`PATH`; this prevents a healthy monitor from silently rejecting checkpoints
because a site interpreter lacks PyTorch.

Set `ICE_MIRROR_INVENTORY_ROOT` to a bounded project or campaign directory to
publish `archive-inventory.json` in the shared state directory. The inventory
discovers checkpoint-bearing run directories and separates registered archived,
registered pending, and unregistered runs. It recognizes completion only from a
valid run-local `COMPLETE.json`; Slurm state alone is not completion evidence.
Discovery is read-only and never guesses a validator, retention policy, or
Skynet destination for an unregistered run.

Array element zero is the maintenance worker. It alone measures whole-scratch
pressure, prunes when explicitly enabled, and publishes `mirror-complete.json`.
Pruning still requires a fresh local validation/hash, an exact remote SHA-256
match, a final unchanged local stat, and retention of at least two recent local
checkpoints per run. A worker or transfer failure never signals a GPU job.

Start with two to four workers and increase only after measuring aggregate
throughput: every worker reads the shared ICE filesystem and writes through the
same remote gateway. `#SBATCH --exclusive` guarantees distinct nodes but makes
the allocation intentionally expensive; remove it only if separate nodes are
not required.
## Clean standard-DP campaign

`launch_planar_v2_dp_cotrain_clean.sbatch` is the single authoritative ICE
entrypoint for the clean Planar V2 standard-DP campaign. Its modes are:

- `MODE=preflight`: resolve and validate source, data, split, normalization,
  pipeline stages, parameter count, and run identity without allocating a GPU.
- `MODE=smoke`: run two real optimizer steps, scheduled validation,
  EnergyScore@32, checkpoint save, and strict reload on one H100 or H200.
- `MODE=full`: start a 240,000-step run from scratch on one H100 or H200 using
  a passed smoke's exact split and normalization artifacts.

Full mode deliberately rejects `ICE_INITIAL_CHECKPOINT`,
`ICE_INITIAL_CHECKPOINT_SHA256`, and `CKPT_PATH`. Its resolved Hydra config must
contain `ckpt_path: null`. A Slurm requeue may use only the scheduler-generated
HPC checkpoint from the same job; it is not an experiment checkpoint restore.

Submit the GPU modes with exactly one compatible GPU and the cluster's current
eight-hour limit, for example:

```bash
sbatch --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G \
  --gres=gpu:1 --constraint='H100|H200' --time=08:00:00 \
  --requeue --signal=B:USR1@600 --export=ALL,MODE=full,... \
  scripts/ice/launch_planar_v2_dp_cotrain_clean.sbatch
```

Always run the exact command through `sbatch --test-only` immediately before
the real submission. Record the actual GPU from `nvidia-smi`; H100 and H200 are
interchangeable for this campaign and do not by themselves invalidate a passed
smoke. Do not infer full-run completion from a completed smoke: the W&B ID and
output directory include `smoke` or `full` explicitly.

The trainer publishes aggregate and per-domain `Train/MSE` on optimizer steps
as well as at epoch end. A running full job is not considered healthy until its
W&B history contains a finite step-level MSE; scheduler `RUNNING` alone is not
enough.
