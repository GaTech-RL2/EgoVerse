# Lambda: independent opscale training and checkpoint evaluation

Base: `79a9815c9e3990d26a47bed784a243075808489b`. This addition is prepared in an isolated worktree; existing Pace training code and jobs are untouched. No training or GPU evaluation has been launched from this package.

## Scientific recipe and scope

- 36 cells: budgets **24, 32, 48, 64, 72, 96 h** × **4, 8, 16, 32, 64, 128 operators**.
- Existing RDT recipe, batch **64**, one GPU,100 optimizer updates/epoch, original AdamW settings and WSD7000/323000/70000 update schedule, target400000 updates/4000epochs. Full optimizer/scheduler/loop state resumes from the checkpoint; this is not a weights-only restart.
- **EMA is off**, as in the actual reference config/checkpoints. Existing checkpoints containing EMA weights are rejected by this setup rather than silently changing the recipe.
- Training performs **no validation** and saves **every50epochs/5000updates**, preserving every complete checkpoint plus a `last.ckpt` symlink.
- Evaluation covers original `seen_matched`, original `seen_perlevel`, and the corrected `unseen` cohort. Preparation verifies unchanged training membership, operators, seen membership and seen video pins. The unseen cohort must be the same400episodes/20operators in all36cells and disjoint from the union of training episodes/operators.
- Validation remains **800batches per cohort**, original batch32, exact even-stride reads, tracking-valid scoring and the existing W&B metrics. Pinned videos retain the original200epoch cadence and final evaluation. The dominant metric remains the existing `keypoints_paired_mse_avg` metric; its meaning is unchanged.
- Each cell has a dedicated training W&B ID and a separate evaluation W&B ID, both linked to its source run by tags. A training process and an evaluation process never write concurrently to the same W&B run.

## Files that must be staged on local Lambda storage

Use local SSD/NVMe paths for code, Python environment, Hugging Face/model caches, canonical Zarr data, normalization, checkpoints and outputs. Do not reuse the stalled NFS runtime path.

1. This repository state and its Python environment/dependencies (see the repository lockfiles). The expected environment has the same Lightning/PyTorch/model packages as the source run. Copying a Pace virtualenv is not portable; recreate it on Lambda and perform the requested runtime checks there.
2. Original private metadata root and corrected private metadata root, both containing `splits/`; the corrected root also needs `moving_opscale_design.json` and the correctly staged `canonical_zarr/` index/data.
3. Per-cell original `norm_stats.json`, unchanged. The new unseen cohort must not recompute training normalization.
4. Downloaded full-state checkpoints and their SHA256 identity. The export producer's immutable READY records include object key, byte size, source run ID, epoch/update counters and SHA256.
5. Credentials provisioned privately in the destination environment; none belong in this code, manifest, handoff or ZIP.

The fixed local paths in a prepared manifest belong to that Lambda machine. Prepare after staging, rather than copying a manifest prepared for Pace.

## Prepare all72 configurations and launch scripts

Create a private input JSON outside Git (all36cells are mandatory):

```json
{
  "private_dir": "/local/private_corrected",
  "source_private_dir": "/local/private_original",
  "run_root": "/local/runs/opscale_lambda_async_v1",
  "evaluation_campaign": "fold-moving-opscale-leakagefixed-reeval-20261117",
  "metrics_only": true,
  "evaluation_only": true,
  "r2_sources": [{
    "bucket": "private-mecka",
    "endpoint_env": "R2_ENDPOINT_URL",
    "ready_prefix": "private_opscale/_training_artifacts/fold-clothes-moving-opscale-tracking-valid-20261103/lambda-async-20261107/checkpoints/ready",
    "object_prefix": "private_opscale/_training_artifacts/fold-clothes-moving-opscale-tracking-valid-20261103/lambda-async-20261107/objects/sha256"
  }],
  "cells": {
    "b24_op4": {
      "source_wandb_id": "SOURCE_ID",
      "norm_stats_dir": "/local/norms/b24_op4"
    }
  }
}
```

`evaluation_campaign` is used as both the common W&B tag and group (with
underscores in the group name), so the 36 evaluations can be selected exactly.
Set `metrics_only` to `true` for score-only reevaluations; this clears all pinned
video episodes and requires zero rendered videos while preserving every metric
and coverage check. Set `evaluation_only` to `true` for a final-checkpoint
campaign: preparation no longer requires training resume checkpoints or stop
receipts, writes no training launchers, and gives each campaign/cell pair one
deterministic W&B run ID shared across clusters.

The abbreviated example shows one entry; provide all36. Add each cell’s `source_stop_receipt` from the required stop-proof section before training. The complete36-cell template `INPUTS.example.json` includes this field. From the activated environment and repository root:

```bash
python -m experiments.fold_clothes_moving_opscale.lambda_async.runtime prepare --inputs /local/private_inputs.json
```

This writes `manifest.json`,36`train.yaml`,36`eval_reference.yaml`, and72executable scripts. It allocates no GPU and starts no worker. Evaluation reference YAML deliberately contains missing checkpoint-clock placeholders; the worker supplies verified values at execution. `inspect --manifest ... --cell b24_op4` prints the training configuration without instantiating datasets/models or contacting W&B.

For an evaluation-only manifest, preparation instead writes 36
`eval_reference.yaml` files and 36 `evaluate_final.sh` scripts. Each final
script reads only that cell's immutable step-400000 READY record, downloads and
SHA-verifies only that checkpoint, validates epoch 4000, evaluates all three
cohorts, writes the normal durable receipt, and exits. It is safe to statically
shard those 36 scripts across clusters because W&B identity depends only on the
campaign and cell, not the cluster-local run root.

## Import available source checkpoints before starting workers

Download each checkpoint using the export READY object's `object_key`. Then:

```bash
python -m experiments.fold_clothes_moving_opscale.lambda_async.runtime import-ready \
  --manifest /local/runs/opscale_lambda_async_v1/manifest.json --cell b24_op4 \
  --record /local/ready/b24_op4-source-record.json \
  --checkpoint /local/checkpoints/b24_op4/downloaded.ckpt
```

This computes the full local SHA and checks full training state before publishing a local READY record. Import **all source checkpoints you want rescored before training/evaluation completion is declared**, not only the resume checkpoint. Distinct SHA variants at the same update are each evaluated; deduplicate byte-identical exported objects by SHA.

## Launch on Lambda, explicitly assigning GPUs

Each script uses exactly one visible GPU. Assign separate device slots to training and evaluation; do not start multiple scoring subprocesses or trainers on the same GPU at once.

```bash
CUDA_VISIBLE_DEVICES=0 /local/runs/opscale_lambda_async_v1/b24_op4/train.sh
CUDA_VISIBLE_DEVICES=1 /local/runs/opscale_lambda_async_v1/b24_op4/worker.sh
```

Repeat for the other35cells using the destination scheduler/GPU allocation plan. There are36logical persistent evaluation workers; that does not require running36GPU evaluations concurrently. Start only the workers covered by available evaluation devices, or schedule them in groups. The worker persists between checkpoints and launches an isolated scoring subprocess for each one, so model/dataloader failures do not contaminate the next attempt. A stopped worker can restart and resume its durable queue.

## Correct checkpoint identity and validation clock

The old standalone `mode=validate` loads model weights but leaves Lightning's training clock at zero and disables W&B. This addition preserves W&B only for the explicitly opted-in external evaluation path, and uses an evaluation-only Trainer whose public `current_epoch`/`global_step` come from the verified checkpoint. Thus step220000 is labeled epoch2199/2200completedepochs in metrics, captions, video cadence and receipts. It never edits Lightning optimizer/fit-loop counters to simulate a resume.

Every completed evaluation receipt contains checkpoint SHA, split SHA, source/training/evaluation W&B IDs, exact epoch/update, coverage, metrics and video paths. The worker accepts completion only after all3cohorts and all required videos succeed and `wandb.finish()` returns. A separate `training_complete.json` records final full-state training completion; `evaluation_complete.json` is written only after every queued checkpoint has an accepted receipt.

## Monitoring and recovery

Per-cell artifacts:

- `queue/worker_status.json`: waiting/evaluating/blocked/stopped plus pending counts.
- `queue/ready/*.json`: immutable, verified full-state checkpoint publications.
- `train/checkpoints/`: all saved epoch checkpoints; never apply a rolling-two-checkpoint cleanup.
- `eval/checkpoints/<step>-<SHA>/evaluation-attempt-*.log` and `attempts.json`: errors and retry history.
- `eval/checkpoints/<step>-<SHA>/evaluation_receipt.json`: accepted score, coverage, videos and input identity.
- `queue/training_complete.json` and `queue/evaluation_complete.json`: separate final receipts.

A per-cell file lock prevents two training writers and another prevents duplicate evaluation workers. Training restart verifies and resumes the newest published local checkpoint; a checkpoint saved immediately before a publication crash is verified and recovered first. Evaluation is retried up to3times, then stops visibly for repair rather than skipping a checkpoint. After fixing the cause, inspect and archive the failed `attempts.json` before restarting that worker. Never remove a successful receipt to make an error disappear.

Successful local receipts make queue restarts idempotent. W&B is not a transactional store: a crash after W&B finishes but before the local receipt is committed may repeat the same checkpoint's log on retry. Group/deduplicate by checkpoint global step and SHA; do not claim exactly-once remote logging.

Each full checkpoint is several GB. Saving all remaining epochs across36cells requires **multiple TB**; size the local volume and R2 upload plan from actual source checkpoint sizes. Ready records are produced only after the checkpoint save and full SHA complete. Reserve sufficient disk and monitor free space; no automatic checkpoint deletion is performed.

## Validation status of this implementation

All72role configurations were composed and fully resolved against the corrected private split metadata and original normalization identities. Training configurations have zero validation datasets and limit0; evaluation configurations retain all3cohorts and limit800. Source syntax was inspected with Python AST parsing. No tests, dataset/model instantiations, or GPU training/evaluation were run on Pace for this migration. Before production use, the destination agent must perform the user's requested runtime verification on Lambda and confirm: config composition, full-state resume counters/LR, local data reads, exact3cohort coverage, W&B checkpoint-axis labels, expected videos, checkpoint READY publication and queue recovery. Do not describe this source-only preparation as a completed GPU verification.

## R2 READY polling and complete-history barrier

Optional `r2_sources` in the preparation input lets workers wait for future R2 READY publications. The worker uses `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` when both are present (the existing `setup_secrets.sh` format), otherwise boto3's normal environment/profile chain. A partially configured R2 pair fails visibly. It uses region `auto` and request/response checksums only when required for R2 compatibility. The endpoint variable contains only the HTTPS endpoint URL.

```json
"r2_sources": [{
  "bucket": "private-mecka",
  "endpoint_env": "R2_ENDPOINT_URL",
  "ready_prefix": "private_opscale/_training_artifacts/fold-clothes-moving-opscale-tracking-valid-20261103/pace-export-20261107/checkpoints/ready",
  "object_prefix": "private_opscale/_training_artifacts/fold-clothes-moving-opscale-tracking-valid-20261103/pace-export-20261107/objects/sha256",
  "complete_key": "private_opscale/_training_artifacts/fold-clothes-moving-opscale-tracking-valid-20261103/pace-export-20261107/COMPLETE.json"
}]
```

Each cell's worker lists only its private READY prefix, downloads the declared object, verifies full SHA/size/clock locally, and imports it into the same durable queue. R2 polling is read-only. This package does not upload Lambda checkpoints; use the destination's artifact uploader to publish verified objects/READY records if evaluation runs on another server.

The Pace exporter publishes `COMPLETE.json` only after all31Pace jobs stop, final capture is stable and every artifact upload commits. It enumerates exact READY keys and checkpoint SHAs per cell. Workers cannot mark source history complete until every enumerated checkpoint has been imported. The5non-Pace cells are explicitly excluded by that marker and need their own source history or manual completion confirmation.

For manually downloaded source history, after importing every intended source checkpoint, explicitly seal that cell:

```bash
python -m experiments.fold_clothes_moving_opscale.lambda_async.runtime seal-imports \
  --manifest /local/runs/opscale_lambda_async_v1/manifest.json --cell b24_op4
```

Without an applicable remote completion marker or this explicit local seal, a worker keeps waiting even after final training evaluation. Distinct checkpoint SHAs at the same step are allowed and each is scored: named and `last.ckpt` files can serialize differently. Remote history never overrides the manifest's chosen training resume; only checkpoints subsequently produced in that trainer's own output directory may supersede it after a restart.

### Share a limited evaluation GPU pool

All36logical workers may run concurrently with fewer evaluation GPUs by assigning each worker a visible device and the **same local lock path for that device**. Training must use different device slots. Example for multiple workers assigned evaluation GPU1:

```bash
CUDA_VISIBLE_DEVICES=1 EGOVERSE_EVAL_GPU_LOCK=/local/locks/eval-gpu1.lock \
  /local/runs/opscale_lambda_async_v1/b24_op4/worker.sh
```

The worker acquires the GPU lock only while a scoring subprocess is active. Its per-cell queue lock remains held throughout. SIGTERM/SIGINT stops its own scoring process group gracefully, then kills that same owned group after180seconds if necessary. No other training or GPU process is signaled.

## Required stop proof before training migration

Evaluation may process historical checkpoints while upload continues. **Training cannot start while the source writer is still active.** Each private input cell may supply:

```json
"source_stop_receipt": {
  "local_path": "/local/source-stop/b24_op4.json",
  "sha256": "SHA256_OF_THIS_RECEIPT_FILE"
}
```

The source stop receipt must contain:

```json
{
  "schema_version": 1,
  "cell": "b24_op4",
  "source_wandb_id": "SOURCE_ID",
  "stopped": true,
  "source_checkpoint_sha256": "EXACT_SELECTED_RESUME_CHECKPOINT_SHA256",
  "source_resume_cutoff": 220000
}
```

The destination agent must create this receipt only after checking the source job/process is stopped and its final selected checkpoint is verified. For Pace, require the stop monitor receipt and completed export; for the5existing Lambda runs, quiesce those writers too. The220000 cutoff is an example: use each selected checkpoint's actual verified update count. `prepare` can run without this receipt so the evaluation queue can be prepared early; `train` refuses to start without it. If preparation occurred early, finish the private manifest's `source_stop_receipt` fields before launch. Never change scientific memberships or checkpoint identity to bypass this gate.

New W&B filters:

- All72new role runs: `fold-moving-opscale-async-20261107`
- Exactly36new training runs: `fold-moving-opscale-async-train-20261107`
- Exactly36new evaluation runs: `fold-moving-opscale-async-eval-20261107`

Source runs and their original tags are untouched. The new evaluation histories use the corrected unseen cohort; do not mix them with original unseen scores as if the cohort were unchanged.
