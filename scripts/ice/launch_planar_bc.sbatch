#!/bin/bash
#SBATCH --job-name=egoverse-planar-bc
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100|H200
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

set -Eeuo pipefail
umask 077

die() {
  printf 'launch_planar_bc: %s\n' "$*" >&2
  exit 64
}

required() {
  local name=$1
  test -n "${!name:-}" || die "set $name"
}

absolute_path() {
  local label=$1 value=$2
  [[ "$value" = /* ]] || die "$label must be an absolute path"
  test "$value" != / || die "$label must be a specific path"
}

positive_integer() {
  local label=$1 value=$2
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label must be a positive integer"
}

positive_number() {
  local label=$1 value=$2
  "$ICE_PYTHON" - "$label" "$value" <<'PY'
import math
import sys

label, raw = sys.argv[1:]
try:
    value = float(raw)
except ValueError as exc:
    raise SystemExit(f"{label} must be numeric") from exc
if not math.isfinite(value) or value <= 0:
    raise SystemExit(f"{label} must be finite and positive")
PY
}

for variable in \
  ICE_REPO ICE_EXPECTED_HEAD ICE_PYTHON ICE_DATASET_DIR ICE_OUTPUT_DIR \
  ICE_EXPERIMENT ICE_WANDB_ENTITY ICE_WANDB_PROJECT ICE_WANDB_RUN_ID \
  ICE_CHECKPOINT_VALIDATOR ICE_EXPECTED_SPLIT_MANIFEST_SHA256 \
  ICE_MAX_STEPS \
  ICE_VAL_CHECK_INTERVAL ICE_LIMIT_TRAIN_BATCHES ICE_LIMIT_VAL_BATCHES \
  ICE_CHECKPOINT_EVERY_N_STEPS ICE_TRAIN_BATCH_SIZE ICE_VALID_BATCH_SIZE; do
  required "$variable"
done

ICE_LAUNCH_MODE=${ICE_LAUNCH_MODE:-preflight}
ICE_EXPECTED_ACCOUNT=${ICE_EXPECTED_ACCOUNT:-coc}
ICE_EXPECTED_PARTITION=${ICE_EXPECTED_PARTITION:-ice-gpu}
ICE_EXPECTED_QOS=${ICE_EXPECTED_QOS:-coc-ice}
ICE_MAX_RESTARTS=${ICE_MAX_RESTARTS:-32}
ICE_WANDB_NAME=${ICE_WANDB_NAME:-$ICE_WANDB_RUN_ID}
ICE_WANDB_GROUP=${ICE_WANDB_GROUP:-planar-bc-ice}
ICE_WANDB_TAGS=${ICE_WANDB_TAGS:-'[pushshapes,planar-v2,bc,ice]'}
ICE_TOOL_DIR=${ICE_TOOL_DIR:-$ICE_REPO/scripts/ice}
ICE_REQUEUE_RUNNER=${ICE_REQUEUE_RUNNER:-$ICE_TOOL_DIR/ice_requeue_runner.py}
ICE_GPU_PROBE=${ICE_GPU_PROBE:-$ICE_TOOL_DIR/ice_gpu_probe.py}
ICE_DATASET_VALIDATOR=${ICE_DATASET_VALIDATOR:-$ICE_TOOL_DIR/validate_planar_dataset.py}
ICE_SPLIT_MANIFEST=${ICE_SPLIT_MANIFEST:-$ICE_REPO/egomimic/hydra_configs/data/pusht/planar_v2_usocket_dp_3k_split_seed42_v1.json}
ICE_PRECISION=${ICE_PRECISION:-bf16-mixed}
ICE_SEED=${ICE_SEED:-42}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
export PATH="$(dirname "$ICE_PYTHON"):$PATH"
export ICE_REQUIRE_LR_SCHEDULERS=1

case "$ICE_LAUNCH_MODE" in
  preflight|dry-run|run) ;;
  *) die "ICE_LAUNCH_MODE must be preflight, dry-run, or run" ;;
esac
case "$ICE_EXPERIMENT" in
  pusht/planar_v2_usocket_direct_bc|pusht/planar_v2_usocket_arc_bc|\
  pusht/planar_v2_usocket_arc_hybrid_D40_M16_R24deg_bc) ;;
  *) die "ICE_EXPERIMENT must select a supported U-Socket BC config" ;;
esac
test "$ICE_PRECISION" = bf16-mixed || die "ICE_PRECISION must be bf16-mixed"
[[ "$ICE_EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || die "ICE_EXPECTED_HEAD must be a full commit"
[[ "$ICE_EXPECTED_SPLIT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "ICE_EXPECTED_SPLIT_MANIFEST_SHA256 must be a lowercase SHA-256"
[[ "$ICE_WANDB_RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe ICE_WANDB_RUN_ID"
positive_integer ICE_MAX_STEPS "$ICE_MAX_STEPS"
positive_integer ICE_VAL_CHECK_INTERVAL "$ICE_VAL_CHECK_INTERVAL"
positive_number ICE_LIMIT_TRAIN_BATCHES "$ICE_LIMIT_TRAIN_BATCHES"
positive_number ICE_LIMIT_VAL_BATCHES "$ICE_LIMIT_VAL_BATCHES"
positive_integer ICE_CHECKPOINT_EVERY_N_STEPS "$ICE_CHECKPOINT_EVERY_N_STEPS"
positive_integer ICE_TRAIN_BATCH_SIZE "$ICE_TRAIN_BATCH_SIZE"
positive_integer ICE_VALID_BATCH_SIZE "$ICE_VALID_BATCH_SIZE"
positive_integer ICE_MAX_RESTARTS "$ICE_MAX_RESTARTS"
positive_integer ICE_SEED "$ICE_SEED"
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] || die "invalid SLURM_RESTART_COUNT"
test "$RESTART_COUNT" -le "$ICE_MAX_RESTARTS" || die "restart limit exceeded"

for binding in \
  "ICE_REPO:$ICE_REPO" "ICE_PYTHON:$ICE_PYTHON" \
  "ICE_DATASET_DIR:$ICE_DATASET_DIR" "ICE_OUTPUT_DIR:$ICE_OUTPUT_DIR" \
  "ICE_TOOL_DIR:$ICE_TOOL_DIR" \
  "ICE_REQUEUE_RUNNER:$ICE_REQUEUE_RUNNER" \
  "ICE_GPU_PROBE:$ICE_GPU_PROBE" \
  "ICE_DATASET_VALIDATOR:$ICE_DATASET_VALIDATOR" \
  "ICE_SPLIT_MANIFEST:$ICE_SPLIT_MANIFEST" \
  "ICE_CHECKPOINT_VALIDATOR:$ICE_CHECKPOINT_VALIDATOR"; do
  absolute_path "${binding%%:*}" "${binding#*:}"
done
test -d "$ICE_REPO" || die "ICE_REPO is not a directory"
test -x "$ICE_PYTHON" || die "ICE_PYTHON is not executable"
test -d "$ICE_DATASET_DIR" || die "ICE_DATASET_DIR is not a directory"
test -d "$ICE_TOOL_DIR" || die "ICE_TOOL_DIR is not a directory"
test -s "$ICE_REQUEUE_RUNNER" || die "generic requeue runner is unavailable"
test -s "$ICE_GPU_PROBE" || die "GPU probe is unavailable"
test -s "$ICE_DATASET_VALIDATOR" || die "dataset validator is unavailable"
test -s "$ICE_SPLIT_MANIFEST" || die "split manifest is unavailable"
test -x "$ICE_CHECKPOINT_VALIDATOR" || die "checkpoint validator must be executable"
test -s "$ICE_REPO/egomimic/hydra_configs/experiment/$ICE_EXPERIMENT.yaml" || \
  die "selected BC experiment config is unavailable"
test "$(git -C "$ICE_REPO" rev-parse HEAD)" = "$ICE_EXPECTED_HEAD" || \
  die "source HEAD does not match ICE_EXPECTED_HEAD"
test -z "$(git -C "$ICE_REPO" status --porcelain=v1 --untracked-files=all)" || \
  die "source repository must be exactly clean"
# The parent runner launches the checkpoint validator directly. Bind it to the
# same exact source tree as the training child so strict checkpoint reloads do
# not depend on the submission shell's ambient Python path.
export PYTHONPATH=$ICE_REPO

BASE_OVERRIDES=(
  --config-name=train_zarr_cartesian
  "+experiment=$ICE_EXPERIMENT"
  mode=train
  "hydra.run.dir=$ICE_OUTPUT_DIR"
  "++paths.root_dir=$ICE_REPO"
  "paths.output_dir=$ICE_OUTPUT_DIR"
  "paths.work_dir=$ICE_REPO"
  launch_params.gpus_per_node=1
  launch_params.nodes=1
  trainer.accelerator=gpu
  trainer.devices=1
  trainer.num_nodes=1
  trainer.strategy=auto
  trainer.sync_batchnorm=false
  "trainer.precision=$ICE_PRECISION"
  trainer.accumulate_grad_batches=1
  "trainer.max_steps=$ICE_MAX_STEPS"
  trainer.max_epochs=-1
  trainer.min_epochs=null
  "trainer.val_check_interval=$ICE_VAL_CHECK_INTERVAL"
  trainer.check_val_every_n_epoch=null
  "trainer.limit_train_batches=$ICE_LIMIT_TRAIN_BATCHES"
  "trainer.limit_val_batches=$ICE_LIMIT_VAL_BATCHES"
  trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=1
  +callbacks.model_checkpoint.monitor=null
  callbacks.model_checkpoint.save_top_k=-1
  callbacks.model_checkpoint.save_last=true
  callbacks.model_checkpoint.every_n_epochs=null
  "+callbacks.model_checkpoint.every_n_train_steps=$ICE_CHECKPOINT_EVERY_N_STEPS"
  +callbacks.model_checkpoint.train_time_interval=null
  +callbacks.model_checkpoint.save_on_train_epoch_end=false
  "callbacks.model_checkpoint.filename='epoch-{epoch}-step-{step}'"
  runtime.slurm_requeue_owner=runner
  runtime.slurm_save_signal=SIGUSR2
  'runtime.slurm_signal_checkpoint_dir=${paths.output_dir}/checkpoints'
  "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$ICE_DATASET_DIR"
  "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$ICE_DATASET_DIR"
  "data.train_dataloader_params.pushshapes_sim_u_socket.batch_size=$ICE_TRAIN_BATCH_SIZE"
  "data.valid_dataloader_params.pushshapes_sim_u_socket.batch_size=$ICE_VALID_BATCH_SIZE"
  logger.wandb.offline=false
  "logger.wandb.entity=$ICE_WANDB_ENTITY"
  "logger.wandb.project=$ICE_WANDB_PROJECT"
  "logger.wandb.group=$ICE_WANDB_GROUP"
  "logger.wandb.id=$ICE_WANDB_RUN_ID"
  "+logger.wandb.name=$ICE_WANDB_NAME"
  "logger.wandb.tags=$ICE_WANDB_TAGS"
  "seed=$ICE_SEED"
)
if test -n "${ICE_NORM_STATS_PATH:-}"; then
  absolute_path ICE_NORM_STATS_PATH "$ICE_NORM_STATS_PATH"
  test -e "$ICE_NORM_STATS_PATH" || die "ICE_NORM_STATS_PATH is unavailable"
  if test -d "$ICE_NORM_STATS_PATH"; then
    EFFECTIVE_NORM_STATS_FILE=$ICE_NORM_STATS_PATH/norm_stats.json
  else
    EFFECTIVE_NORM_STATS_FILE=$ICE_NORM_STATS_PATH
  fi
  test -s "$EFFECTIVE_NORM_STATS_FILE" || die "precomputed norm_stats.json is unavailable"
  BASE_OVERRIDES+=("norm_stats.precomputed_norm_path=$ICE_NORM_STATS_PATH" norm_stats.save_cache_dir=null)
elif test "$RESTART_COUNT" -gt 0; then
  EFFECTIVE_NORM_STATS_FILE=$ICE_OUTPUT_DIR/norm_stats/norm_stats.json
  test -s "$EFFECTIVE_NORM_STATS_FILE" || \
    die "requeue requires the original run's norm_stats/norm_stats.json"
  BASE_OVERRIDES+=("norm_stats.precomputed_norm_path=$EFFECTIVE_NORM_STATS_FILE" norm_stats.save_cache_dir=null)
else
  EFFECTIVE_NORM_STATS_FILE=
  BASE_OVERRIDES+=(norm_stats.precomputed_norm_path=null "norm_stats.save_cache_dir=$ICE_OUTPUT_DIR")
fi

PREFLIGHT_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/egoverse-planar-bc-preflight.XXXXXX")
cleanup_preflight() {
  rm -rf -- "$PREFLIGHT_ROOT"
}
trap cleanup_preflight EXIT
PREFLIGHT_CONFIG=$PREFLIGHT_ROOT/resolved_config.yaml
PREFLIGHT_DATASET=$PREFLIGHT_ROOT/dataset_validation.json
PYTHONPATH=$ICE_REPO "$ICE_PYTHON" "$ICE_DATASET_VALIDATOR" \
  --repo "$ICE_REPO" \
  --dataset-root "$ICE_DATASET_DIR" \
  --manifest "$ICE_SPLIT_MANIFEST" \
  --expected-manifest-sha256 "$ICE_EXPECTED_SPLIT_MANIFEST_SHA256" \
  --output "$PREFLIGHT_DATASET" >/dev/null
test -s "$PREFLIGHT_DATASET" || die "dataset validation produced no artifact"
(
  cd "$ICE_REPO"
  PYTHONPATH=$ICE_REPO HYDRA_FULL_ERROR=1 "$ICE_PYTHON" -m egomimic.trainHydra \
    "${BASE_OVERRIDES[@]}" ckpt_path=null +logger.wandb.resume=never \
    --cfg job --resolve > "$PREFLIGHT_CONFIG"
)
test -s "$PREFLIGHT_CONFIG" || die "Hydra preflight produced no resolved config"
"$ICE_PYTHON" - "$PREFLIGHT_CONFIG" "$ICE_EXPERIMENT" \
  "$ICE_MAX_STEPS" "$ICE_VAL_CHECK_INTERVAL" "$ICE_LIMIT_TRAIN_BATCHES" \
  "$ICE_LIMIT_VAL_BATCHES" "$ICE_CHECKPOINT_EVERY_N_STEPS" \
  "$ICE_EXPECTED_SPLIT_MANIFEST_SHA256" <<'PY'
import math
import pathlib
import sys

from omegaconf import OmegaConf

(
    path,
    experiment,
    max_steps,
    val_interval,
    train_limit,
    val_limit,
    checkpoint_interval,
    split_manifest_sha256,
) = sys.argv[1:]
cfg = OmegaConf.load(pathlib.Path(path))
assert cfg.mode == "train"
assert cfg.trainer.accelerator == "gpu"
assert int(cfg.trainer.devices) == int(cfg.trainer.num_nodes) == 1
assert cfg.trainer.strategy == "auto"
assert cfg.trainer.sync_batchnorm is False
assert str(cfg.trainer.precision) == "bf16-mixed"
assert int(cfg.trainer.max_steps) == int(max_steps)
assert int(cfg.trainer.val_check_interval) == int(val_interval)
assert math.isclose(float(cfg.trainer.limit_train_batches), float(train_limit))
assert math.isclose(float(cfg.trainer.limit_val_batches), float(val_limit))
assert int(cfg.callbacks.model_checkpoint.every_n_train_steps) == int(
    checkpoint_interval
)
assert int(cfg.callbacks.model_checkpoint.save_top_k) == -1
assert cfg.callbacks.model_checkpoint.save_last is True
assert cfg.runtime.slurm_requeue_owner == "runner"
assert cfg.runtime.slurm_save_signal == "SIGUSR2"
assert cfg.paths.root_dir == cfg.paths.work_dir
if cfg.norm_stats.precomputed_norm_path is None:
    assert cfg.norm_stats.save_cache_dir == cfg.paths.output_dir
else:
    assert cfg.norm_stats.save_cache_dir is None
assert cfg.run_provenance.split_seed == 42
assert cfg.run_provenance.valid_ratio == 0.01
assert cfg.run_provenance.split_manifest_sha256 == split_manifest_sha256
assert cfg.name.startswith("planar_v2_usocket_")
assert experiment in {
    "pusht/planar_v2_usocket_direct_bc",
    "pusht/planar_v2_usocket_arc_bc",
    "pusht/planar_v2_usocket_arc_hybrid_D40_M16_R24deg_bc",
}
PY

"$ICE_PYTHON" "$ICE_REQUEUE_RUNNER" \
  --state-dir "$PREFLIGHT_ROOT/runner/state" \
  --checkpoint-glob "$PREFLIGHT_ROOT/checkpoints/*.ckpt" \
  --checkpoint-validator "$ICE_CHECKPOINT_VALIDATOR" \
  --checkpoint-signal USR2 \
  --checkpoint-forwarding slurm-steps \
  --max-restarts "$ICE_MAX_RESTARTS" \
  --requeue-owner runner \
  --confirm-child-requeue-disabled \
  --completion-sentinel "$PREFLIGHT_ROOT/run/COMPLETE.json" \
  --dry-run -- /usr/bin/env true >/dev/null

if test "$ICE_LAUNCH_MODE" != run; then
  "$ICE_PYTHON" - "$ICE_REPO" "$ICE_EXPECTED_HEAD" "$ICE_EXPERIMENT" \
    "$ICE_DATASET_DIR" "$ICE_OUTPUT_DIR" "$ICE_WANDB_ENTITY" \
    "$ICE_WANDB_PROJECT" "$ICE_WANDB_RUN_ID" \
    "$ICE_EXPECTED_SPLIT_MANIFEST_SHA256" <<'PY'
import json
import sys

repo, head, experiment, dataset, output, entity, project, run_id, split_sha = sys.argv[1:]
print(json.dumps({
    "status": "PREFLIGHT_READY",
    "source": {"repo": repo, "head": head},
    "experiment": experiment,
    "dataset": dataset,
    "split_manifest_sha256": split_sha,
    "output": output,
    "wandb": {"entity": entity, "project": project, "run_id": run_id},
    "world_size": 1,
    "strategy": "auto",
    "precision": "bf16-mixed",
    "requeue_owner": "runner",
}, sort_keys=True))
PY
  exit 0
fi

for variable in \
  SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION \
  SLURM_JOB_QOS SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK; do
  required "$variable"
done
test "$SLURM_JOB_ACCOUNT" = "$ICE_EXPECTED_ACCOUNT" || die "unexpected Slurm account"
test "$SLURM_JOB_PARTITION" = "$ICE_EXPECTED_PARTITION" || die "unexpected Slurm partition"
test "$SLURM_JOB_QOS" = "$ICE_EXPECTED_QOS" || die "unexpected Slurm QoS"
test "$SLURM_JOB_NUM_NODES" = 1 || die "launcher requires one node"
test "$SLURM_NTASKS" = 1 || die "launcher requires world size one"
positive_integer SLURM_CPUS_PER_TASK "$SLURM_CPUS_PER_TASK"

if test "$RESTART_COUNT" = 0; then
  test ! -e "$ICE_OUTPUT_DIR" || die "unique ICE_OUTPUT_DIR already exists"
else
  test -d "$ICE_OUTPUT_DIR" || die "requeued output directory is unavailable"
fi
mkdir -p "$ICE_OUTPUT_DIR/checkpoints" "$ICE_OUTPUT_DIR/provenance" \
  "$ICE_OUTPUT_DIR/runner-state"
ATTEMPT_PROVENANCE=$ICE_OUTPUT_DIR/provenance/restart-$RESTART_COUNT
test ! -e "$ATTEMPT_PROVENANCE" || die "attempt provenance already exists"
mkdir -p "$ATTEMPT_PROVENANCE"
cp -- "$PREFLIGHT_CONFIG" "$ATTEMPT_PROVENANCE/resolved_config.yaml"
cp -- "$PREFLIGHT_DATASET" "$ATTEMPT_PROVENANCE/dataset_validation.json"
cp -- "$ICE_SPLIT_MANIFEST" "$ATTEMPT_PROVENANCE/split_manifest.json"
sha256sum "$ATTEMPT_PROVENANCE/resolved_config.yaml" \
  > "$ATTEMPT_PROVENANCE/resolved_config.sha256"
sha256sum "$ATTEMPT_PROVENANCE/dataset_validation.json" \
  > "$ATTEMPT_PROVENANCE/dataset_validation.sha256"
test "$(sha256sum "$ATTEMPT_PROVENANCE/split_manifest.json" | awk '{print $1}')" = \
  "$ICE_EXPECTED_SPLIT_MANIFEST_SHA256" || die "copied split manifest changed"
if test -n "$EFFECTIVE_NORM_STATS_FILE"; then
  sha256sum "$EFFECTIVE_NORM_STATS_FILE" \
    > "$ATTEMPT_PROVENANCE/normalization.sha256"
fi
git -C "$ICE_REPO" status --short --branch \
  > "$ATTEMPT_PROVENANCE/source_status.txt"
nvidia-smi -q > "$ATTEMPT_PROVENANCE/nvidia_smi_q.txt"
scontrol show job -dd "$SLURM_JOB_ID" \
  > "$ATTEMPT_PROVENANCE/slurm_job.txt"

ICE_SRUN=$(command -v srun)
test -x "$ICE_SRUN" || die "srun is unavailable"
export ICE_REPO ICE_PYTHON ICE_EXPERIMENT ICE_OUTPUT_DIR ICE_DATASET_DIR
export ICE_WANDB_ENTITY ICE_WANDB_PROJECT ICE_WANDB_RUN_ID ICE_WANDB_NAME
export ICE_WANDB_GROUP ICE_WANDB_TAGS ICE_SRUN

"$ICE_SRUN" --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 --unbuffered \
  "$ICE_PYTHON" "$ICE_GPU_PROBE" \
  --expected-world-size 1 \
  --allowed-gpu-name "NVIDIA H100 80GB HBM3" \
  --allowed-gpu-name "NVIDIA H200" \
  --output "$ATTEMPT_PROVENANCE/gpu_probe.json"

read -r -d '' CHILD_CODE <<'BASH' || true
set -Eeuo pipefail
checkpoint_override=ckpt_path=null
wandb_resume=never
if test -n "${ICE_RESUME_CHECKPOINT:-}"; then
  "$ICE_PYTHON" - "$ICE_RESUME_CHECKPOINT" \
    "${ICE_RESUME_CHECKPOINT_SHA256:?}" "${ICE_RESUME_GLOBAL_STEP:?}" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
expected_sha, raw_step = sys.argv[2:]
step = int(raw_step)
assert step >= 0
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == expected_sha
PY
  checkpoint_override="ckpt_path=$ICE_RESUME_CHECKPOINT"
  wandb_resume=must
else
  test -z "${ICE_RESUME_CHECKPOINT_SHA256:-}${ICE_RESUME_GLOBAL_STEP:-}"
fi
cd "$ICE_REPO"
export PYTHONPATH=$ICE_REPO HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export WANDB_MODE=online WANDB_SILENT=true MUJOCO_GL=egl
exec "$ICE_SRUN" --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --kill-on-bad-exit=1 --unbuffered \
  "$ICE_PYTHON" -m egomimic.trainHydra "$@" \
  "$checkpoint_override" "+logger.wandb.resume=$wandb_resume"
BASH

RUNNER_ARGS=(
  --state-dir "$ICE_OUTPUT_DIR/runner-state"
  --checkpoint-glob "$ICE_OUTPUT_DIR/checkpoints/*.ckpt"
  --checkpoint-validator "$ICE_CHECKPOINT_VALIDATOR"
  --checkpoint-signal USR2
  --checkpoint-forwarding slurm-steps
  --checkpoint-grace-seconds 480
  --max-restarts "$ICE_MAX_RESTARTS"
  --requeue-owner runner
  --confirm-child-requeue-disabled
  --completion-sentinel "$ICE_OUTPUT_DIR/COMPLETE.json"
)
exec "$ICE_PYTHON" "$ICE_REQUEUE_RUNNER" "${RUNNER_ARGS[@]}" -- \
  /usr/bin/env bash -c "$CHILD_CODE" ice-planar-bc "${BASE_OVERRIDES[@]}"
