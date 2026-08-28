#!/bin/bash
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

ARCHITECTURE=${1:?usage: image_generation_matched.sbatch ARCHITECTURE smoke|full}
MODE=${2:?usage: image_generation_matched.sbatch ARCHITECTURE smoke|full}
case "$ARCHITECTURE" in
  jit_b16|endpoint_latent|jit_endpoint) ;;
  *) echo "unsupported architecture: $ARCHITECTURE" >&2; exit 2 ;;
esac
case "$MODE" in
  smoke|full) ;;
  *) echo "unsupported mode: $MODE" >&2; exit 2 ;;
esac

SOURCE_ROOT=${SOURCE_ROOT:?submitter must export immutable SOURCE_ROOT}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:?submitter must export EXPERIMENT_ROOT}
EXPECTED_SHA=${EXPECTED_SHA:?submitter must export EXPECTED_SHA}
DATA_ROOT=${DATA_ROOT:-/coc/dataset/ImageNet/imagenet}
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv
SLURM_BIN=/opt/slurm/Ubuntu-20.04/current/bin
JIT_ROOT="$SOURCE_ROOT/external/JiT"

test -x "$PY_ENV/bin/python"
test -x "$PY_ENV/bin/torchrun"
test -d "$DATA_ROOT/train"
test -d "$DATA_ROOT/val"
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)"
test "${SLURM_JOB_PARTITION:?}" = hoffman-lab
test "${SLURM_JOB_ACCOUNT:?}" = hoffman-lab
test "${SLURM_GPUS_ON_NODE:?}" = 4

mkdir -p "$EXPERIMENT_ROOT/slurm" "$EXPERIMENT_ROOT/provenance"
PROVENANCE_DIR="$EXPERIMENT_ROOT/provenance/$MODE-$ARCHITECTURE-job${SLURM_JOB_ID}"
mkdir -p "$PROVENANCE_DIR"
cp "$0" "$PROVENANCE_DIR/launcher.sbatch"
cp "$SOURCE_ROOT/scripts/train/verify_image_generation_smoke.py" "$PROVENANCE_DIR/"
git -C "$SOURCE_ROOT" log -1 --format=fuller > "$PROVENANCE_DIR/git_commit.txt"
git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all > "$PROVENANCE_DIR/git_status.txt"
"$SLURM_BIN/scontrol" show job -dd "$SLURM_JOB_ID" > "$PROVENANCE_DIR/slurm_job.txt"
nvidia-smi -L > "$PROVENANCE_DIR/gpus.txt"
cp "$EXPERIMENT_ROOT/provenance/dataset_inventory.txt" "$PROVENANCE_DIR/"
cp "$EXPERIMENT_ROOT/provenance/dataset_inventory.sha256" "$PROVENANCE_DIR/"
sha256sum "$PROVENANCE_DIR"/* > "$PROVENANCE_DIR/artifacts.sha256"

source "$PY_ENV/bin/activate"
export PYTHONPATH="$SOURCE_ROOT:$JIT_ROOT"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$EXPERIMENT_ROOT/torchinductor/$ARCHITECTURE"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
cd "$JIT_ROOT"

PARAM_ARGS=(--expected-min-params 120000000 --expected-max-params 140000000)
COMMON_ARGS=(
  --architecture "$ARCHITECTURE"
  --data-path "$DATA_ROOT"
  --image-size 256
  --num-classes 1000
  --seed 42
  --num-workers 8
  "${PARAM_ARGS[@]}"
)

if [[ "$ARCHITECTURE" == jit_b16 ]]; then
  FULL_BATCH_SIZE=64
  FULL_GRAD_ACCUM=4
elif [[ "$ARCHITECTURE" == jit_endpoint ]]; then
  FULL_BATCH_SIZE=8
  FULL_GRAD_ACCUM=32
else
  FULL_BATCH_SIZE=32
  FULL_GRAD_ACCUM=8
fi

REAL_OPT_ARGS=(
  --base-lr 5e-5
  --warmup-epochs 5
  --weight-decay 0
  --adam-beta1 0.9
  --adam-beta2 0.95
)
FULL_LIMIT_ARGS=()
if [[ "$ARCHITECTURE" == jit_endpoint ]]; then
  # Exact action-model optimization recipe.  In particular, do not apply the
  # image harness's effective-batch LR scaling to the iterative field.
  REAL_OPT_ARGS=(
    --base-lr 3e-5
    --lr-schedule action_warmup_cosine
    --min-lr 3e-6
    --warmup-steps 3000
    --warmup-start-factor 0.1
    --lr-total-steps 240000
    --weight-decay 1e-4
    --adam-beta1 0.9
    --adam-beta2 0.999
  )
  FULL_LIMIT_ARGS=(--max-optimizer-steps 240000)
fi

if [[ "$MODE" == smoke ]]; then
  SMOKE_ROOT="$EXPERIMENT_ROOT/smokes/$ARCHITECTURE/job_${SLURM_JOB_ID}"
  OVERFIT_DIR="$SMOKE_ROOT/overfit"
  REAL_DIR="$SMOKE_ROOT/real"
  mkdir -p "$OVERFIT_DIR" "$REAL_DIR"

  TORCHDYNAMO_DISABLE=1 "$PY_ENV/bin/python" -m pytest -q \
    "$JIT_ROOT/test_matched_models.py" \
    2>&1 | tee "$SMOKE_ROOT/unit-tests.log"

  "$PY_ENV/bin/torchrun" --standalone --nproc_per_node=4 train_matched.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$OVERFIT_DIR" \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 1 \
    --base-lr 1e-4 \
    --warmup-epochs 0 \
    --max-optimizer-steps 40 \
    --save-every-steps 40 \
    --val-every-steps 0 \
    --log-every-steps 5 \
    --overfit-one-batch \
    --overfit-force-steps 2 \
    --wandb-mode disabled \
    2>&1 | tee "$SMOKE_ROOT/overfit.log"

  "$PY_ENV/bin/torchrun" --standalone --nproc_per_node=4 train_matched.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$REAL_DIR" \
    --epochs 1 \
    --batch-size "$FULL_BATCH_SIZE" \
    --grad-accum "$FULL_GRAD_ACCUM" \
    "${REAL_OPT_ARGS[@]}" \
    --max-optimizer-steps 2 \
    --save-every-steps 2 \
    --val-every-steps 1 \
    --val-batches 1 \
    --sample-batch 1 \
    --sample-steps 16 \
    --log-every-steps 1 \
    --smoke-require-validation \
    --wandb-project image-generation-latent-denoise \
    --wandb-name "smoke-${ARCHITECTURE}-${EXPECTED_SHA:0:8}" \
    --wandb-mode offline \
    2>&1 | tee "$SMOKE_ROOT/real.log"

  "$PY_ENV/bin/torchrun" --standalone --nproc_per_node=4 train_matched.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$REAL_DIR" \
    --epochs 1 \
    --batch-size "$FULL_BATCH_SIZE" \
    --grad-accum "$FULL_GRAD_ACCUM" \
    "${REAL_OPT_ARGS[@]}" \
    --max-optimizer-steps 3 \
    --save-every-steps 3 \
    --val-every-steps 3 \
    --val-batches 1 \
    --sample-batch 1 \
    --sample-steps 16 \
    --log-every-steps 1 \
    --smoke-require-validation \
    --resume \
    --wandb-mode disabled \
    2>&1 | tee "$SMOKE_ROOT/resume.log"

  if [[ "$ARCHITECTURE" == jit_endpoint ]]; then
    STRESS_DIR="$SMOKE_ROOT/j8-memory"
    mkdir -p "$STRESS_DIR"
    "$PY_ENV/bin/torchrun" --standalone --nproc_per_node=4 train_matched.py \
      "${COMMON_ARGS[@]}" \
      --output-dir "$STRESS_DIR" \
      --epochs 1 \
      --batch-size "$FULL_BATCH_SIZE" \
      --grad-accum 1 \
      "${REAL_OPT_ARGS[@]}" \
      --max-optimizer-steps 1 \
      --train-force-steps 8 \
      --save-every-steps 1 \
      --val-every-steps 0 \
      --log-every-steps 1 \
      --wandb-mode disabled \
      2>&1 | tee "$SMOKE_ROOT/j8-memory.log"
    echo "JIT_ENDPOINT_J8_MEMORY_PASSED"
  fi

  VERIFY_ARGS=(
    --overfit-log "$SMOKE_ROOT/overfit.log"
    --smoke-log "$SMOKE_ROOT/real.log"
    --smoke-dir "$REAL_DIR"
    --expected-architecture "$ARCHITECTURE"
    --resume-log "$SMOKE_ROOT/resume.log"
  )
  if [[ "$ARCHITECTURE" == jit_endpoint ]]; then
    VERIFY_ARGS+=(--stress-log "$SMOKE_ROOT/j8-memory.log")
  fi
  "$PY_ENV/bin/python" "$SOURCE_ROOT/scripts/train/verify_image_generation_smoke.py" \
    "${VERIFY_ARGS[@]}"
  echo "MATCHED_SMOKE_COMPLETE architecture=$ARCHITECTURE job=$SLURM_JOB_ID"
  exit 0
fi

RUN_DIR="$EXPERIMENT_ROOT/runs/$ARCHITECTURE"
STOP_FILE="$RUN_DIR/stop-requested"
mkdir -p "$RUN_DIR"
rm -f "$STOP_FILE"

FULL_ARGS=(
  "${COMMON_ARGS[@]}"
  --output-dir "$RUN_DIR"
  --epochs 600
  --batch-size "$FULL_BATCH_SIZE"
  --grad-accum "$FULL_GRAD_ACCUM"
  "${REAL_OPT_ARGS[@]}"
  "${FULL_LIMIT_ARGS[@]}"
  --ema-decay 0.9999
  --save-every-steps 1000
  --val-every-steps 5000
  --val-batches 8
  --sample-batch 4
  --sample-steps 16
  --cfg-scale 1.0
  --log-every-steps 10
  --resume
  --wandb-project image-generation-latent-denoise
  --wandb-name "imagenet256-${ARCHITECTURE}-${EXPECTED_SHA:0:8}"
  --wandb-mode online
  --stop-file "$STOP_FILE"
)

REQUEUE_REQUESTED=0
handle_usr1() {
  REQUEUE_REQUESTED=1
  touch "$STOP_FILE"
  echo "SLURM_REQUEUE_CHECKPOINT_REQUESTED job=$SLURM_JOB_ID"
}
trap handle_usr1 USR1

set +e
"$PY_ENV/bin/torchrun" --standalone --nproc_per_node=4 train_matched.py \
  "${FULL_ARGS[@]}" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  wait "$TRAIN_PID"
  TRAIN_STATUS=$?
done
wait "$TRAIN_PID" 2>/dev/null
TRAIN_STATUS=${TRAIN_STATUS:-$?}
set -e

if [[ "$REQUEUE_REQUESTED" == 1 && "$TRAIN_STATUS" == 0 ]]; then
  echo "REQUEUEING_FROM_CHECKPOINT job=$SLURM_JOB_ID"
  "$SLURM_BIN/scontrol" requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$TRAIN_STATUS"
