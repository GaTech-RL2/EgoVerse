#!/bin/bash
# Submit one phase of an immutable Flow Transfer run bundle with pinned resources.

set -Eeuo pipefail

test "$#" = 2
MODE=$1
MANIFEST=$(realpath "$2")
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
PY_ENV=${PY_ENV:?set the pinned Python environment}
SLURM_BIN=${SLURM_BIN:-/opt/slurm/Ubuntu-20.04/24.11.0/bin}
PYTHON=$PY_ENV/bin/python
TOOL=$REPO/egomimic/scripts/flow_transfer_run_bundle.py
LAUNCHER=$REPO/scripts/train/flow_transfer_run_bundle.sbatch
SBATCH=$SLURM_BIN/sbatch

test -x "$PYTHON"
test -x "$SBATCH"
test -f "$TOOL"
test -f "$LAUNCHER"

case "$MODE" in
  norm|smoke|full) ;;
  *)
    printf 'usage: %s {norm|smoke|full} MANIFEST\n' "$0" >&2
    exit 2
    ;;
esac

field() {
  "$PYTHON" "$TOOL" print-field --manifest "$MANIFEST" --field "$1"
}

"$PYTHON" "$TOOL" verify --manifest "$MANIFEST" --phase "$MODE"

OUTPUT_ROOT=$(field outputs.root)
RUN_ID=$(field run_id)
ACCOUNT=$(field resources.account)
PARTITION=$(field resources.partition)
CPUS=$(field resources.cpus_per_task)
MEMORY=$(field resources.memory)
WORLD_SIZE=$(field resources.world_size)
GPU_TYPE=$(field resources.gpu_type)
mkdir -p "$OUTPUT_ROOT/slurm" "$OUTPUT_ROOT/submissions"

COMMON=(
  "$SBATCH"
  --parsable
  "--account=$ACCOUNT"
  "--partition=$PARTITION"
  --nodes=1
  "--cpus-per-task=$CPUS"
  "--mem=$MEMORY"
  --open-mode=append
)

case "$MODE" in
  norm)
    JOB_NAME=${RUN_ID}-norm
    EXPORTS="ALL,MODE=norm,RUN_SPEC=$MANIFEST,REPO=$REPO,PY_ENV=$PY_ENV"
    PHASE_ARGS=(
      "--qos=$(field resources.smoke_qos)"
      "--time=$(field resources.smoke_time)"
      --ntasks=1
      --no-requeue
    )
    ;;
  smoke)
    JOB_NAME=${RUN_ID}-smoke
    EXPORTS="ALL,MODE=smoke,RUN_BUNDLE=$MANIFEST,REPO=$REPO,PY_ENV=$PY_ENV"
    PHASE_ARGS=(
      "--qos=$(field resources.smoke_qos)"
      "--time=$(field resources.smoke_time)"
      "--ntasks-per-node=$WORLD_SIZE"
      "--gres=gpu:${GPU_TYPE,,}:$WORLD_SIZE"
      --no-requeue
    )
    ;;
  full)
    JOB_NAME=$RUN_ID
    FULL_SIGNAL=$(field resources.full_signal)
    EXPORTS="ALL,MODE=full,RUN_BUNDLE=$MANIFEST,REPO=$REPO,PY_ENV=$PY_ENV,FULL_SIGNAL=$FULL_SIGNAL"
    PHASE_ARGS=(
      "--qos=$(field resources.full_qos)"
      "--time=$(field resources.full_time)"
      "--ntasks-per-node=$WORLD_SIZE"
      "--gres=gpu:${GPU_TYPE,,}:$WORLD_SIZE"
      --requeue
      "--signal=$FULL_SIGNAL"
    )
    ;;
esac

COMMAND=(
  "${COMMON[@]}"
  "${PHASE_ARGS[@]}"
  "--job-name=$JOB_NAME"
  "--output=$OUTPUT_ROOT/slurm/%x-%j.out"
  "--export=$EXPORTS"
  "$LAUNCHER"
)
JOB_ID=$("${COMMAND[@]}")
test -n "$JOB_ID"

RECEIPT=$OUTPUT_ROOT/submissions/${MODE}_job_${JOB_ID}.txt
{
  printf 'job_id=%s\n' "$JOB_ID"
  printf 'mode=%s\n' "$MODE"
  printf 'manifest=%s\n' "$MANIFEST"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} > "$RECEIPT"
chmod 444 "$RECEIPT"
printf '%s\n' "$JOB_ID"
