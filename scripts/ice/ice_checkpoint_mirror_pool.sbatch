#!/bin/bash
#SBATCH --job-name=ice-ckpt-pool
#SBATCH --account=coc
#SBATCH --partition=ice-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=18:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

: "${ICE_MIRROR_MANIFEST:?set ICE_MIRROR_MANIFEST}"
: "${ICE_MIRROR_STATE_DIR:?set ICE_MIRROR_STATE_DIR}"
: "${ICE_SCRATCH_ROOT:?set ICE_SCRATCH_ROOT}"
: "${ICE_SCRATCH_QUOTA_BYTES:?set ICE_SCRATCH_QUOTA_BYTES}"
: "${ICE_MIRROR_REMOTE_HOST:?set ICE_MIRROR_REMOTE_HOST}"
: "${ICE_MIRROR_POOL_SCRIPT:?set the exact absolute ice_checkpoint_mirror_pool.py path}"
: "${ICE_MIRROR_PYTHON:?set the exact absolute Python with validator dependencies}"
: "${ICE_MIRROR_POOL_SIZE:?set ICE_MIRROR_POOL_SIZE to the exact array task count}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"
: "${SLURM_ARRAY_TASK_COUNT:?submit this wrapper as a Slurm array}"

[[ "${ICE_MIRROR_POOL_SCRIPT}" = /* ]]
[[ "${ICE_MIRROR_PYTHON}" = /* ]]
test -r "${ICE_MIRROR_POOL_SCRIPT}"
test -x "${ICE_MIRROR_PYTHON}"
[[ "${ICE_MIRROR_POOL_SIZE}" =~ ^[0-9]+$ ]]
(( ICE_MIRROR_POOL_SIZE >= 2 ))
(( SLURM_ARRAY_TASK_COUNT == ICE_MIRROR_POOL_SIZE ))

ARGS=(
  --manifest "${ICE_MIRROR_MANIFEST}"
  --state-dir "${ICE_MIRROR_STATE_DIR}"
  --scratch-root "${ICE_SCRATCH_ROOT}"
  --quota-bytes "${ICE_SCRATCH_QUOTA_BYTES}"
  --remote-host "${ICE_MIRROR_REMOTE_HOST}"
  --worker-index "${SLURM_ARRAY_TASK_ID}"
  --worker-count "${ICE_MIRROR_POOL_SIZE}"
)
if [[ "${ICE_MIRROR_PRUNE:-0}" == "1" ]]; then ARGS+=(--prune-after-verify); fi
if [[ -n "${ICE_MIRROR_POLL_SECONDS:-}" ]]; then ARGS+=(--poll-seconds "${ICE_MIRROR_POLL_SECONDS}"); fi
if [[ -n "${ICE_MIRROR_DU_TIMEOUT_SECONDS:-}" ]]; then ARGS+=(--du-timeout-seconds "${ICE_MIRROR_DU_TIMEOUT_SECONDS}"); fi
if [[ -n "${ICE_MIRROR_SSH:-}" ]]; then ARGS+=(--ssh "${ICE_MIRROR_SSH}"); fi
if [[ -n "${ICE_MIRROR_RSYNC_RSH:-}" ]]; then ARGS+=(--rsync-rsh "${ICE_MIRROR_RSYNC_RSH}"); fi
if [[ -n "${ICE_MIRROR_INVENTORY_ROOT:-}" ]]; then
  ARGS+=(--inventory-search-root "${ICE_MIRROR_INVENTORY_ROOT}")
fi
if [[ -n "${ICE_MIRROR_INVENTORY_MAX_DEPTH:-}" ]]; then
  ARGS+=(--inventory-max-depth "${ICE_MIRROR_INVENTORY_MAX_DEPTH}")
fi

"${ICE_MIRROR_PYTHON}" "${ICE_MIRROR_POOL_SCRIPT}" "${ARGS[@]}" &
CHILD_PID=$!

requeue_at_boundary() {
  kill -USR1 "${CHILD_PID}" 2>/dev/null || true
  local grace=${ICE_MIRROR_SHUTDOWN_GRACE_SECONDS:-240}
  [[ "${grace}" =~ ^[0-9]+$ ]] && (( grace >= 1 && grace <= 270 )) || exit 64
  local waited=0
  while kill -0 "${CHILD_PID}" 2>/dev/null && (( waited < grace )); do
    sleep 1
    ((waited += 1))
  done
  kill -TERM "${CHILD_PID}" 2>/dev/null || true
  wait "${CHILD_PID}" || true
  command -v scontrol >/dev/null || exit 69
  scontrol requeue "${SLURM_JOB_ID}" || exit 70
  exit 0
}

trap requeue_at_boundary USR1
trap 'kill -TERM "${CHILD_PID}" 2>/dev/null || true; wait "${CHILD_PID}" || true' TERM INT
wait "${CHILD_PID}"
