#!/bin/bash
#SBATCH --job-name=ice-ckpt-mirror
#SBATCH --account=coc
#SBATCH --partition=ice-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

: "${ICE_MIRROR_MANIFEST:?set ICE_MIRROR_MANIFEST}"
: "${ICE_MIRROR_STATE_DIR:?set ICE_MIRROR_STATE_DIR}"
: "${ICE_SCRATCH_ROOT:?set ICE_SCRATCH_ROOT}"
: "${ICE_SCRATCH_QUOTA_BYTES:?set ICE_SCRATCH_QUOTA_BYTES}"
: "${ICE_MIRROR_REMOTE_HOST:?set ICE_MIRROR_REMOTE_HOST}"
: "${ICE_MIRROR_SCRIPT:?set the exact absolute ice_checkpoint_mirror.py path}"
: "${ICE_MIRROR_PYTHON:?set the exact absolute Python with validator dependencies}"

[[ "${ICE_MIRROR_SCRIPT}" = /* ]]
[[ "${ICE_MIRROR_PYTHON}" = /* ]]
test -f "${ICE_MIRROR_SCRIPT}"
test -r "${ICE_MIRROR_SCRIPT}"
test -x "${ICE_MIRROR_PYTHON}"

ARGS=(
  --manifest "${ICE_MIRROR_MANIFEST}"
  --state-dir "${ICE_MIRROR_STATE_DIR}"
  --scratch-root "${ICE_SCRATCH_ROOT}"
  --quota-bytes "${ICE_SCRATCH_QUOTA_BYTES}"
  --remote-host "${ICE_MIRROR_REMOTE_HOST}"
)
if [[ "${ICE_MIRROR_PRUNE:-0}" == "1" ]]; then
  ARGS+=(--prune-after-verify)
fi
if [[ -n "${ICE_MIRROR_POLL_SECONDS:-}" ]]; then
  ARGS+=(--poll-seconds "${ICE_MIRROR_POLL_SECONDS}")
fi
if [[ -n "${ICE_MIRROR_DU_TIMEOUT_SECONDS:-}" ]]; then
  ARGS+=(--du-timeout-seconds "${ICE_MIRROR_DU_TIMEOUT_SECONDS}")
fi
if [[ -n "${ICE_MIRROR_SSH:-}" ]]; then
  ARGS+=(--ssh "${ICE_MIRROR_SSH}")
fi
if [[ -n "${ICE_MIRROR_RSYNC_RSH:-}" ]]; then
  ARGS+=(--rsync-rsh "${ICE_MIRROR_RSYNC_RSH}")
fi

"${ICE_MIRROR_PYTHON}" "${ICE_MIRROR_SCRIPT}" "${ARGS[@]}" &
CHILD_PID=$!

requeue_at_boundary() {
  kill -USR1 "${CHILD_PID}" 2>/dev/null || true
  local grace=${ICE_MIRROR_SHUTDOWN_GRACE_SECONDS:-240}
  if [[ ! "${grace}" =~ ^[0-9]+$ ]] || (( grace < 1 || grace > 270 )); then
    echo "ICE_MIRROR_SHUTDOWN_GRACE_SECONDS must be an integer in 1..270" >&2
    exit 64
  fi
  local waited=0
  while kill -0 "${CHILD_PID}" 2>/dev/null && (( waited < grace )); do
    sleep 1
    ((waited += 1))
  done
  if kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
    sleep 5
  fi
  if kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -KILL "${CHILD_PID}" 2>/dev/null || true
  fi
  wait "${CHILD_PID}" || true
  local scontrol_bin
  scontrol_bin=$(command -v scontrol) || {
    echo "scontrol is unavailable on PATH" >&2
    exit 69
  }
  if ! "${scontrol_bin}" requeue "${SLURM_JOB_ID}"; then
    echo "scontrol requeue failed for ${SLURM_JOB_ID}" >&2
    exit 70
  fi
  exit 0
}

trap requeue_at_boundary USR1
trap 'kill -TERM "${CHILD_PID}" 2>/dev/null || true; wait "${CHILD_PID}" || true' TERM INT
wait "${CHILD_PID}"
