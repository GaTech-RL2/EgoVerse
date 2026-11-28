#!/usr/bin/env bash
# Resource selection is provided by submit.py, never copied from historical overcap jobs.
set -euo pipefail
export PATH=/opt/slurm/Ubuntu-24.04/current/bin:$PATH
source "${CAMPAIGN_ROOT:?}/scripts/env.sh"
config="$CAMPAIGN_ROOT/configs/${RUN_ID:?}.json"
args=()
[[ "${SMOKE:-0}" == 1 ]] && args+=(--smoke)
"$CAMPAIGN_PYTHON" -m campaign.train "$config" "${args[@]}" &
child=$!
# skynet-progress-sidecar-v1
if [[ "${SMOKE:-0}" != 1 ]]; then
  "$CAMPAIGN_PYTHON" /coc/flash7/cxin34/skynet-training-monitor/auto_report.py --pid "$child" --config "$config" &
fi
trap 'kill -USR1 "$child" 2>/dev/null || true' USR1
trap 'kill -TERM "$child" 2>/dev/null || true' TERM
set +e
wait "$child"; rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
set -e
if [[ "$rc" == 75 ]]; then
  if [[ "${SMOKE:-0}" == 1 ]]; then
    # Explicit second process verifies full-state restoration at step100 -> step200.
    exec "$CAMPAIGN_PYTHON" -m campaign.train "$config" --smoke
  fi
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
fi
exit "$rc"
