#!/bin/bash
# Lightweight Skynet-login watchdog for a temporary Slurm-controller outage.
# It submits the all-six-mode generation array once, after first checking that
# a matching job does not already exist. The atomic lock prevents two copies
# of this watchdog from racing to submit duplicate arrays.

set -uo pipefail

EXP=/coc/flash7/paphiwetsa3/experiments/pushshapes_gapgen_all6x13_1000_20260826
LAUNCHER="$EXP/source/Tsimulation/sim_v2/generate/control_gap_all6_1000_array.sbatch"
SBATCH=/opt/slurm/Ubuntu-20.04/current/bin/sbatch
SQUEUE=/opt/slurm/Ubuntu-20.04/current/bin/squeue
SCONTROL=/opt/slurm/Ubuntu-20.04/current/bin/scontrol
LOCK="$EXP/submit_watchdog.lock"

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FATAL: another submit watchdog owns $LOCK" >&2
  exit 2
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM

while true; do
  status=$(timeout 8 "$SCONTROL" ping 2>&1 || true)
  printf "%s controller: %s\n" \
    "$(date --iso-8601=seconds)" \
    "$(printf '%s' "$status" | tr '\n' ';')"

  if printf '%s' "$status" | grep -q 'UP'; then
    existing=$(timeout 15 "$SQUEUE" -u "$USER" -h \
      -n ps_gap6x13_1k -o '%A' 2>/dev/null || true)
    existing=$(printf '%s\n' "$existing" | head -1)
    if [[ -n "$existing" ]]; then
      echo "EXISTING_JOB=$existing"
      exit 0
    fi

    cd "$EXP/source" || exit 3
    if submitted=$(timeout 30 "$SBATCH" --parsable "$LAUNCHER" 2>&1); then
      echo "SUBMITTED_JOB=$submitted"
      exit 0
    else
      rc=$?
      echo "sbatch rc=$rc output=$submitted"
    fi
  fi
  sleep 30
done
