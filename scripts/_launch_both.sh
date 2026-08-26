#!/bin/bash
# Per-arm idempotent submit, hardened against a hanging/flaky SLURM controller
# (every slurm call is timeout-wrapped so the script can never block).
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
LOG=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/_launch_long.log
: > "$LOG"
TO="timeout 20"

# Probe controller; bail (rc=2) if unreachable/hung so a retry can try later.
if ! $TO squeue -u "$USER" -h >/dev/null 2>&1; then
  echo "CONTROLLER_DOWN_OR_HUNG" | tee -a "$LOG"; exit 2
fi

# Clear any stray mis-named long jobs from earlier failed attempts.
$TO scancel -u "$USER" --name=ppoC3kLONG 2>/dev/null && echo "cleared stray ppoC3kLONG" >> "$LOG"

submit_if_absent () {
  local name="$1"; shift
  if $TO squeue -u "$USER" -h -n "$name" 2>/dev/null | grep -q .; then
    echo "skip $name (already queued/running)" >> "$LOG"
  else
    $TO sbatch --job-name="$name" "$@" scripts/train_ppo_c3000_long.sh >> "$LOG" 2>&1
  fi
}

submit_if_absent ppoLONG_D1 --export=ALL,ARM=D1_lr4e6_anc05_ent002,LR=4e-6,ANCHOR=0.05,ENT=0.002,ITERS=500
submit_if_absent ppoLONG_D2 --export=ALL,ARM=D2_lr4e6_anc0_ent01,LR=4e-6,ANCHOR=0.0,ENT=0.01,ITERS=500

echo "=== queue now ===" >> "$LOG"
$TO squeue -u "$USER" -o "%.10i %.18j %.2t %.6M %R" 2>/dev/null | grep -iE "ppoLONG|JOBID" >> "$LOG"
cat "$LOG"
