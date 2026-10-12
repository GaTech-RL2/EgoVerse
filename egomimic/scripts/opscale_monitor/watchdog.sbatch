#!/bin/bash
#SBATCH --account=loaner
#SBATCH --partition=rl2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=14-00:00:00
#SBATCH --job-name=opscale-watchdog
#SBATCH --output=/workspace/users/ani-cheluva/EgoVerse/logs/fixed_budget_opscale_hist/watchdog/watchdog-%j.out
#SBATCH --requeue
#SBATCH --chdir=/workspace/users/ani-cheluva/EgoVerse

set -euo pipefail
REPO=/workspace/users/ani-cheluva/EgoVerse
MONITOR="$REPO/logs/fixed_budget_opscale_hist/watchdog"
cd "$REPO"
source emimic/bin/activate
mkdir -p "$MONITOR"
exec 9>"$MONITOR/watchdog.lock"
flock -n 9 || { echo "another watchdog holds the lock"; exit 0; }
echo "watchdog started job=$SLURM_JOB_ID at $(date -u +%FT%TZ)"
while true; do
  if [[ -f "$MONITOR/stop" ]]; then
    echo "stop file found; exiting"
    break
  fi
  set +e
  python egomimic/scripts/opscale_monitor/scan.py
  scan_result=$?
  set -e
  if [[ "$scan_result" == 3 ]]; then
    echo "all four rungs complete; exiting"
    break
  fi
  if [[ "$scan_result" == 10 ]]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    echo "waking Codex at $stamp"
    set +e
    timeout --signal=TERM --kill-after=30s 1800 \
      codex exec --approve-for-me --ephemeral -C "$REPO" \
      --output-last-message "$MONITOR/agent-$stamp.md" - \
      < "$REPO/egomimic/scripts/opscale_monitor/agent_prompt.md" \
      > "$MONITOR/agent-$stamp.log" 2>&1
    agent_result=$?
    set -e
    echo "Codex exit=$agent_result at $(date -u +%FT%TZ); see agent-$stamp.log"
  fi
  sleep 1800
done
