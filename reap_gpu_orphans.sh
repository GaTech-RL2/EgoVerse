#!/usr/bin/env bash
# Find (and optionally kill) your own dataloader workers that outlived a cancelled
# job and are still pinning GPU memory.
#
# A worker forked after CUDA init holds its rank's /dev/nvidia* fds, so the rank's
# whole context stays resident even after the rank is gone -- nvidia-smi then
# reports memory against a PID that is no longer in /proc. ProctrackType on this
# cluster is proctrack/linuxproc, which tracks the PPID tree, so workers that got
# reparented to init are never swept.
#
#   ./reap_gpu_orphans.sh gpu19            # report only
#   ./reap_gpu_orphans.sh --kill gpu19     # reclaim
#   ./reap_gpu_orphans.sh --kill gpu11 gpu19
#
# Only processes that are yours, orphaned (PPID 1) and holding an nvidia fd are
# ever signalled, so this is safe to run against a node with live jobs on it.
set -uo pipefail

KILL=0
[[ ${1:-} == --kill ]] && { KILL=1; shift; }
if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--kill] <node> [node...]" >&2
  exit 2
fi

PROBE='
ME=$(id -u); victims=()
for d in /proc/[0-9]*; do
  p=${d#/proc/}
  [[ $(stat -c %u "$d" 2>/dev/null) == "$ME" ]] || continue
  # PPid from status, as gpu_orphans._ppid reads it: in stat, field 2 is comm,
  # which may contain spaces and shift every field after it.
  [[ $(awk "/^PPid:/{print \$2}" "$d/status" 2>/dev/null) == 1 ]] || continue
  ls -l "$d/fd" 2>/dev/null | grep -q /dev/nvidia || continue
  victims+=("$p")
done
echo "node=$(hostname -s) orphans=${#victims[@]}"
[[ ${#victims[@]} -eq 0 ]] && exit 0
for p in "${victims[@]}"; do
  echo "  $p $(cat /proc/$p/comm 2>/dev/null) rss=$(awk "/VmRSS/{print \$2}" /proc/$p/status 2>/dev/null)kB age=$(ps -o etime= -p $p 2>/dev/null | tr -d " ")"
done
echo "before: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " ")"
if [[ $KILL == 1 ]]; then
  kill -9 "${victims[@]}" 2>/dev/null
  for i in 1 2 3 4 5 6; do
    sleep 4
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [[ $n -eq 0 ]] && break
  done
  echo "after:  $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " ")"
  echo "compute apps left: $n"
fi
'

probe() {  # probe <srun args...>; returns srun's status, not grep's
  "$@" bash -c "KILL=$KILL; $PROBE" 2>&1 | grep -v '^srun: job'
  return "${PIPESTATUS[0]}"
}

rc=0
for node in "$@"; do
  echo "=== $node ==="
  # --immediate everywhere: the partition is EXCLUSIVE, so a fresh allocation on a
  # busy node queues forever, and a step inside a job whose training step already
  # holds every CPU never starts. Better to say so than to hang.
  job=$(squeue -h -u "$USER" -w "$node" -t R -o %i | head -1)
  if [[ -n $job ]]; then
    echo "(node held by your job $job; attaching to it)"
    probe srun --overlap --immediate=20 --jobid="$job" -n1 -c 1 && continue
    echo "(could not get a step in $job -- it is probably mid-training)"
  fi
  probe srun -p loaner -A loaner --nodelist="$node" -n1 -c 2 --mem=4G --overlap \
    --immediate=30 -t 5 && continue
  echo "$node is busy; rerun when it is idle. A job starting there cleans it" >&2
  echo "automatically anyway (egomimic.utils.gpu_orphans.reap_orphans)." >&2
  rc=1
done
exit "$rc"
