#!/usr/bin/env bash
# Find (and optionally kill) your own dataloader workers that outlived a cancelled
# job and are still pinning GPU memory.
#
# The probe is `python -m egomimic.utils.gpu_orphans` (see there for why workers
# leak); run this with the venv active, srun hands the step this shell's PATH.
#
#   ./reap_gpu_orphans.sh gpu19            # report only
#   ./reap_gpu_orphans.sh --kill gpu19     # reclaim
#   ./reap_gpu_orphans.sh --kill gpu11 gpu19
#
# Only processes that are yours, orphaned (PPID 1) and holding an nvidia fd are
# ever signalled, so this is safe to run against a node with live jobs on it.
set -uo pipefail
cd "$(dirname "$0")"  # srun keeps the cwd, so `python -m` finds egomimic

KILL=
[[ ${1:-} == --kill ]] && { KILL=--kill; shift; }
if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--kill] <node> [node...]" >&2
  exit 2
fi

probe() {  # probe <srun args...>; returns srun's status, not grep's
  "$@" python -m egomimic.utils.gpu_orphans $KILL 2>&1 | grep -v '^srun: job'
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
