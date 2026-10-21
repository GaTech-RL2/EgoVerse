#!/bin/bash
# Fetch, convert and verify one SharpaDex chunk planned by sharpadex_ingest.
#
# Every stage resumes: fetch skips files already on disk, convert skips
# episodes whose <hash>.conversion.json exists. A job that hits the time
# limit or is preempted is submitted again with the same variables. This job
# never uploads; `sharpadex_ingest upload` is a separate, gated step.
#
# Usage (paths are examples):
#   sbatch --export=ALL,CHUNK=chunk_0000,STATE_DIR=/data/sharpadex_state,\
#   DATASET_ROOT=/data/SharpaDex-v1.0,WORK_ROOT=/data/sharpadex_work,\
#   REPO=$PWD,PYTHON=/path/to/env/bin/python \
#       egomimic/scripts/sharpa_process/sharpadex_chunk.sbatch
#
# Variables:
#   CHUNK, STATE_DIR, DATASET_ROOT, WORK_ROOT, REPO   required
#   PYTHON           the EgoVerse env's interpreter (default: python3)
#   PLAN             plan file (default: $STATE_DIR/plan.json)
#   MANIFEST         manifest (default: the one the plan records)
#   CPUS_PER_WORKER  CPUs per converter process (default 4); the job runs
#                    (allocated CPUs / CPUS_PER_WORKER) workers
#   SHARD            i/n: convert every n-th episode of the chunk from the
#                    i-th, so n jobs can split one chunk. Run fetch once
#                    before (STAGES=fetch) and verify once after all shards
#                    finish (STAGES=verify).
#   STAGES           default "fetch convert verify"; "convert" with SHARD
#   ALLOW_PROVISIONAL_HASHES=1  trial runs only: fetch and convert although
#                    the manifest's hashes are not collision-checked
#   FETCH_ARGS, VERIFY_ARGS     extra flags for fetch / verify
#   EXTRA_ARGS       extra flags for convert (e.g. "--max-load-ratio 1.5")
#
# Time limit: rl2-lab caps jobs at 4 h (MaxTime). For longer runs use the
# preemptible overcap partition (up to 2 days):
#   sbatch -p overcap -A rl2-lab --time=2-00:00:00 --requeue ...
# A requeued job keeps its job id; the driver treats a lock left by the same
# job id as stale, so the requeued job resumes.
#
# convert refuses to start on a node whose load exceeds its CPU count
# (oversubscribed nodes convert many times slower); exclude busy nodes with
# `sbatch --exclude=...`.
#SBATCH --job-name=sharpadex_chunk
#SBATCH -p rl2-lab
#SBATCH -A rl2-lab
#SBATCH --time=04:00:00
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --output=sharpadex_chunk_%j.log

set -euo pipefail

: "${CHUNK:?set CHUNK, e.g. chunk_0000}"
: "${STATE_DIR:?set STATE_DIR (the plan and ledger folder)}"
: "${DATASET_ROOT:?set DATASET_ROOT (the local release copy)}"
: "${WORK_ROOT:?set WORK_ROOT (outputs go to WORK_ROOT/out/CHUNK)}"
: "${REPO:?set REPO (the EgoVerse checkout to run)}"
PYTHON=${PYTHON:-python3}
PLAN=${PLAN:-$STATE_DIR/plan.json}
MANIFEST=${MANIFEST:-}
CPUS_PER_WORKER=${CPUS_PER_WORKER:-4}
SHARD=${SHARD:-}
if [[ -n $SHARD ]]; then
    STAGES=${STAGES:-convert}
else
    STAGES=${STAGES:-fetch convert verify}
fi
read -r -a fetch_args <<< "${FETCH_ARGS:-}"
read -r -a convert_args <<< "${EXTRA_ARGS:-}"
read -r -a verify_args <<< "${VERIFY_ARGS:-}"

common=(--state-dir "$STATE_DIR" --plan "$PLAN" --chunk "$CHUNK")
if [[ -n $MANIFEST ]]; then
    common+=(--manifest "$MANIFEST")
fi
gate=()
if [[ ${ALLOW_PROVISIONAL_HASHES:-0} == 1 ]]; then
    gate=(--allow-provisional-hashes)
fi
shard=()
if [[ -n $SHARD ]]; then
    shard=(--shard "$SHARD")
fi

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

ingest() {
    "$PYTHON" -m egomimic.scripts.sharpa_process.sharpadex_ingest "$@"
}

resume_hint() {
    local status=$?
    local vars="CHUNK=$CHUNK,STATE_DIR=$STATE_DIR,DATASET_ROOT=$DATASET_ROOT"
    vars+=",WORK_ROOT=$WORK_ROOT,REPO=$REPO,PYTHON=$PYTHON"
    vars+=",CPUS_PER_WORKER=$CPUS_PER_WORKER"
    [[ -n $SHARD ]] && vars+=",SHARD=$SHARD"
    [[ -n $MANIFEST ]] && vars+=",MANIFEST=$MANIFEST"
    [[ $PLAN != "$STATE_DIR/plan.json" ]] && vars+=",PLAN=$PLAN"
    [[ ${ALLOW_PROVISIONAL_HASHES:-0} == 1 ]] && vars+=",ALLOW_PROVISIONAL_HASHES=1"
    echo "sharpadex_chunk: job ${SLURM_JOB_ID:-none} ($CHUNK, stages: $STAGES) exited with status $status"
    echo "Every stage resumes; to continue, resubmit with the same variables:"
    echo "  sbatch --export=ALL,$vars,STAGES=\"$STAGES\" \\"
    echo "      $REPO/egomimic/scripts/sharpa_process/sharpadex_chunk.sbatch"
    echo "(add FETCH_ARGS / EXTRA_ARGS / VERIFY_ARGS again if you used them)"
    echo "Progress: $PYTHON -m egomimic.scripts.sharpa_process.sharpadex_ingest status --state-dir $STATE_DIR"
}
trap resume_hint EXIT

echo "host $(hostname), job ${SLURM_JOB_ID:-none}, $(nproc) CPUs, load $(cut -d' ' -f1-3 /proc/loadavg)"
for stage in $STAGES; do
    echo "=== $stage $CHUNK $(date -Is)"
    case $stage in
        fetch)
            # Only the converter workers run offline.
            env -u HF_HUB_OFFLINE "$PYTHON" -m egomimic.scripts.sharpa_process.sharpadex_ingest \
                fetch "${common[@]}" --dataset-root "$DATASET_ROOT" --work-root "$WORK_ROOT" \
                "${gate[@]}" "${fetch_args[@]}"
            ;;
        convert)
            ingest convert "${common[@]}" --dataset-root "$DATASET_ROOT" --work-root "$WORK_ROOT" \
                --cpus-per-worker "$CPUS_PER_WORKER" "${shard[@]}" "${gate[@]}" "${convert_args[@]}"
            ;;
        verify)
            ingest verify "${common[@]}" --work-root "$WORK_ROOT" "${verify_args[@]}"
            ;;
        *)
            echo "unknown stage '$stage' (expected fetch, convert, verify)" >&2
            exit 2
            ;;
    esac
done
