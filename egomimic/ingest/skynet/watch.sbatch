#!/bin/bash
# One ingestion watcher pass on SkyNet, then the next one INGEST_INTERVAL_MIN later.
# Setup, start and stop: egomimic/ingest/skynet/README.md.
#SBATCH --job-name=egoverse-ingest
#SBATCH --partition=overcap
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
set -euo pipefail

: "${INGEST_ROOT:?set INGEST_ROOT, e.g. /coc/flash7/scratch/mcap_pipeline}"
CODE="$INGEST_ROOT/code/current"        # deploy.sh: an exact revision, no git checkout
VENV="$INGEST_ROOT/venv"
ENV_FILE="$INGEST_ROOT/egoverse_env"    # R2_* credentials, chmod 600
FFMPEG_DIR="${INGEST_FFMPEG_DIR:-/coc/flash7/rco3/ffmpeg-7.1-shared}"
INTERVAL="${INGEST_INTERVAL_MIN:-15}"
LOGS="$INGEST_ROOT/logs"
LANCE_ROOT="${INGEST_LANCE_ROOT:-$INGEST_ROOT/lance}"   # a smoke test sets its own
WORK="${INGEST_WORK:-$INGEST_ROOT/work}"

if [ -e "$INGEST_ROOT/STOP" ]; then
    echo "$(date -u +%FT%TZ) STOP file present: no pass, no next job"
    exit 0
fi

# Queue the next pass before this one runs, so a crash, timeout or preemption of this
# job cannot end the chain. Only if none is queued yet: one chain, never two.
# (grep finding nothing exits 1, which pipefail would turn into the job failing.)
next=""
others=$(squeue -h -u "$USER" -n egoverse-ingest -o %i | { grep -v -x "${SLURM_JOB_ID:-none}" || true; } | wc -l)
if [ "${INGEST_ONCE:-0}" = 1 ]; then
    echo "$(date -u +%FT%TZ) INGEST_ONCE: one pass, no next job"
elif [ "$others" -eq 0 ]; then
    next=$(sbatch --parsable \
        --begin="now+${INTERVAL}minutes" \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --output="$LOGS/%j.out" \
        --export=ALL \
        "$CODE/egomimic/ingest/skynet/watch.sbatch")
    echo "$(date -u +%FT%TZ) next pass queued as job $next"
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$FFMPEG_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH
cd "$CODE"

# MANO fit for Aria-layout hands, on by default: the mano package is installed in the
# venv and the license-gated model files live outside the code, in INGEST_MANO_MODEL_DIR.
# INGEST_MANO=0 stores hands in the Aria layout instead (as aria_to_lance --no-mano).
MANO_FLAG=(--mano --mano-model-dir "${INGEST_MANO_MODEL_DIR:-$INGEST_ROOT/models/mano}")
if [ "${INGEST_MANO:-1}" = 0 ]; then
    MANO_FLAG=(--no-mano)
fi

echo "$(date -u +%FT%TZ) pass with revision $(cat egomimic/ingest/REVISION 2>/dev/null || echo unknown)"
rc=0
python -m egomimic.ingest.watch \
    --lance "$LANCE_ROOT" \
    --work "$WORK" \
    --bucket "${INGEST_BUCKET:-rldb}" \
    --root "${INGEST_PREFIX:-}" \
    --codec libsvtav1 \
    --ffmpeg "$FFMPEG_DIR/bin/ffmpeg" \
    "${MANO_FLAG[@]}" || rc=$?
if [ "$rc" -eq 2 ]; then
    # Set up wrong (e.g. MANO missing): every later pass would fail the same way, so
    # stop the chain instead of retrying every INGEST_INTERVAL_MIN.
    [ -n "$next" ] && scancel "$next"
    echo "$(date -u +%FT%TZ) setup error: chain stopped; fix it and start again"
    exit 2
fi
echo "$(date -u +%FT%TZ) pass done (exit $rc)"
exit "$rc"
