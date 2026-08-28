#!/bin/bash
set -euo pipefail

MODE=${1:?usage: submit_image_generation_unified.sh smoke|full}
case "$MODE" in
  smoke|full) ;;
  *) echo "unsupported mode: $MODE" >&2; exit 2 ;;
esac

WORKTREE=${WORKTREE:-/coc/flash7/paphiwetsa3/worktrees/image-unified-latent-20260828}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/coc/flash7/paphiwetsa3/experiments/imagenet256_unified_latent_20260828}
DATA_ROOT=${DATA_ROOT:-/coc/dataset/ImageNet/imagenet}
SLURM_BIN=/opt/slurm/Ubuntu-20.04/current/bin
PY_ENV=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv

test -z "$(git -C "$WORKTREE" status --porcelain=v1 --untracked-files=all)"
SHA=$(git -C "$WORKTREE" rev-parse HEAD)
SOURCE_ROOT="$EXPERIMENT_ROOT/source_${SHA:0:8}"
mkdir -p "$EXPERIMENT_ROOT/slurm" "$EXPERIMENT_ROOT/provenance"
if [[ ! -e "$SOURCE_ROOT" ]]; then
  git -C "$WORKTREE" worktree add --detach "$SOURCE_ROOT" "$SHA"
  git -C "$WORKTREE" worktree lock "$SOURCE_ROOT" \
    --reason "immutable ImageNet unified-latent source $SHA"
fi
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)"

INVENTORY="$EXPERIMENT_ROOT/provenance/dataset_inventory.txt"
if [[ ! -s "$INVENTORY" || ! -s "${INVENTORY%.txt}.sha256" ]]; then
  "$PY_ENV/bin/python" "$SOURCE_ROOT/scripts/train/inventory_imagenet.py" \
    --data-root "$DATA_ROOT" \
    --output "$INVENTORY"
fi

EXPORTS="ALL,SOURCE_ROOT=$SOURCE_ROOT,EXPERIMENT_ROOT=$EXPERIMENT_ROOT,EXPECTED_SHA=$SHA,DATA_ROOT=$DATA_ROOT"
LAUNCHER="$SOURCE_ROOT/scripts/train/image_generation_matched.sbatch"
QOS_ARGS=()
if [[ "$MODE" == smoke ]]; then
  QOS_ARGS=(--qos=short --time=02:00:00 --no-requeue)
fi

JOB=$(
  "$SLURM_BIN/sbatch" --parsable \
    --job-name="img-unified-${MODE}" \
    "${QOS_ARGS[@]}" \
    --output="$EXPERIMENT_ROOT/slurm/%x-%j.out" \
    --error="$EXPERIMENT_ROOT/slurm/%x-%j.err" \
    --export="$EXPORTS" \
    "$LAUNCHER" unified_latent "$MODE"
)

printf 'SHA=%s\nSOURCE_ROOT=%s\nMODE=%s\nJOB=%s\n' \
  "$SHA" "$SOURCE_ROOT" "$MODE" "$JOB"
