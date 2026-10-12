#!/bin/bash
# Submit four full-node, eight-GPU offline validation sweeps.
set -euo pipefail
PARTITION="${1:?authorized Slurm partition required}"
UNSEEN_DIR="${2:?unseen validation directory required}"
FLAGSHIP="${3:?flagship preview episode required}"
UNSEEN_PREVIEW="${4:?unseen preview episode required}"
EXPECTED_EPISODES="${5:?expected unseen episode count required}"
ROOT=/workspace/users/ani-cheluva/EgoVerse
OUT="$ROOT/logs/fixed_budget_opscale_hist/offline_sweep_20261010"
SCRIPT="$ROOT/egomimic/scripts/offline_opscale_sweep/sweep.sbatch"
cd "$ROOT"
mkdir -p "$OUT"
ACTUAL_EPISODES=$(find "$UNSEEN_DIR" -maxdepth 1 -mindepth 1 -name '*.zarr' | wc -l)
if [[ "$ACTUAL_EPISODES" -ne "$EXPECTED_EPISODES" ]]; then
  echo "Expected $EXPECTED_EPISODES unseen episodes, found $ACTUAL_EPISODES" >&2
  exit 2
fi
for ITEM in "$UNSEEN_DIR"/*.zarr; do test -e "$ITEM"; done
test -e "/workspace/datasets/fixed_budget_opscale/flagship_unseen_val/$FLAGSHIP.zarr"
test -e "$UNSEEN_DIR/$UNSEEN_PREVIEW.zarr"
node egomimic/scripts/offline_opscale_sweep/make_manifest.mjs
for OPS in 4 8 12 16; do
  MANIFEST="$OUT/op${OPS}-checkpoints.tsv"
  JOB_ID=$(sbatch --parsable --account=loaner --partition="$PARTITION" \
    --job-name="opscale-offline-op${OPS}" \
    --output="$OUT/op${OPS}-%j.out" \
    "$SCRIPT" "$OPS" "$MANIFEST" "$UNSEEN_DIR" "$FLAGSHIP" "$UNSEEN_PREVIEW")
  printf '%s\t%s\t%s\n' "$OPS" "$JOB_ID" "$MANIFEST" | tee -a "$OUT/jobs.tsv"
done
