#!/bin/bash
set -euo pipefail
ROOT=/workspace/users/ani-cheluva/EgoVerse
OUT="$ROOT/logs/fixed_budget_opscale_hist/offline_sweep_20261010"
POOL="$ROOT/logs/fixed_budget_opscale_hist/temporal_sweep_20261010/strict_folding_pool"
cd "$ROOT"
for OPS in 4 8 12 16; do
  MANIFEST="$OUT/op${OPS}-checkpoints.tsv"
  COUNT=$(wc -l < "$MANIFEST")
  JOB=$(sbatch --parsable --array="0-$((COUNT - 1))%8" \
    --job-name="opscale-offline-op${OPS}" --output="$OUT/op${OPS}-%A_%a.out" \
    egomimic/scripts/offline_opscale_sweep/checkpoint_array.sbatch \
    "$OPS" "$MANIFEST" "$POOL" 692eb28d21fc595fbe6f9600 69b1e2191c6c05bcabe09dcd)
  printf '%s\t%s\t%s\n' "$OPS" "$JOB" "$MANIFEST" | tee -a "$OUT/array-jobs.tsv"
done
