#!/usr/bin/env bash
# The DataLoader knobs, end to end. Each row is a full trainHydra run on the
# locally cached mecka episodes; read `run_training_batch` (per-step cost) and
# `run_training_epoch` minus 40 x that (per-epoch worker overhead).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/../logs/e2e}"
P="+data.train_dataloader_params.human_bimanual"

declare -A ROWS=(
  [off]="$P.persistent_workers=false $P.pin_memory=false $P.prefetch_factor=2"
  [persist]="$P.persistent_workers=true $P.pin_memory=false $P.prefetch_factor=4"
  [persist_pin]="$P.persistent_workers=true $P.pin_memory=true $P.prefetch_factor=4"
  [persist_compile]="$P.persistent_workers=true $P.pin_memory=false $P.prefetch_factor=4 model.compile.enabled=true"
)
for row in off persist persist_pin persist_compile; do
  rm -f "$OUT.$row.log"
  # shellcheck disable=SC2086
  bash "$HERE/e2e_hpt.sh" "$row" ${ROWS[$row]} > "$OUT.$row.log" 2>&1
done

for row in off persist persist_pin persist_compile; do
  printf '%-18s ' "$row"
  grep -h "steady state over" "$OUT.$row.log" || echo FAILED
done
