#!/usr/bin/env bash
# persistent_workers, isolated: everything else identical.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../logs/loader"
P="+data.train_dataloader_params.human_bimanual"
for v in false true; do
  rm -f "$OUT.$v.log"
  bash "$HERE/e2e_hpt.sh" "persist_$v" "$P.persistent_workers=$v" \
    "$P.pin_memory=false" "$P.prefetch_factor=2" > "$OUT.$v.log" 2>&1
done
for v in false true; do
  echo "--- persistent_workers=$v"
  grep -h "\[step\]" "$OUT.$v.log" || echo FAILED
done
