#!/usr/bin/env bash
# num_workers, end to end. The data configs ship 8; a loaner node has 224 cores.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../logs/workers"
for n in 8 24 48; do
  rm -f "$OUT.$n.log"
  bash "$HERE/e2e_hpt.sh" "w$n" "data.train_dataloader_params.human_bimanual.num_workers=$n" \
    > "$OUT.$n.log" 2>&1
done
rm -f "$OUT.24c.log"
bash "$HERE/e2e_hpt.sh" w24c "data.train_dataloader_params.human_bimanual.num_workers=24" \
  model.compile.enabled=true > "$OUT.24c.log" 2>&1
for n in 8 24 48 24c; do
  printf '%-6s ' "$n"
  grep -h "steady state over" "$OUT.$n.log" || echo FAILED
done
