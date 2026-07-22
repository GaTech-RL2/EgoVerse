#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --job-name=norm_foldviz_dbg
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/norm_foldviz_dbg_%j.log

# DEBUG-scale foldviz norm stats: tiny sample (0.5%, 50k cap) so the smoke
# can validate the new-extrinsics path end to end without waiting for the
# full recompute (norm_abc_eva_foldviz.sbatch). Writes to *_debug dirs only —
# never to the real norm_stats/mecka_abc_6d the full pretrain reads.

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"

OUT=/storage/project/r-dxu345-0/agao81/norm_stats/abc_eva_6d_foldviz_debug
mkdir -p "$OUT"

"$PY" -u egomimic/scripts/precompute_norm_stats.py \
  --data abc_eva_pi_6d_foldviz \
  --model pi0.5_ft_abc_eva_6d \
  --sample-frac 0.005 \
  --max-samples 50000 \
  --num-workers 22 \
  --out "$OUT"

# Debug merged stats (mecka id 3 real + eva id 6 debug-quality foldviz).
"$PY" - <<'PYEOF'
import json, os
base = '/storage/project/r-dxu345-0/agao81/norm_stats'
m = json.load(open(f'{base}/mecka_all_6d/norm_stats/norm_stats.json'))
e = json.load(open(f'{base}/abc_eva_6d_foldviz_debug/norm_stats/norm_stats.json'))
assert set(m['stats']) & set(e['stats']) == set(), 'embodiment id collision'
out = {'stats': {**m['stats'], **e['stats']},
       'merged_from': ['mecka_all_6d (id 3)', 'abc_eva_6d_foldviz_debug (id 6, DEBUG-QUALITY)']}
d = f'{base}/mecka_abc_6d_debug/norm_stats'
os.makedirs(d, exist_ok=True)
json.dump(out, open(f'{d}/norm_stats.json', 'w'))
print(f'[merge] wrote {d}/norm_stats.json stats keys:', sorted(out['stats']))
PYEOF