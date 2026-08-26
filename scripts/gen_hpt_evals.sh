#!/bin/bash
set -euo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
mkdir -p logs/sbatch

BASE=scripts/eval_cotrain_act8.sbatch

make_script () {
  local CELL=$1 RUNDIR=$2 DS=$3 PUSHER=$4
  local OUT=scripts/eval_indomain_1min_${CELL}.sbatch
  cp "$BASE" "$OUT"

  # job name + output/error paths
  sed -i "s#^#SBATCH --job-name=.*#XX#" "$OUT" 2>/dev/null || true
  sed -i "s|--job-name=evalCotrainAct8|--job-name=evalID_${CELL}|" "$OUT"
  sed -i "s|logs/uecotrain_%j.out|logs/sbatch/evalID_${CELL}_%j.out|" "$OUT"
  sed -i "s|logs/uecotrain_%j.err|logs/sbatch/evalID_${CELL}_%j.err|" "$OUT"

  # RUNDIR / DS
  sed -i "s|^RUNDIR=.*|RUNDIR=${RUNDIR}|" "$OUT"
  sed -i "s|^DS=.*|DS=${DS}|" "$OUT"

  # video dir
  sed -i "s|ROLLOUT_VIDEO_DIR=\${RUNDIR}/eval/.*|ROLLOUT_VIDEO_DIR=\${RUNDIR}/eval/indomain_1min_${CELL}_20seeds_thr70|" "$OUT"

  # name + description
  sed -i "s|name=uecotrain_HptCo description=uecotrain|name=evalID_${CELL} description=indomain_1min_${CELL}|" "$OUT"

  # explicit rollout_mode=chunk_openloop + chunk_k=8 + temporal_ensemble=false + pusher.
  # Insert right after the evaluator.max_steps=1800 line.
  python3 - "$OUT" "$PUSHER" <<'PYEOF'
import sys
path, pusher = sys.argv[1], sys.argv[2]
with open(path) as f:
    lines = f.readlines()
out = []
for ln in lines:
    out.append(ln)
    if "evaluator.max_steps=1800" in ln:
        indent = "  "
        out.append(f"{indent}'+evaluator.rollout_mode=chunk_openloop' \\\n")
        out.append(f"{indent}'+evaluator.chunk_k=8' \\\n")
        out.append(f"{indent}'+evaluator.temporal_ensemble=false' \\\n")
        out.append(f"{indent}evaluator.env_kwargs.pusher_shape={pusher} \\\n")
with open(path, "w") as f:
    f.writelines(out)
PYEOF

  echo "=== wrote $OUT ==="
  grep -nE "job-name|RUNDIR=|^DS=|pusher_shape|rollout_mode|chunk_k|max_steps=1800|init_seeds|coverage_threshold|name=evalID" "$OUT"
  echo
}

make_script bc_circle      /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowC3000CausalAct8/hpt_flow_circle_3000_causal_obs1_act8_2026-06-15_01-27-54           /coc/flash7/paphiwetsa3/datasets/circle_3000        circle
make_script bc_small       /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptFlowSmallCircle3000Act8/hpt_flow_small_circle_3000_causal_obs1_act8_2026-06-19_18-09-20 /coc/flash7/paphiwetsa3/datasets/small_circle_3000  circle_small
make_script cotrain_circle /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptCotrainBigSmallAct8/hpt_flow_cotrain_big_small_3000_causal_obs1_act8_2026-06-16_17-41-59 /coc/flash7/paphiwetsa3/datasets/circle_3000        circle
make_script cotrain_small  /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hptCotrainBigSmallAct8/hpt_flow_cotrain_big_small_3000_causal_obs1_act8_2026-06-16_17-41-59 /coc/flash7/paphiwetsa3/datasets/small_circle_3000  circle_small
