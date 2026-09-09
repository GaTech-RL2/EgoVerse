#!/bin/bash
#SBATCH --job-name=e1rescore
#SBATCH --partition=coe-gpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/e1rescore_%x_%j.log
# Re-score one finished E1 cell under METRIC (paired_mse = the repo's all-actions MSE; arcmatch_paired_mse = the lab's
# arc-matched MSE, aniketh/arc 57763c49) with a non-leaky selection:
#   sbatch --job-name=rs_<run> --export=ALL,RUN=<run dir>,VARIANT=arcmean,SPREAD=full,SEED=42,EXP=abc abc_rescore.sbatch
# 1. both surviving checkpoints (best.json's best_ckpt / final_ckpt) are evaluated on val_e1 (skipped when
#    eval_<tag>_val_e1 already carries paired_mse);  2. the one with the lower val_e1 paired_mse is "selected";
#    3. the selected checkpoint is evaluated on test_low/mid/high (skipped where paired_mse already exists);
#    4. best.json gains {"rescore": {"val_e1": {tag: paired_mse}, "selected": tag}}.
set -euo pipefail
WT=${WT:-$HOME/scratch/EgoVerse/.claude/worktrees/arc-e1}; PY=${PY:-$HOME/scratch/EgoVerse/.venv/bin/python}
DATA=${DATA:-$HOME/scratch/data/e1_abc}; EXP=${EXP:-abc}; VAL=${VAL:-val_e1}; METRIC=${METRIC:-paired_mse}
cd $WT; export PYTHONPATH=$WT HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID
H_MATCH=${H_MATCH:-$(cat $DATA/h_match_frames)}
NORM_DIR=$(dirname $(find $RUN/train -name "norm_stats*.json" | head -1))
has_metric() { $PY - "$1" "$METRIC" <<'PYEOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))["results"][0]
except Exception:
    sys.exit(1)
k = sys.argv[2]
ok = any(kk.startswith(f"Valid/E1/{k}/") for kk in r) or (k == "paired_mse" and any("paired_mse_full" in kk for kk in r))
sys.exit(0 if ok else 1)
PYEOF
}
metric_val() { $PY - "$1" "$METRIC" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))["results"][0]; k = sys.argv[2]
if k == "paired_mse":
    kk = next((x for x in r if "paired_mse_full" in x), None) or next(x for x in r if "/paired_mse/" in x)
else:
    kk = next(x for x in r if x.startswith(f"Valid/E1/{k}/"))
print(r[kk])
PYEOF
}
run_eval() {  # tag set ckpt
  local TAG=$1 TS=$2 CKPT=$3 OUTD=$RUN/eval_${1}_$2
  if has_metric $OUTD/eval_metrics.json; then echo "have $TAG/$TS ($METRIC)"; return; fi
  echo "=== EVAL $TAG on $TS ckpt=$CKPT ($(date)) ==="
  $PY egomimic/trainHydra.py --config-name=train_zarr_cartesian +experiment=e1/${EXP}_${VARIANT} mode=eval "ckpt_path=\"$CKPT\"" \
    e1.spread=$SPREAD e1.train_root=$DATA/$SPREAD e1.valid_root=$DATA/$TS e1.h_match_frames=$H_MATCH e1.limit_val_batches=1.0 \
    seed=$SEED norm_stats.precomputed_norm_path=$NORM_DIR trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 \
    ${EVAL_EXTRA:-} hydra.run.dir=$OUTD
}
BEST=$($PY -c "import json;print(json.load(open('$RUN/best.json'))['best_ckpt'])")
FINAL=$($PY -c "import json;print(json.load(open('$RUN/best.json'))['final_ckpt'])")
[ -f "$BEST" ] || { echo "missing best ckpt $BEST"; exit 1; }
[ -f "$FINAL" ] || { echo "final ckpt $FINAL was purged earlier; best-only"; FINAL=$BEST; }
TAGS="best"; [ "$FINAL" != "$BEST" ] && TAGS="best final"
for TAG in $TAGS; do CK=$BEST; [ $TAG = final ] && CK=$FINAL; run_eval $TAG $VAL $CK; done
SEL=best; VB=$(metric_val $RUN/eval_best_$VAL/eval_metrics.json); VF=$VB
if [ "$TAGS" != "best" ]; then VF=$(metric_val $RUN/eval_final_$VAL/eval_metrics.json); SEL=$($PY -c "print('final' if $VF < $VB else 'best')"); fi
echo "val_e1 $METRIC: best=$VB final=$VF -> selected=$SEL"
CK=$BEST; [ $SEL = final ] && CK=$FINAL
for TS in ${TEST_SETS:-test_mid test_low test_high}; do run_eval $SEL $TS $CK; done
$PY - "$RUN" "$SEL" "$VB" "$VF" "$METRIC" <<'PYEOF'
import json, sys
run, sel, vb, vf, metric = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
d = json.load(open(f"{run}/best.json")); key = "rescore" if metric == "paired_mse" else f"rescore_{metric}"
d[key] = {"val": "val_e1", "metric": metric, "val_e1": {"best": vb, "final": vf}, "selected": sel}
json.dump(d, open(f"{run}/best.json", "w"), indent=2)
PYEOF
echo "=== DONE ($(date)) ==="
