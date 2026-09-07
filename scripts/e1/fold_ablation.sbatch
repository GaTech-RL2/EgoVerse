#!/bin/bash
#SBATCH --job-name=e1abl
#SBATCH --partition=coe-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=6:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/e1abl_%x_%j.log
# Tempo-invariance ablation cell (Ideas note "Arc Tokenizer Tempo Invariance Changes"):
#   sbatch --job-name=e1abl_${VARIANT}_${SPREAD}_s${SEED} --export=ALL,VARIANT=arclogdur_ps,SPREAD=full,SEED=42 fold_ablation.sbatch
# VARIANT is the experiment suffix: arcvel (A0) | arclogdur (A1: #1+#2) | arclogdur_ps (A2: +#3+#4).
# 10k steps, val + checkpoint every 1k; the best-validation checkpoint (min E_time on test_mid)
# AND the final checkpoint are evaluated on all three tempo tertiles; other checkpoints are deleted.
set -euo pipefail
WT=${WT:-$HOME/scratch/EgoVerse/.claude/worktrees/arc-e1}
PY=${PY:-$HOME/scratch/EgoVerse/.venv/bin/python}
DATA=${DATA:-$HOME/scratch/data/e1_fold}
OUT=${OUT:-$HOME/scratch/runs/e1_abl}
VARIANT=${VARIANT:?arcvel|arclogdur|arclogdur_ps}; SPREAD=${SPREAD:?narrow|medium|full}; SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-10000}; MAX_EPOCHS=$(( MAX_STEPS / 100 )); VAL_EVERY=${VAL_EVERY:-10}; MAX_TIME=${MAX_TIME:-00:04:30:00}
H_MATCH=${H_MATCH:-$(cat $DATA/h_match_frames 2>/dev/null || echo 40)}
RUN=$OUT/${VARIANT}_${SPREAD}_s${SEED}
LOG=$(scontrol show job $SLURM_JOB_ID -o | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p')
cd $WT
export PYTHONPATH=$WT HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=${WANDB_MODE:-offline}
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CC CXX LD 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== TRAIN $VARIANT $SPREAD s$SEED -> $RUN ($(date)) commit=$(git rev-parse --short HEAD) steps=$MAX_STEPS val_every=${VAL_EVERY}00 ==="
$PY egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  +experiment=e1/fold_${VARIANT} \
  e1.spread=$SPREAD e1.train_root=$DATA/$SPREAD e1.valid_root=$DATA/${VALID_SET:-test_mid} e1.h_match_frames=$H_MATCH \
  seed=$SEED trainer.max_epochs=$MAX_EPOCHS trainer.min_epochs=0 +trainer.max_time=$MAX_TIME \
  trainer.check_val_every_n_epoch=$VAL_EVERY callbacks.model_checkpoint.every_n_epochs=$VAL_EVERY callbacks.model_checkpoint.save_top_k=-1 \
  trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 \
  ${TRAIN_EXTRA:-} hydra.run.dir=$RUN/train
# --- pick the best-validation checkpoint from the in-training E1 reads ---------------------
$PY - "$LOG" "$RUN" "$VAL_EVERY" <<'PYEOF'
import glob, json, os, re, sys
log, run, val_every = sys.argv[1], sys.argv[2], int(sys.argv[3])
vals = [json.loads(l.split("E1_TEMPO_RESULT ", 1)[1]) for l in open(log) if l.startswith("E1_TEMPO_RESULT ")]
ckpts = sorted(glob.glob(f"{run}/train/checkpoints/epoch_epoch=*.ckpt"), key=lambda p: int(re.search(r"epoch=(\d+)", p).group(1)))
epochs = [int(re.search(r"epoch=(\d+)", p).group(1)) for p in ckpts]
print(f"vals={len(vals)} ckpts={epochs}")
if len(vals) != len(ckpts):
    print("WARNING: val count != checkpoint count; pairing by order of the min(len) prefix")
n = min(len(vals), len(ckpts))
e = [v["e_time"] for v in vals[:n]]
best = int(min(range(n), key=lambda i: e[i]))
out = {"steps": [(epochs[i] + 1) * 100 for i in range(n)], "e_time": e,
       "e_time_prog": [v.get("e_time_prog") for v in vals[:n]], "e_arc": [v["e_arc"] for v in vals[:n]], "d_clock": [v["d_clock"] for v in vals[:n]],
       "best_idx": best, "best_step": (epochs[best] + 1) * 100, "best_ckpt": ckpts[best], "final_ckpt": ckpts[n - 1]}
json.dump(out, open(f"{run}/best.json", "w"), indent=2)
print(f"BEST step={out['best_step']} e_time={e[best]:.4f} ckpt={ckpts[best]}; FINAL {ckpts[n-1]}")
PYEOF
BEST=$($PY -c "import json;print(json.load(open('$RUN/best.json'))['best_ckpt'])")
FINAL=$($PY -c "import json;print(json.load(open('$RUN/best.json'))['final_ckpt'])")
NORM=$(find $RUN/train -name 'norm_stats*.json' | head -1); NORM_DIR=$(dirname "$NORM")
echo "best=$BEST final=$FINAL norm=$NORM"
for TAG in best final; do
  CKPT=$BEST; [ "$TAG" = final ] && CKPT=$FINAL
  if [ "$TAG" = final ] && [ "$FINAL" = "$BEST" ]; then echo "final == best, skipping duplicate evals"; continue; fi
  for TS in ${TEST_SETS:-test_mid test_low test_high}; do
  echo "=== EVAL $TAG on $TS ($(date)) ==="
  $PY egomimic/trainHydra.py --config-name=train_zarr_cartesian \
    +experiment=e1/fold_${VARIANT} mode=eval "ckpt_path=\"$CKPT\"" \
    e1.spread=$SPREAD e1.train_root=$DATA/$SPREAD e1.valid_root=$DATA/$TS e1.h_match_frames=$H_MATCH e1.limit_val_batches=1.0 \
    seed=$SEED norm_stats.precomputed_norm_path=$NORM_DIR \
    trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 \
    ${EVAL_EXTRA:-} hydra.run.dir=$RUN/eval_${TAG}_$TS || echo "EVAL FAILED $TAG on $TS"
  done
done
# --- disposable intermediate checkpoints (quota) -----------------------------------------
for f in $RUN/train/checkpoints/epoch_epoch=*.ckpt; do [ "$f" = "$BEST" ] || [ "$f" = "$FINAL" ] || rm -f "$f"; done
echo "=== DONE ($(date)) ==="
