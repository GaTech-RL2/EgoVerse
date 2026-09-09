#!/bin/bash
#SBATCH --job-name=e1abc
#SBATCH --partition=coe-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=10:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/e1abc_%x_%j.log
# E1 cell on the ABC / YAM skirts episodes (experiment/e1/abc_<variant>): same launcher as the tempo ablation —
#   sbatch --job-name=e1abc_${VARIANT}_${SPREAD}_s${SEED} --export=ALL,VARIANT=arcmean_D75,SPREAD=full,SEED=42 abc_e1.sbatch
# VARIANT is the experiment suffix: time | arcmean | arcvel | arclogdur | arcmean_D75 | arclogdur_D75 | *_fs.
# Validation reads every VAL_EVERY epochs on VALID_SET (val_e1 = the 60 carved episodes; test_mid was the old,
# leaky default). The best checkpoint is picked on SELECT (paired_mse = the repo's all-actions MSE; e_time
# for logs that predate it) AND the final checkpoint; both are evaluated on the three tempo tertiles.
# A pruner runs alongside training so a live cell only ever holds the current best + newest checkpoints
# (~1 GB): ten concurrent cells holding six 0.47 GB checkpoints each is what blew the 300 GB quota on 09-08.
set -euo pipefail
WT=${WT:-$HOME/scratch/EgoVerse/.claude/worktrees/arc-e1}
PY=${PY:-$HOME/scratch/EgoVerse/.venv/bin/python}
DATA=${DATA:-$HOME/scratch/data/e1_abc}
OUT=${OUT:-$HOME/scratch/runs/e1_abc}
VARIANT=${VARIANT:?time|arcmean|arcvel|arclogdur}; SPREAD=${SPREAD:?narrow|medium|full}; SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-10000}; MAX_EPOCHS=$(( MAX_STEPS / 100 )); VAL_EVERY=${VAL_EVERY:-10}; MAX_TIME=${MAX_TIME:-00:04:30:00}
VALID_SET=${VALID_SET:-val_e1}; SELECT=${SELECT:-paired_mse}; LIMIT_VAL=${LIMIT_VAL:-60}
H_MATCH=${H_MATCH:-$(cat $DATA/h_match_frames 2>/dev/null || echo 40)}
RUN=$OUT/${VARIANT}_${SPREAD}_s${SEED}
LOG=$(scontrol show job $SLURM_JOB_ID -o | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p')
cd $WT
export PYTHONPATH=$WT HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=${WANDB_MODE:-offline}
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CC CXX LD 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== TRAIN $VARIANT $SPREAD s$SEED -> $RUN ($(date)) commit=$(git rev-parse --short HEAD) steps=$MAX_STEPS val_every=${VAL_EVERY}00 valid=$VALID_SET select=$SELECT ==="
# --- shared picker: pairs the i-th E1_TEMPO_RESULT line with epoch (i+1)*VAL_EVERY-1 --------------------
mkdir -p $RUN
cat > $RUN.pick.py <<'PYEOF'
import glob, json, os, re, sys
log, run, val_every, select, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
vals = [json.loads(l.split("E1_TEMPO_RESULT ", 1)[1]) for l in open(log) if l.startswith("E1_TEMPO_RESULT ")]
if not vals:
    sys.exit(0)
key = select if select in vals[0] else "e_time"
e = [v[key] for v in vals]
epochs = [(i + 1) * val_every - 1 for i in range(len(vals))]
best = int(min(range(len(e)), key=lambda i: e[i]))
ck = lambda ep: f"{run}/train/checkpoints/epoch_epoch={ep}.ckpt"
if mode == "prune":
    # keep the current best and the newest; everything older than the newest is complete on disk
    keep = {ck(epochs[best]), ck(epochs[-1])}
    for p in glob.glob(f"{run}/train/checkpoints/epoch_epoch=*.ckpt"):
        ep = int(re.search(r"epoch=(\d+)", p).group(1))
        if p not in keep and ep < epochs[-1]:
            os.remove(p); print(f"pruned {os.path.basename(p)}", flush=True)
    sys.exit(0)
have = sorted(glob.glob(f"{run}/train/checkpoints/epoch_epoch=*.ckpt"), key=lambda p: int(re.search(r"epoch=(\d+)", p).group(1)))
final = have[-1]
best_ckpt = ck(epochs[best]) if os.path.exists(ck(epochs[best])) else final
out = {"steps": [(ep + 1) * 100 for ep in epochs], "select": key, "selected": e,
       "e_time": [v["e_time"] for v in vals], "e_time_prog": [v.get("e_time_prog") for v in vals],
       "e_arc": [v["e_arc"] for v in vals], "d_clock": [v["d_clock"] for v in vals],
       "paired_mse": [v.get("paired_mse") for v in vals], "xyz_mse": [v.get("xyz_mse") for v in vals],
       "best_idx": best, "best_step": (epochs[best] + 1) * 100, "best_ckpt": best_ckpt, "final_ckpt": final}
json.dump(out, open(f"{run}/best.json", "w"), indent=2)
print(f"BEST({key}) step={out['best_step']} {key}={e[best]:.5f} ckpt={best_ckpt}; FINAL {final}")
PYEOF
( while sleep 600; do $PY $RUN.pick.py "$LOG" "$RUN" "$VAL_EVERY" "$SELECT" prune 2>/dev/null || true; done ) &
PRUNER=$!
$PY egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  +experiment=e1/${EXP:-abc}_${VARIANT} \
  e1.spread=$SPREAD e1.train_root=$DATA/$SPREAD e1.valid_root=$DATA/$VALID_SET e1.h_match_frames=$H_MATCH e1.limit_val_batches=$LIMIT_VAL \
  seed=$SEED trainer.max_epochs=$MAX_EPOCHS trainer.min_epochs=0 +trainer.max_time=$MAX_TIME \
  trainer.check_val_every_n_epoch=$VAL_EVERY callbacks.model_checkpoint.every_n_epochs=$VAL_EVERY callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.save_last=false \
  trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 \
  ${TRAIN_EXTRA:-} hydra.run.dir=$RUN/train
kill $PRUNER 2>/dev/null || true
# --- pick the best-validation checkpoint from the in-training E1 reads ---------------------
$PY $RUN.pick.py "$LOG" "$RUN" "$VAL_EVERY" "$SELECT" pick
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
    +experiment=e1/${EXP:-abc}_${VARIANT} mode=eval "ckpt_path=\"$CKPT\"" \
    e1.spread=$SPREAD e1.train_root=$DATA/$SPREAD e1.valid_root=$DATA/$TS e1.h_match_frames=$H_MATCH e1.limit_val_batches=1.0 \
    seed=$SEED norm_stats.precomputed_norm_path=$NORM_DIR \
    trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 \
    ${EVAL_EXTRA:-} hydra.run.dir=$RUN/eval_${TAG}_$TS || echo "EVAL FAILED $TAG on $TS"
  done
done
# --- disposable intermediate checkpoints (quota) -----------------------------------------
for f in $RUN/train/checkpoints/*.ckpt; do [ "$f" = "$BEST" ] || [ "$f" = "$FINAL" ] || rm -f "$f"; done
echo "=== DONE ($(date)) ==="
