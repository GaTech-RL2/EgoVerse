#!/bin/bash
#SBATCH --job-name=e1fold
#SBATCH --partition=coe-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/e1fold_%x_%j.log
# Usage: sbatch --export=ALL,VARIANT=arcmean,SPREAD=narrow,SEED=42[,MAX_STEPS=30000] fold_train_eval.sbatch
# Trains one E1 cell on the mecka freeform fold split, then evaluates it on the
# held-out tempo tertiles in the same allocation.
set -euo pipefail
WT=${WT:-$HOME/scratch/EgoVerse/.claude/worktrees/arc-e1}
PY=${PY:-$HOME/scratch/EgoVerse/.venv/bin/python}
DATA=${DATA:-$HOME/scratch/data/e1_fold}
OUT=${OUT:-$HOME/scratch/runs/e1_fold}
VARIANT=${VARIANT:?time|arcmean|arcvel}; SPREAD=${SPREAD:?narrow|medium|full}; SEED=${SEED:-42}
# trainer/base.yaml fixes limit_train_batches=100, so an epoch is 100 optimizer steps.
MAX_STEPS=${MAX_STEPS:-30000}; MAX_EPOCHS=$(( MAX_STEPS / 100 )); MAX_TIME=${MAX_TIME:-00:09:00:00}
until [ -f $DATA/manifest.json ] && [ -d $DATA/$SPREAD ]; do echo "waiting for $DATA/manifest.json ($(date))"; sleep 60; done
H_MATCH=${H_MATCH:-$(cat $DATA/h_match_frames 2>/dev/null || echo 40)}
RUN=$OUT/${VARIANT}_${SPREAD}_s${SEED}
cd $WT
export PYTHONPATH=$WT HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=${WANDB_MODE:-online}
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID
unset CC CXX LD 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== TRAIN $VARIANT $SPREAD s$SEED -> $RUN ($(date)) commit=$(git rev-parse --short HEAD) ==="
$PY egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  +experiment=e1/fold_${VARIANT} \
  e1.spread=$SPREAD e1.train_root=$DATA/$SPREAD e1.valid_root=$DATA/test_mid e1.h_match_frames=$H_MATCH \
  seed=$SEED trainer.max_epochs=$MAX_EPOCHS trainer.min_epochs=0 +trainer.max_time=$MAX_TIME \
  trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 \
  ${TRAIN_EXTRA:-} hydra.run.dir=$RUN/train
CKPT=$(ls -t $RUN/train/checkpoints/*.ckpt | grep -v last | head -1 || true); [ -n "$CKPT" ] || CKPT=$RUN/train/checkpoints/last.ckpt
NORM=$(find $RUN/train -name 'norm_stats*.json' | head -1); NORM_DIR=$(dirname "$NORM")
echo "ckpt=$CKPT norm=$NORM"
for TS in ${TEST_SETS:-test_mid test_low test_high}; do
echo "=== EVAL on $TS ($(date)) ==="
$PY egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  +experiment=e1/fold_${VARIANT} mode=eval "ckpt_path=\"$CKPT\"" \
  e1.spread=$SPREAD e1.train_root=$DATA/$SPREAD e1.valid_root=$DATA/$TS e1.h_match_frames=$H_MATCH e1.limit_val_batches=1.0 \
  seed=$SEED norm_stats.precomputed_norm_path=$NORM_DIR \
  trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 \
  ${EVAL_EXTRA:-} hydra.run.dir=$RUN/eval_$TS || echo "EVAL FAILED on $TS"
done
echo "=== DONE ($(date)) ==="
