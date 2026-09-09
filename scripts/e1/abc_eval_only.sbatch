#!/bin/bash
#SBATCH --job-name=e1eval
#SBATCH --partition=coe-gpu
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/e1eval_%x_%j.log
# Eval-only backfill: sbatch --export=ALL,VARIANT=arcmean,SPREAD=train_nofast,SEED=43,RUN=<run dir>,TAG=final,TS=test_rand2,DATA=...,EXP=abc abc_eval_only.sbatch
set -euo pipefail
WT=${WT:-$HOME/scratch/EgoVerse/.claude/worktrees/arc-e1}; PY=${PY:-$HOME/scratch/EgoVerse/.venv/bin/python}
cd $WT; export PYTHONPATH=$WT HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline OMP_NUM_THREADS=1
H_MATCH=$(cat $DATA/h_match_frames); CKPT=$($PY -c "import json;print(json.load(open(\"$RUN/best.json\"))[\"${TAG}_ckpt\"])")
NORM_DIR=$(dirname $(find $RUN/train -name "norm_stats*.json" | head -1))
echo "=== EVAL $TAG on $TS ckpt=$CKPT ($(date)) ==="
$PY egomimic/trainHydra.py --config-name=train_zarr_cartesian +experiment=e1/${EXP}_${VARIANT} mode=eval "ckpt_path=\"$CKPT\"" \
  e1.spread=$SPREAD e1.train_root=$DATA/$SPREAD e1.valid_root=$DATA/$TS e1.h_match_frames=$H_MATCH e1.limit_val_batches=1.0 seed=$SEED norm_stats.precomputed_norm_path=$NORM_DIR \
  trainer.devices=1 trainer.num_nodes=1 launch_params.gpus_per_node=1 launch_params.nodes=1 hydra.run.dir=$RUN/eval_${TAG}_$TS
echo "=== DONE ($(date)) ==="
