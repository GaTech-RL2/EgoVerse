#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH -J norm_sortpp_subtask
#SBATCH -o %x_%j.out

# Compute FULL norm stats (sample_frac=1.0) over the entire sort_pp_subtask_6d
# dataset on a single h200, cache to a fixed dir, then exit after 1 step (the
# norm json is written during datamodule setup, before the step). The main
# 4xh200 run then consumes it via norm_stats.precomputed_norm_path.
#   sbatch run_norm_sortpp_subtask.sbatch

cd /storage/project/r-dxu345-0/agao81/EgoVerse
set -a; source ./apikey.txt; set +a
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
VENV=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
WEIGHTS=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
NORMDIR=/storage/project/r-dxu345-0/agao81/EgoVerse/norm_cache/sortpp_subtask_full
mkdir -p "$NORMDIR"

srun $VENV egomimic/trainHydra.py --config-name train_zarr_cartesian_pi \
  model=pi0.5_subtask data=sort_pp_subtask_6d evaluator=eval_pi_wristframe_6d \
  launch_params.gpus_per_node=1 \
  model.robomimic_model.config.pytorch_weight_path="$WEIGHTS" \
  norm_stats.sample_frac=1.0 \
  norm_stats.num_workers=8 \
  norm_stats.save_cache_dir="$NORMDIR" \
  ++trainer.max_steps=1 ++trainer.limit_train_batches=1 ++trainer.limit_val_batches=0 \
  +data.train_dataloader_params.eva_bimanual.timeout=600 \
  +data.train_dataloader_params.human_bimanual.timeout=600 \
  name=norm_sortpp_subtask description="full norm stats"
