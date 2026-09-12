#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=smoke_hpt_6d
#SBATCH --output=logs/smoke_hpt_6d_flagship_%j.log

# GPU smoke of the HPT 6D flagship recipe (train_zarr_mecka_flagship_6d_hpt):
# 2 episodes per split (resolver.debug caps the S3 sync too — the
# MultiDataset-level debug alone syncs the whole filter set first), batch 2,
# 2 train steps + 1 val batch through
# eval_hpt_wristframe_6d (revert + metrics + overlay viz) and the train_viz
# (seen-operator) head. Needs the norm job's output (or NORM_PATH). Passes
# when the log ends with "[smoke] exit=0" and the W&B/CSV logs carry both
# Valid/human_bimanual_actions_cartesian_cam_paired_mse_avg and
# train_viz/Valid/human_bimanual_actions_cartesian_cam_paired_mse_avg.
# Submit from the repo root (`mkdir -p logs` first).

set -euo pipefail
REPO="${REPO-$SLURM_SUBMIT_DIR}"
NORM_ROOT="${NORM_ROOT-$REPO/norm_stats}"
DATASET_DIR="${DATASET_DIR-/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/emimic/bin/python"
export PATH="$REPO/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NORM_PATH="${NORM_PATH-$NORM_ROOT/fold_flagship_opsplit_hpt_6d/norm_stats}"

echo "[smoke] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
srun "$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_mecka_flagship_6d_hpt \
  "paths.dataset_dir=$DATASET_DIR" \
  launch_params.gpus_per_node=1 \
  trainer.max_epochs=1 trainer.min_epochs=1 ++trainer.max_steps=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=1 \
  trainer.num_sanity_val_steps=0 \
  +data.train_datasets.human_bimanual.resolver.debug=2 \
  +data.train_datasets.human_bimanual.debug=2 +data.valid_datasets.human_bimanual.debug=2 +data.train_viz_datasets.human_bimanual.debug=2 \
  data.train_dataloader_params.human_bimanual.batch_size=2 data.valid_dataloader_params.human_bimanual.batch_size=2 data.train_viz_dataloader_params.human_bimanual.batch_size=2 \
  data.train_dataloader_params.human_bimanual.num_workers=2 data.valid_dataloader_params.human_bimanual.num_workers=2 data.train_viz_dataloader_params.human_bimanual.num_workers=1 \
  "norm_stats.precomputed_norm_path=$NORM_PATH" \
  evaluator.viz_max_batches=1 evaluator.viz_every_n_epochs=1 \
  train_viz_evaluator.limit_val_batches=1 train_viz_evaluator.base.viz_every_n_epochs=1 train_viz_evaluator.base.viz_max_batches=1 \
  name=smoke_hpt_6d_flagship description=gpu_2steps
echo "[smoke] exit=$?"
