#!/bin/bash
#SBATCH --job-name=overfit_hpt_kp
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/coc/flash7/rco3/EgoVerse2/logs/overfit_hpt_kp_freeform_%j.log

# SKYNET single-episode overfit test of the HPT KEYPOINT freeform recipe
# (train_zarr_mecka_kp_wrist_hpt, 138-D wrist-frame MANO keypoint action):
# the FIRST freeform train-filter episode is used as train, train_viz AND
# valid (valid filters overridden to the train filters), so all heads score
# the trained episode. bs 8, 20 batches/epoch x 40 epochs = 800 steps; val
# every 5 epochs; viz at epochs 20/40 (1 batch). Norm stats are computed on
# the fly from that one episode (precomputed_norm_path=null) and cached in
# the run dir. Passes when Train/action_loss and both
# Valid/human_bimanual_actions_keypoints_{paired,cam_paired}_mse_avg and their
# train_viz/ twins fall over the run, and the overlay video shows hand skeletons.
#
# Data: the episode is synced from S3 (s5cmd, R2 creds from ~/.egoverse_env)
# into DATA_DIR on first use (resolver.debug=1 limits the sync to that one
# episode). The parent data config hard-codes the PACE mirror as folder_path,
# so the resolver override points it at the Skynet mirror. The freeform
# parent already uses mode: total for train/train_viz, so no split overrides
# are needed there.

set -euo pipefail
case "$(hostname)" in sky1*|sky2*) echo "ABORT: on login node"; exit 1;; esac
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

REPO=/coc/flash7/rco3/EgoVerse2
DATA_DIR="${DATA_DIR-/coc/flash7/scratch/egoverseS3ZarrDataset}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/emimic/bin/python"
export PATH="$REPO/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID}"
mkdir -p "$DATA_DIR"

echo "[overfit] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) data=$DATA_DIR"
srun "$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_mecka_kp_wrist_hpt \
  launch_params.gpus_per_node=1 \
  "paths.dataset_dir=$DATA_DIR" \
  "data.train_datasets.human_bimanual.resolver.folder_path=$DATA_DIR" \
  +data.train_datasets.human_bimanual.resolver.debug=1 \
  +data.train_datasets.human_bimanual.debug=1 +data.valid_datasets.human_bimanual.debug=1 +data.train_viz_datasets.human_bimanual.debug=1 \
  'data.valid_datasets.human_bimanual.filters=${data.train_datasets.human_bimanual.filters}' \
  data.train_dataloader_params.human_bimanual.batch_size=8 data.valid_dataloader_params.human_bimanual.batch_size=8 data.train_viz_dataloader_params.human_bimanual.batch_size=8 \
  data.train_dataloader_params.human_bimanual.num_workers=2 data.valid_dataloader_params.human_bimanual.num_workers=2 data.train_viz_dataloader_params.human_bimanual.num_workers=1 \
  norm_stats.precomputed_norm_path=null norm_stats.num_workers=2 \
  trainer.max_epochs=40 trainer.limit_train_batches=20 trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=5 trainer.num_sanity_val_steps=0 \
  evaluator.viz_every_n_epochs=20 evaluator.viz_max_batches=1 \
  train_viz_evaluator.limit_val_batches=2 train_viz_evaluator.base.viz_every_n_epochs=20 train_viz_evaluator.base.viz_max_batches=1 \
  callbacks.model_checkpoint.every_n_epochs=40 \
  logger.wandb.offline=true \
  +logger.csv._target_=lightning.pytorch.loggers.CSVLogger '+logger.csv.save_dir=${paths.output_dir}' +logger.csv.name=csv \
  name=overfit_hpt_kp_freeform description=1ep_bs8_800steps
echo "[overfit] exit=$?"
