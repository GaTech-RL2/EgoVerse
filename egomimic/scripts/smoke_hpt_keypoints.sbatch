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
#SBATCH --job-name=smoke_hpt_kp
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/smoke_hpt_keypoints_%j.log

# GPU smoke of the HPT keypoint recipe (train_zarr_mecka_kp_wrist_hpt): 2
# episodes per split, batch 2, 2 train steps + 1 val batch through
# eval_hpt_keypoints_wrist (revert + metrics + overlay viz) and the train_viz
# head, using the pi twin's precomputed norm stats (exercises the provenance
# check). Passes when the log ends with "[smoke] exit=0" and W&B/CSV logs
# carry Valid/human_bimanual_actions_keypoints_cam_paired_mse_avg.

set -euo pipefail
WT="${WT-/storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/kp-wristframe-port}"
cd "$WT"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_opsplit_kp/norm_stats}"

echo "[smoke] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
srun "$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_mecka_kp_wrist_hpt \
  launch_params.gpus_per_node=1 \
  trainer.max_epochs=1 trainer.min_epochs=1 ++trainer.max_steps=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=1 \
  trainer.num_sanity_val_steps=0 \
  model.robomimic_model.reverse_kl_samples=1 \
  +data.train_datasets.human_bimanual.debug=2 +data.valid_datasets.human_bimanual.debug=2 +data.train_viz_datasets.human_bimanual.debug=2 \
  data.train_dataloader_params.human_bimanual.batch_size=2 data.valid_dataloader_params.human_bimanual.batch_size=2 data.train_viz_dataloader_params.human_bimanual.batch_size=2 \
  data.train_dataloader_params.human_bimanual.num_workers=2 data.valid_dataloader_params.human_bimanual.num_workers=2 data.train_viz_dataloader_params.human_bimanual.num_workers=1 \
  "norm_stats.precomputed_norm_path=$NORM_PATH" \
  evaluator.viz_max_batches=1 evaluator.viz_every_n_epochs=1 train_viz_evaluator.limit_val_batches=1 \
  name=smoke_hpt_keypoints description=gpu_2steps \
  'hydra.run.dir=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/${name}/${description}_${now:%Y-%m-%d_%H-%M-%S}'
echo "[smoke] exit=$?"
