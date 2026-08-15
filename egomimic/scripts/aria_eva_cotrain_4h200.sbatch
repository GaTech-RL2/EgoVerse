#!/bin/bash
#SBATCH --job-name=aria_eva_cotrain
#SBATCH --partition=gpu-h200
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:4
#SBATCH --mem-per-gpu=250G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/aria_eva_cotrain_wam/prod_4h200_%j.out
#SBATCH --error=logs/aria_eva_cotrain_wam/prod_4h200_%j.err

set -euo pipefail
REPO_ROOT="/storage/project/r-dxu345-0/acheluva3/EgoVerse"
cd "${REPO_ROOT}"
mkdir -p logs/aria_eva_cotrain_wam
export PATH="${REPO_ROOT}/emimic/bin:${PATH}"

# 4×h200 DDP: srun spawns 4 ranks (one per GPU); Lightning's SLURMEnvironment
# reads SLURM_PROCID and configures DDP. Per-GPU physical batch:
#   - human_bimanual bs=1 + eva_bimanual bs=1 = 2 samples/step (half of aria
#     production; 2 embodiments' concurrent activations already push memory
#     past what aria bs=4 single-embodiment used).
# Effective batch = 4 GPUs × 2 samples × 4 grad_accum = 32 (matches production).
srun --unbuffered python egomimic/trainHydra.py \
  --config-name train_zarr_human_wam_wan22_5b \
  data=aria_eva_cotrain_wam \
  trainer=ddp_wam \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.accumulate_grad_batches=4 \
  trainer.check_val_every_n_epoch=10 \
  trainer.limit_val_batches=1 \
  evaluator/viz@evaluator.viz_func=wam_cartesian_aria_eva \
  name=aria_eva_cotrain_wam \
  description=aria_eva_cotrain_wan22_5b_bs1x2_4h200_gradaccum4_effbs32_vizfix_0802 \
  data.train_dataloader_params.human_bimanual.batch_size=1 \
  data.valid_dataloader_params.human_bimanual.batch_size=1 \
  data.train_dataloader_params.eva_bimanual.batch_size=1 \
  data.valid_dataloader_params.eva_bimanual.batch_size=1 \
  data.train_datasets.human_bimanual.resolver.key_map.cam_horizon=33 \
  data.train_datasets.human_bimanual.resolver.key_map.action_horizon=192 \
  data.train_datasets.human_bimanual.resolver.key_map.state_horizon=8 \
  +data.train_datasets.human_bimanual.resolver.key_map.video_stride=6 \
  +data.train_datasets.human_bimanual.resolver.transform_list.video_stride=6 \
  data.train_datasets.eva_bimanual.resolver.key_map.cam_horizon=33 \
  data.train_datasets.eva_bimanual.resolver.key_map.action_horizon=192 \
  data.train_datasets.eva_bimanual.resolver.key_map.state_horizon=8 \
  +data.train_datasets.eva_bimanual.resolver.key_map.video_stride=6 \
  +data.train_datasets.eva_bimanual.resolver.transform_list.video_stride=6 \
  model.robomimic_model.num_video_frames=9 \
  model.robomimic_model.action_horizon=192 \
  model.robomimic_model.dit.num_action_per_block=48 \
  +model.robomimic_model.dit.num_frame_per_block=2 \
  model.robomimic_model.dit.num_state_per_block=2 \
  model.robomimic_model.dit.lora=False \
  model.robomimic_model.dit.freeze_video=False \
  ~model.scheduler=null \
  'model.robomimic_model.domains=[human_bimanual,eva_bimanual]' \
  +model.robomimic_model.ac_keys.eva_bimanual=actions_cartesian \
  norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/norm_stats_cache/aria_eva_wam_norm_stats.json \
  launch_params.gpus_per_node=4 \
  launch_params.nodes=1 \
  mode=train
