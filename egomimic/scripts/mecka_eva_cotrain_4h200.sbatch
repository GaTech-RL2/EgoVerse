#!/bin/bash
#SBATCH --job-name=mecka_eva_cotrain
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
#SBATCH --output=logs/mecka_eva_cotrain_wam/prod_4h200_%j.out
#SBATCH --error=logs/mecka_eva_cotrain_wam/prod_4h200_%j.err

set -euo pipefail
REPO_ROOT="/storage/project/r-dxu345-0/acheluva3/EgoVerse"
cd "${REPO_ROOT}"
mkdir -p logs/mecka_eva_cotrain_wam
export PATH="${REPO_ROOT}/emimic/bin:${PATH}"

# --- Val knobs (atomic unit: NUM_EPISODES) --------------------------------
# 3 full episodes per embodiment per val loop. VAL_CAM_HORIZON=97 → ~19.4s
# per sample (12 rolling steps × K=2). VAL_ACTION_HORIZON=582 lets the GT
# green overlay extend the full video duration. Model input action_horizon
# stays 192 via _prep_state_action's slice.
NUM_EPISODES="${NUM_EPISODES:-3}"
VAL_CAM_HORIZON="${VAL_CAM_HORIZON:-97}"
VAL_ACTION_HORIZON="${VAL_ACTION_HORIZON:-$((VAL_CAM_HORIZON * 6))}"

# 4×h200 DDP. Per-GPU physical batch: mecka bs=1 + eva bs=1 = 2/step.
# Effective batch = 4 × 2 × 4 grad_accum = 32 (matches the prior mecka+eva
# cotrain from 2026-08-04). Paper-aligned defaults from the yaml
# (state_horizon=4, num_state_per_block=1, state_stride=48). Norm-stats
# cache pairs mecka (emb 3) with eva (emb 6); h4 variant slices state stats
# from (8, D) to (4, D) to match state_horizon=4.
srun --unbuffered python egomimic/trainHydra.py \
  --config-name train_zarr_human_wam_wan22_5b \
  data=mecka_eva_cotrain_wam \
  trainer=ddp_wam \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.accumulate_grad_batches=4 \
  trainer.check_val_every_n_epoch=10 \
  trainer.limit_val_batches=${NUM_EPISODES} \
  evaluator/viz@evaluator.viz_func=wam_cartesian_aria_eva \
  name=mecka_eva_cotrain_wam \
  description=mecka_eva_cotrain_wan22_5b_bs1x2_4h200_gradaccum4_effbs32_paperv1_0827 \
  data.train_dataloader_params.human_bimanual.batch_size=1 \
  data.valid_dataloader_params.human_bimanual.batch_size=1 \
  data.train_dataloader_params.eva_bimanual.batch_size=1 \
  data.valid_dataloader_params.eva_bimanual.batch_size=1 \
  data.train_datasets.human_bimanual.resolver.key_map.cam_horizon=${VAL_CAM_HORIZON} \
  data.train_datasets.human_bimanual.resolver.key_map.action_horizon=${VAL_ACTION_HORIZON} \
  +data.train_datasets.human_bimanual.resolver.key_map.video_stride=6 \
  +data.train_datasets.human_bimanual.resolver.transform_list.video_stride=6 \
  data.train_datasets.eva_bimanual.resolver.key_map.cam_horizon=${VAL_CAM_HORIZON} \
  data.train_datasets.eva_bimanual.resolver.key_map.action_horizon=${VAL_ACTION_HORIZON} \
  +data.train_datasets.eva_bimanual.resolver.key_map.video_stride=6 \
  +data.train_datasets.eva_bimanual.resolver.transform_list.video_stride=6 \
  model.robomimic_model.num_video_frames=9 \
  model.robomimic_model.action_horizon=192 \
  model.robomimic_model.dit.num_action_per_block=48 \
  +model.robomimic_model.dit.num_frame_per_block=2 \
  model.robomimic_model.dit.lora=False \
  model.robomimic_model.dit.freeze_video=False \
  ~model.scheduler=null \
  'model.robomimic_model.domains=[human_bimanual,eva_bimanual]' \
  +model.robomimic_model.ac_keys.eva_bimanual=actions_cartesian \
  norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/norm_stats_cache/mecka_eva_wam_norm_stats_h4.json \
  launch_params.gpus_per_node=4 \
  launch_params.nodes=1 \
  mode=train
