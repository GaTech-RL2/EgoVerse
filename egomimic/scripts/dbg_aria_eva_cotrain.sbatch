#!/bin/bash
#SBATCH --job-name=dbg_cotrain
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-dxu345-rl2
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=180G
#SBATCH --time=00:60:00
#SBATCH --output=logs/aria_eva_cotrain_wam/dbg_1gpu_%j.out
#SBATCH --error=logs/aria_eva_cotrain_wam/dbg_1gpu_%j.err

set -euo pipefail
REPO_ROOT="/storage/project/r-dxu345-0/acheluva3/EgoVerse"
cd "${REPO_ROOT}"
mkdir -p logs/aria_eva_cotrain_wam
export PATH="${REPO_ROOT}/emimic/bin:${PATH}"

# --- Val knobs (atomic unit: NUM_EPISODES) ---------------------------------
# NUM_EPISODES = full episodes per embodiment per val loop. With bs=1 and
# limit_val_batches=NUM_EPISODES the val loop yields exactly NUM_EPISODES
# samples per embodiment → 2 mp4s each (predicted + validation) = 2×NUM_EPISODES
# mp4s per embodiment per epoch. VAL_CAM_HORIZON sets per-sample coverage
# (K=2 rolling requires (cam_horizon-1)%4==0 AND (F_gt-1) odd → cam_horizon
# in {33, 97, 177, 361, ...}). 97 ≈ 19.4s / sample; 177 ≈ 35.4s (covers max
# eva 35.6s); 361 ≈ 72.2s.
NUM_EPISODES="${NUM_EPISODES:-3}"
VAL_CAM_HORIZON="${VAL_CAM_HORIZON:-97}"
# GT actions read horizon = VAL_CAM_HORIZON * video_stride (=6) so the green GT
# overlay extends across the full val video duration. Model input action_horizon
# stays at 192 via _prep_state_action's slice — this only bumps what the DATASET
# reads. norm_stats (192, D) tile block-cyclically inside _apply_norm_one /
# _apply_unnorm_one so 582 vs 192 shape mismatches don't crash.
VAL_ACTION_HORIZON="${VAL_ACTION_HORIZON:-$((VAL_CAM_HORIZON * 6))}"

# Direct (non-multirun) launch: hydra/launcher=basic runs train() locally on
# this h100 node instead of submitting another slurm job. Small limits to
# exercise the whole pipeline (2 embodiments in CombinedLoader) in ~minutes.
python egomimic/trainHydra.py \
  --config-name train_zarr_human_wam_wan22_5b \
  evaluator/viz@evaluator.viz_func=wam_cartesian_aria_eva \
  data=aria_eva_cotrain_wam \
  trainer=ddp_wam \
  trainer.strategy=auto \
  trainer.max_epochs=1 \
  trainer.min_epochs=1 \
  trainer.check_val_every_n_epoch=1 \
  trainer.limit_train_batches=3 \
  trainer.limit_val_batches=${NUM_EPISODES} \
  name=aria_eva_cotrain_wam \
  description=aria_eva_cotrain_dbg_1gpu_0730 \
  data.train_dataloader_params.human_bimanual.batch_size=1 \
  data.valid_dataloader_params.human_bimanual.batch_size=1 \
  data.train_dataloader_params.eva_bimanual.batch_size=1 \
  data.valid_dataloader_params.eva_bimanual.batch_size=1 \
  data.train_dataloader_params.human_bimanual.num_workers=2 \
  data.valid_dataloader_params.human_bimanual.num_workers=2 \
  data.train_dataloader_params.eva_bimanual.num_workers=2 \
  data.valid_dataloader_params.eva_bimanual.num_workers=2 \
  data.train_datasets.human_bimanual.resolver.key_map.cam_horizon=${VAL_CAM_HORIZON} \
  data.train_datasets.eva_bimanual.resolver.key_map.cam_horizon=${VAL_CAM_HORIZON} \
  data.train_datasets.human_bimanual.resolver.key_map.action_horizon=${VAL_ACTION_HORIZON} \
  data.train_datasets.eva_bimanual.resolver.key_map.action_horizon=${VAL_ACTION_HORIZON} \
  +data.train_datasets.human_bimanual.resolver.key_map.video_stride=6 \
  +data.train_datasets.human_bimanual.resolver.transform_list.video_stride=6 \
  +data.train_datasets.eva_bimanual.resolver.key_map.video_stride=6 \
  +data.train_datasets.eva_bimanual.resolver.transform_list.video_stride=6 \
  model.robomimic_model.num_video_frames=9 \
  model.robomimic_model.action_horizon=192 \
  model.robomimic_model.dit.num_action_per_block=48 \
  +model.robomimic_model.dit.num_frame_per_block=2 \
  model.robomimic_model.dit.num_state_per_block=1 \
  model.robomimic_model.dit.lora=True \
  model.robomimic_model.dit.freeze_video=True \
  ~model.scheduler=null \
  'model.robomimic_model.domains=[human_bimanual,eva_bimanual]' \
  +model.robomimic_model.ac_keys.eva_bimanual=actions_cartesian \
  norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/norm_stats_cache/aria_eva_wam_norm_stats_h4.json \
  launch_params.gpus_per_node=1 \
  launch_params.nodes=1 \
  mode=train
