#!/bin/bash
#SBATCH --job-name=mecka_eva_cotrain_r
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

# Resume the mecka+eva cotrain that TIMED OUT at 2026-08-06T23:23 (2-day sbatch
# --time limit hit at epoch ~799). Loads model+optimizer+scheduler+step state
# from ``last.ckpt`` via Lightning's ``trainer.fit(ckpt_path=..., weights_only=False)``.
# Writes NEW checkpoints + logs into the ORIGINAL sweep dir (via ``hydra.run.dir``
# override) so ``checkpoints/last.ckpt`` keeps overwriting the same file and
# the epoch counter continues from 799 instead of restarting at 0.
ORIG_SWEEP=/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/mecka_eva_cotrain_wam/mecka_eva_cotrain_wan22_5b_bs1x2_4h200_gradaccum4_effbs32_0804_2026-08-04_23-24-18
# Use the ORIGINAL clean ckpt (11672215, epoch 799 mtime 08-06) — NOT the
# ``-v1``/``-v2`` variants written by the earlier resumes 11742195 / 11789754.
# Those v-suffixed ckpts had global_step values from a restarted Lightning
# counter (load_weights_only zeroed it), so their internal step is low. Resuming
# from them + adding a step-offset from the ckpt's own low value would still
# overlap the pre-restart wandb history. This clean ckpt's global_step matches
# the wandb history's high-water mark.
CKPT=${ORIG_SWEEP}/checkpoints/epoch_799.ckpt  # symlink → epoch_epoch=799.ckpt (dodges Hydra ``=`` parser)

# wandb resume: user moved the run to project ``wam`` with id matching the
# ORIGINAL run name (``{name}_{description}_{original_timestamp}``). Passing
# id explicitly + resume=allow makes Lightning's WandbLogger APPEND to that
# run instead of starting a fresh one keyed by ``now:%Y-%m-%d_%H-%M-%S``.
WANDB_RUN_ID="mecka_eva_cotrain_wam_mecka_eva_cotrain_wan22_5b_bs1x2_4h200_gradaccum4_effbs32_0804_2026-08-04_23-24-18"

srun --unbuffered python egomimic/trainHydra.py \
  --config-name train_zarr_human_wam_wan22_5b \
  hydra.run.dir=${ORIG_SWEEP} \
  ckpt_path=${CKPT} \
  +load_weights_only=true \
  logger.wandb.project=wam \
  logger.wandb.id=${WANDB_RUN_ID} \
  +logger.wandb.resume=allow \
  data=mecka_eva_cotrain_wam \
  trainer=ddp_wam \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.accumulate_grad_batches=4 \
  trainer.check_val_every_n_epoch=10 \
  trainer.limit_val_batches=1 \
  evaluator/viz@evaluator.viz_func=wam_cartesian_aria_eva \
  name=mecka_eva_cotrain_wam \
  description=mecka_eva_cotrain_wan22_5b_bs1x2_4h200_gradaccum4_effbs32_0804 \
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
  norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/norm_stats_cache/mecka_eva_wam_norm_stats.json \
  launch_params.gpus_per_node=4 \
  launch_params.nodes=1 \
  mode=train
