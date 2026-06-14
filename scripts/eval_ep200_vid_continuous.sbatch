#!/bin/bash
#SBATCH --job-name=ep200vid
#SBATCH --time=01:10:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/ep200vid_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/ep200vid_%x_%j.err
# env knobs: RUNDIR (abs path to run dir), TAG (short name), NHIST (n_obs_history)
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export ROLLOUT_VIDEO_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/d${TAG}_ep200_videos
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
cp -n ${RUNDIR}/checkpoints/epoch_epoch=199.ckpt ${RUNDIR}/checkpoints/ep200.ckpt
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=${TAG}_ep200_vid description=${TAG}_ep200 mode=eval data=tsimulation \
  ckpt_path=${RUNDIR}/checkpoints/ep200.ckpt \
  model=hnet_pushshapes_fused_windowed_crossattn_resnet \
  ++model.robomimic_model.action_head=continuous \
  ++model.robomimic_model.chunk_k=32 ++model.robomimic_model.n_obs_history=${NHIST} \
  evaluator=eval_hnet_sim evaluator.rollout_mode=chunk_openloop \
  evaluator.temporal_ensemble=false evaluator.chunk_k=32 \
  evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.precomputed_norm_path=$NORM
echo "VID_EXIT=$?"
ls $ROLLOUT_VIDEO_DIR
