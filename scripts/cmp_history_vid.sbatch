#!/bin/bash
#SBATCH --job-name=cmp_hist_vid
#SBATCH --time=02:00:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/cmp_hist_vid_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/cmp_hist_vid_%j.err
set -uxo pipefail
export PATH=/opt/slurm/Ubuntu-20.04/current/bin:$PATH
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval

# ---------------- LEFT: window-10 (best ckpt ep799, table mean 0.605) ----------------
RUNDIR_L=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8_nc3/bc_rnn_hnet_chunk8_2026-06-06_19-38-16
cp -n "${RUNDIR_L}/checkpoints/epoch_epoch=799.ckpt" "${RUNDIR_L}/checkpoints/cmphist_ep799.ckpt"
export ROLLOUT_VIDEO_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/cmp_history_left_win10
rm -f ${ROLLOUT_VIDEO_DIR}/roll_ep*.mp4 2>/dev/null || true
echo "================ LEFT window-10 ep799 ================"
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=cmphist_win10_ep799 description=cmp_history_left mode=eval data=tsimulation \
  ckpt_path=${RUNDIR_L}/checkpoints/cmphist_ep799.ckpt \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk8 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  +evaluator.max_videos=20 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=${RUNDIR_L}/norm_stats/norm_stats.json
echo "LEFT_EXIT=$?"
ls -la $ROLLOUT_VIDEO_DIR

# ---------------- RIGHT: full-history (best ckpt ep599, table mean 0.440) ----------------
RUNDIR_R=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetC8FHR_nc3/bc_rnn_hnet_chunk8_fullhist_ratio_2026-06-06_22-31-32
cp -n "${RUNDIR_R}/checkpoints/epoch_epoch=599.ckpt" "${RUNDIR_R}/checkpoints/cmphist_ep599.ckpt"
export ROLLOUT_VIDEO_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/cmp_history_right_fullhist
rm -f ${ROLLOUT_VIDEO_DIR}/roll_ep*.mp4 2>/dev/null || true
echo "================ RIGHT full-history ep599 ================"
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=cmphist_fullhist_ep599 description=cmp_history_right mode=eval data=tsimulation \
  ckpt_path=${RUNDIR_R}/checkpoints/cmphist_ep599.ckpt \
  model=bc_rnn_pushshapes_paperexact_hnet_chunk8_fullhist_ratio \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar \
  evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  +evaluator.max_videos=20 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.precision=32 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.norm_mode=minmax \
  norm_stats.precomputed_norm_path=${RUNDIR_R}/norm_stats/norm_stats.json
echo "RIGHT_EXIT=$?"
ls -la $ROLLOUT_VIDEO_DIR
echo "CMP_HISTORY_ALL_DONE"
