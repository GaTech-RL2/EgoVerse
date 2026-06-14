#!/bin/bash
#SBATCH --job-name=D_ep200_te
#SBATCH --time=00:50:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/D_ep200_te_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/D_ep200_te_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export ROLLOUT_VIDEO_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/dD_ep200_te_videos
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
CKPT=logs/hnet_baseline_nc3/chunkD_crossattn_gmm_2026-06-04_20-58-29/checkpoints/d_ep200.ckpt
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=D_ep200_te description=te mode=eval data=tsimulation \
  ckpt_path=${CKPT} model=hnet_pushshapes_fused_windowed_crossattn_resnet \
  ++model.robomimic_model.action_head=gmm ++model.robomimic_model.gmm_num_modes=5 \
  ++model.robomimic_model.chunk_k=32 ++model.robomimic_model.n_obs_history=1 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=chunk_te \
  evaluator.temporal_ensemble=true evaluator.chunk_k=32 \
  evaluator.init_mode=seeds evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  evaluator.limit_val_batches=1 trainer.limit_val_batches=1 \
  trainer=debug trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.precomputed_norm_path=$NORM
echo "D_TE_EXIT=$?"
