#!/bin/bash
#SBATCH --job-name=bgmm_video
#SBATCH --time=00:40:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bgmm_video_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bgmm_video_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
CKPT=logs/hnet_baseline_nc3/chunkB_gmm_2026-06-04_13-38-52/checkpoints/b_ep400.ckpt
export SPLIT_DIAG_NEPS=20
export SPLIT_DIAG_CONT=8
export SPLIT_DIAG_SPLIT=128
export SPLIT_DIAG_DIR=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bgmm_video_ep400_b3
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=bgmm_video description=ep400 mode=eval data=tsimulation \
  ckpt_path=${CKPT} model=hnet_pushshapes_fused_windowed_resnet \
  +model.robomimic_model.action_head=gmm +model.robomimic_model.gmm_num_modes=5 \
  ++model.robomimic_model.chunk_k=32 ++model.robomimic_model.action_horizon=1024 \
  evaluator=eval_hnet_sim evaluator.rollout_mode=chunk_openloop \
  evaluator.temporal_ensemble=false evaluator.chunk_k=32 \
  evaluator.init_mode=replay evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  evaluator.limit_val_batches=3 \
  trainer=debug trainer.limit_val_batches=3 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.precomputed_norm_path=$NORM
echo "VIDEO_EXIT=$?"
