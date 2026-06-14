#!/bin/bash
#SBATCH --job-name=eval_B_arte
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/eval_B_arte_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/eval_B_arte_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
CKPT=${CKPT}
DESC=${DESC:-B_arTE}
# B is chunk_k=1: DON'T override the model's chunk_k (would mismatch the ckpt's head).
# evaluator.chunk_k=32 is the GENERATE horizon -> AR-generate a 32-step chunk each rollout
# step (obs broadcast across the horizon), then temporal-ensemble the overlapping chunks.
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=${DESC} mode=eval data=tsimulation \
  ckpt_path=${CKPT} \
  model=hnet_pushshapes_fused_windowed \
  ++model.robomimic_model.cond_encoder.img_encoders.front_img_1.spatial=true \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=chunk_te evaluator.temporal_ensemble=true evaluator.chunk_k=32 \
  trainer=debug trainer.limit_val_batches=4 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=400 evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
