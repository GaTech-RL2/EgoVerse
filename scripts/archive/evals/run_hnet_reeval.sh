#!/bin/bash
#SBATCH --job-name=hnet-reeval
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hnet_reeval_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/hnet_reeval_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=.
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
CKPT=${CKPT_OVERRIDE}
MODEL=${MODEL_OVERRIDE:-hnet_pushshapes_fused}
NORM=${NORM_OVERRIDE}
RMODE=${ROLLOUT_MODE:-chunk_te}
CHUNKK=${CHUNK_K:-32}
DESC=${DESC_OVERRIDE:-hnet_reeval}
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=hnet_reeval description=${DESC} \
  mode=eval \
  model=${MODEL} \
  +model.robomimic_model.cond_encoder.img_encoders.front_img_1.spatial=true \
  data=tsimulation \
  evaluator=eval_hnet_sim \
  evaluator.rollout_mode=${RMODE} \
  evaluator.temporal_ensemble=true \
  evaluator.chunk_k=${CHUNKK} \
  ckpt_path=${CKPT} \
  trainer=debug \
  trainer.devices=1 \
  trainer.limit_val_batches=4 \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=null \
  evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "EXIT_CODE=$?"
