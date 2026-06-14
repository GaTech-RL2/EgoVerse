#!/bin/bash
#SBATCH --job-name=eval_B_tfdump
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/eval_B_tfdump_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/eval_B_tfdump_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
export TF_DUMP_DIR=${TF_DUMP_DIR:-/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/tf_dump_B}
export TF_DUMP_N=4
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
CKPT=${CKPT}
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=B_tfdump mode=eval data=tsimulation \
  ckpt_path=${CKPT} model=hnet_pushshapes_fused_windowed \
  ++model.robomimic_model.cond_encoder.img_encoders.front_img_1.spatial=true \
  evaluator=eval_hnet_sim evaluator.rollout_mode=ar evaluator.temporal_ensemble=false \
  evaluator.init_mode=replay \
  evaluator.max_steps=8 evaluator.coverage_threshold=0.8 \
  trainer=debug trainer.limit_val_batches=4 trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  norm_stats.precomputed_norm_path=$NORM
echo "EVAL_EXIT=$?"
