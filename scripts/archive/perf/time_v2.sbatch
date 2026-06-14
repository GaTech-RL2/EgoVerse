#!/bin/bash
#SBATCH --job-name=time_v2
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/time_v2_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/time_v2_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
OBS=${OBS:-demo}; K=${K:-8}; MAXW=${MAXW:-256}
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=timev2_${OBS}_k${K} mode=train data=tsimulation \
  model=hnet_pushshapes_fused_windowed \
  ++model.robomimic_model.obs_source=${OBS} \
  ++model.robomimic_model.k_schedule="[[0,${K}]]" \
  ++model.robomimic_model.max_windows=${MAXW} \
  evaluator=eval_hnet_sim callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=100 \
  trainer=debug trainer.max_epochs=4 trainer.min_epochs=4 \
  trainer.limit_train_batches=8 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=100 \
  trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.train_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=null evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "TIME_EXIT=$?"
