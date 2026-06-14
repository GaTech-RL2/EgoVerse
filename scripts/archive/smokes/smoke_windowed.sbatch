#!/bin/bash
#SBATCH --job-name=smk_windowed
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smk_windowed_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smk_windowed_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200
NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hnet_smoke/fused_nochunk_nc3_5ep_2026-05-30_23-34-47/norm_stats/norm_stats.json
srun python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  name=hnet_baseline_nc3 description=smk_windowed_demo mode=train data=tsimulation \
  model=hnet_pushshapes_fused_windowed \
  evaluator=eval_hnet_sim callbacks=checkpoints callbacks.model_checkpoint.every_n_epochs=10 \
  trainer=debug trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=1 trainer.check_val_every_n_epoch=10 \
  trainer.profiler=null logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=null evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "SMOKE_EXIT=$?"
