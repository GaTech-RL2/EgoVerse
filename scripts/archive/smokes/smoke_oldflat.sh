#!/bin/bash
#SBATCH --job-name=smoke-oldflat
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --time=0:20:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smoke_oldflat_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smoke_oldflat_%j.err
set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
mkdir -p logs/sbatch
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
hostname

export PYTHONPATH=.
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export PACK_COLLATE_MAX_TOTAL_FRAMES=3200

NC3=/coc/flash7/paphiwetsa3/datasets/new_circle_3
KM=egomimic.rldb.embodiment.pushshapes.get_keymap_eval
NORM=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/reeval_corrected/re_hptfull_500ep_80ep_2026-06-01_03-02-50/norm_stats/norm_stats.json

# OLD RUN PATH: base flat config, NO backbone override at all.
srun --kill-on-bad-exit=1 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=smoke_oldflat description=oldflat_2ep \
  mode=train \
  data=tsimulation \
  model=hnet_pushshapes_chunktoken \
  model.scheduler.max_steps=16 \
  evaluator=eval_hnet_sim \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=100 \
  trainer=debug \
  trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=8 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=100 \
  trainer.profiler=null \
  logger=wandb \
  data.train_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=$NC3 \
  data.valid_datasets.pushshapes_sim.resolver.key_map._target_=$KM \
  evaluator.max_steps=null \
  evaluator.coverage_threshold=0.8 \
  norm_stats.precomputed_norm_path=$NORM
echo "SMOKE_EXIT=$?"
