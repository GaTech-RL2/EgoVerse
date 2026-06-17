#!/bin/bash
#SBATCH --job-name=pact-pixel-heavy
#SBATCH -A hoffman-lab
#SBATCH -p overcap
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=short
#SBATCH --no-requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/pact_heavy_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/pact_heavy_%j.err

set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
mkdir -p logs/sbatch

source .venv/bin/activate
export PYTHONPATH=.
export PACK_COLLATE_MAX_TOTAL_FRAMES=512

srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=pact_pixel_dfot description=pixel_dfot_9frame_heavy \
  mode=train \
  data=tsimulation/full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  model=dfot/pushshapes_pixel \
  model.scheduler.max_steps=200000 \
  model.scheduler.warmup_steps=1000 \
  evaluator=eval_dfot_pixel \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=50 \
  trainer=debug \
  trainer.max_epochs=400 \
  trainer.min_epochs=400 \
  trainer.limit_train_batches=512 data.train_dataloader_params.pushshapes_sim.batch_size=2 data.valid_dataloader_params.pushshapes_sim.batch_size=2 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=100 \
  trainer.profiler=simple \
  trainer.precision=16-mixed \
  +trainer.gradient_clip_val=1.0 \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/external_ckpts/pushshapes_circle_750_norm_stats.json \
  logger=csv_wandb
