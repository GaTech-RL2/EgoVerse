#!/bin/bash
#SBATCH --job-name=oa-variant
#SBATCH -A hoffman-lab
#SBATCH -p overcap
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=short
#SBATCH --no-requeue
#SBATCH --exclude=sonny
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/oa_variant_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/oa_variant_%j.err

set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
mkdir -p logs/sbatch
source .venv/bin/activate
export PYTHONPATH=.
export PACK_COLLATE_MAX_TOTAL_FRAMES=512

MODEL=${MODEL:-dfot_pushshapes_obs_action_image}

# Real training, validation DISABLED (check_val > max_epochs) so the not-yet-
# ported obs-action eval can't crash; checkpoints save every 30 epochs via
# every_n_epochs (independent of val). Verify offline via anchored re-eval.
srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=oa_${MODEL} description=oa_${MODEL}_real \
  mode=train \
  data=tsimulation/full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  model=${MODEL} \
  model.scheduler.max_steps=19200 \
  evaluator=dfot/obs_action_image \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=30 \
  trainer=debug \
  trainer.max_epochs=120 \
  trainer.min_epochs=120 \
  trainer.limit_train_batches=160 data.train_dataloader_params.pushshapes_sim.batch_size=2 data.valid_dataloader_params.pushshapes_sim.batch_size=2 \
  trainer.limit_val_batches=1 \
  trainer.check_val_every_n_epoch=200 \
  trainer.precision=16-mixed \
  +trainer.gradient_clip_val=1.0 \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/external_ckpts/pushshapes_circle_750_norm_stats.json \
  logger=csv_wandb
