#!/bin/bash
#SBATCH --job-name=vae-v6
#SBATCH -A overcap
#SBATCH --qos=scavenger_qos
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/vae_v6_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/vae_v6_%j.err

set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
mkdir -p logs/sbatch

source .venv/bin/activate
export PYTHONPATH=.
export PACK_COLLATE_MAX_TOTAL_FRAMES=128

srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=vae_pretrain description=vae_v6_kl0.001_400ep \
  mode=train \
  data=tsimulation/full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  data.train_dataloader_params.pushshapes_sim.batch_size=2 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=2 \
  model=vae/pushshapes_v6 \
  evaluator=eval_vae \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=50 \
  trainer=debug \
  trainer.max_epochs=400 \
  trainer.min_epochs=400 \
  trainer.limit_train_batches=80 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=50 \
  trainer.profiler=simple \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/external_ckpts/pushshapes_circle_750_norm_stats.json \
  logger=csv_wandb
