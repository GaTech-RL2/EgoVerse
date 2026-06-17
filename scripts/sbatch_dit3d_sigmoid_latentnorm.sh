#!/bin/bash
#SBATCH --job-name=dit3d-sig
#SBATCH -A overcap
#SBATCH --qos=scavenger_qos
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/dit3d_sig_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/logs/sbatch/dit3d_sig_%j.err

set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-pact
mkdir -p logs/sbatch

source .venv/bin/activate
export PYTHONPATH=.
export PACK_COLLATE_MAX_TOTAL_FRAMES=512

# Sigmoid weighting + latent norm + no-padding + additive fusion
srun --kill-on-bad-exit=1 .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=dit3d_sigmoid_lnorm description=dit3d_nopad_sigmoid_addfusion_latentnorm \
  mode=train \
  data=tsimulation/full \
  data.train_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  data.valid_datasets.pushshapes_sim.resolver.folder_path=/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle \
  data.train_dataloader_params.pushshapes_sim.batch_size=4 \
  data.valid_dataloader_params.pushshapes_sim.batch_size=4 \
  model=dfot/pushshapes_image_spatial_cont_sigmoid \
  evaluator=dfot/image_spatial \
  evaluator.evals.0.n_chunk_steps=100 \
  callbacks=checkpoints \
  callbacks.model_checkpoint.every_n_epochs=50 \
  trainer=debug \
  trainer.max_epochs=400 \
  trainer.min_epochs=400 \
  trainer.limit_train_batches=80 \
  trainer.limit_val_batches=4 \
  trainer.check_val_every_n_epoch=25 \
  trainer.profiler=simple \
  norm_stats.precomputed_norm_path=/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/external_ckpts/pushshapes_circle_750_norm_stats.json \
  logger=csv_wandb
