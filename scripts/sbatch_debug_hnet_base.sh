#!/bin/bash
#SBATCH --job-name=dbg-hnet
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH --output=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/logs/sbatch/dbg_hnet_%j.out
#SBATCH --error=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3/logs/sbatch/dbg_hnet_%j.err

set -uxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-clone-3
mkdir -p logs/sbatch
source .venv/bin/activate
export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"
export PYTHONPATH=.

srun .venv/bin/python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian \
  name=refactor_smoke description=hnet_base_debug \
  mode=train \
  data=tsimulation_full \
  model=hnet/pushshapes \
  norm_stats.sample_frac=0.1 \
  trainer=debug \
  trainer.max_epochs=2 trainer.min_epochs=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=2 \
  trainer.profiler=null \
  ~callbacks \
  ~evaluator \
  logger=csv
