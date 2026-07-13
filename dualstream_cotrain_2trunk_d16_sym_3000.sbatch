#!/bin/bash
#SBATCH --job-name=indomain_c4_ds2trunk_d16_sym
#SBATCH --output=logs/indomain_c4/launcher_logs/ds2trunk_d16_sym_%j.out
#SBATCH --error=logs/indomain_c4/launcher_logs/ds2trunk_d16_sym_%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --requeue

set -eo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-dualstream
mkdir -p logs/indomain_c4/launcher_logs
source .venv/bin/activate
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PACK_COLLATE_MAX_TOTAL_FRAMES=512

# Reuse full-data minmax norm_stats (data-only, model-independent) -> identical asym/sym norm.
NORM=logs/indomain_c4/dualstream_cotrain_m16_sym_2026-06-24_15-12-49/norm_stats/norm_stats.json

# Requeue-safe resume: newest last.ckpt for THIS (sym) run dir only.
SRC=$(ls -t logs/indomain_c4/dualstream_cotrain_2trunk_d16_200m_sym_2026*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
RESUME=""
if [ -n "$SRC" ]; then RESUME="ckpt_path=$SRC"; echo "[resume] from $SRC"; else echo "[resume] fresh"; fi

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian $RESUME \
  +experiment=indomain_c4/dualstream_cotrain_2trunk_d16_200m model.robomimic_model.outer_stage.hnet.stages.0.mask_mode=sym description=dualstream_cotrain_2trunk_d16_200m_sym \
  norm_stats.precomputed_norm_path=$NORM \
  trainer.limit_train_batches=100 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=500 \
  callbacks.model_checkpoint.save_last=true \
  +callbacks.model_checkpoint.every_n_train_steps=10000 \
  callbacks.model_checkpoint.every_n_epochs=0
