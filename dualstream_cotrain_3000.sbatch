#!/bin/bash
#SBATCH --job-name=indomain_c4_dualstream_cotrain
#SBATCH --output=logs/indomain_c4/launcher_logs/dualstream_cotrain_%j.out
#SBATCH --error=logs/indomain_c4/launcher_logs/dualstream_cotrain_%j.out
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

# Requeue-safe resume hook: pick up the newest last.ckpt for this run if one exists.
SRC=$(ls -t logs/indomain_c4/dualstream_cotrain_2026*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
RESUME=""
if [ -n "$SRC" ]; then
  RESUME="ckpt_path=$SRC"
  echo "[resume] Resuming from $SRC"
else
  echo "[resume] No existing checkpoint, starting fresh."
fi

srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian $RESUME \
  +experiment=indomain_c4/dualstream_cotrain \
  trainer.limit_train_batches=100 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=500 \
  callbacks.model_checkpoint.save_last=true \
  +callbacks.model_checkpoint.every_n_train_steps=10000 \
  callbacks.model_checkpoint.every_n_epochs=0
