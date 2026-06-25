#!/bin/bash
#SBATCH --job-name=indomain_c4_dualstream_cotrain_m16
#SBATCH --output=logs/indomain_c4/launcher_logs/dualstream_cotrain_m16_%j.out
#SBATCH --error=logs/indomain_c4/launcher_logs/dualstream_cotrain_m16_%j.out
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

# Requeue-safe resume hook (|| true so an empty glob doesn't trip set -e under pipefail).
SRC=$(ls -t logs/indomain_c4/dualstream_cotrain_m16_2026*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
RESUME=""
if [ -n "$SRC" ]; then
  RESUME="ckpt_path=$SRC"
  echo "[resume] Resuming from $SRC"
else
  echo "[resume] No existing checkpoint, starting fresh."
fi

# 16-mode (10 agnostic / 6 specific) asym dual-stream, trunk/encoders identical to
# the 5-mode runs. Adds the DualStreamProbeCallback -> live Train/grad_norm_{A,S,trunk}
# + Train/wA_<emb>/wS_<emb>.
srun python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian $RESUME \
  +experiment=indomain_c4/dualstream_cotrain_m16 \
  +callbacks.ds_probe._target_=egomimic.pl_utils.callbacks.dual_stream_probe.DualStreamProbeCallback \
  trainer.limit_train_batches=100 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=500 \
  callbacks.model_checkpoint.save_last=true \
  +callbacks.model_checkpoint.every_n_train_steps=10000 \
  callbacks.model_checkpoint.every_n_epochs=0
