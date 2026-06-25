#!/bin/bash
#SBATCH --job-name=indomain_c4_dualstream_cotrain_200m_sym
#SBATCH --output=logs/indomain_c4/launcher_logs/dualstream_cotrain_200m_sym_%j.out
#SBATCH --error=logs/indomain_c4/launcher_logs/dualstream_cotrain_200m_sym_%j.out
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --requeue

set -eo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-dualstream
mkdir -p logs/indomain_c4/launcher_logs
source .venv/bin/activate
export PYTHONPATH=.
export PATH=/opt/slurm/Ubuntu-20.04/current/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PACK_COLLATE_MAX_TOTAL_FRAMES=256

# Requeue-safe resume hook (|| true so empty glob doesn't trip set -e under pipefail).
SRC=$(ls -t logs/indomain_c4/dualstream_cotrain_200m_sym_2026*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
RESUME=""
if [ -n "$SRC" ]; then
  RESUME="ckpt_path=$SRC"
  echo "[resume] Resuming from $SRC"
else
  echo "[resume] No existing checkpoint, starting fresh."
fi

# 200M-matched dual-stream SYM (mask_mode=sym) = mask A/B twin of the 200M asym run.
srun --ntasks=2 --ntasks-per-node=2 python -m egomimic.trainHydra \
  --config-name=train_zarr_cartesian $RESUME \
  +experiment=indomain_c4/dualstream_cotrain_200m \
  description=dualstream_cotrain_200m_sym \
  model.robomimic_model.outer_stage.hnet.stages.0.mask_mode=sym \
  launch_params.gpus_per_node=2 \
  +callbacks.ds_probe._target_=egomimic.pl_utils.callbacks.dual_stream_probe.DualStreamProbeCallback \
  trainer.limit_train_batches=100 \
  trainer.limit_val_batches=2 \
  trainer.check_val_every_n_epoch=500 \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.every_n_epochs=100 \
  callbacks.model_checkpoint.save_top_k=-1
