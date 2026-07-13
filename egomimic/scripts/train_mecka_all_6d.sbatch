#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
# 1TB: 8 ranks x ~75GB/rank tree (8 workers fork the 41k-episode dataset;
# refcount CoW drift — see the 4g OOM at 63GB/rank, job 10894868).
#SBATCH --mem=1000G
#SBATCH --gres=gpu:h200:8
#SBATCH --time=72:00:00
# atl1-1-01-009-16-0: 5x backward slowdown at 8 ranks (12.6 s/step vs 3.05
# on healthy nodes; raw NCCL allreduce fine at 476GB/s). Both slow runs sat
# on this node; matrix probe 10923198 exonerated the config.
#SBATCH --exclude=atl1-1-01-009-16-0
#SBATCH --job-name=pi05_mecka_all_6d
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_mecka_all_6d_%j.log

# Full pi0.5 BC on ALL SQL mecka episodes (cartesian_6d, 6D proprio), 8xH200.
#
# Cadence (epoch = limit_train_batches = 100 steps):
#   val every 50 epochs (5k steps, 80 batches, no shuffle)
#   viz every 100 epochs (10k steps)  ckpt every 100 epochs, top-3 by step
#
# Requires the CPU norm job's output (norm_mecka_all_6d.sbatch):
#   /storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d/norm_stats
#
# Env knobs:
#   NORM_PATH  - precomputed norm_stats dir (default above)
#   CKPT_PATH  - resume from a Lightning checkpoint
#   LR         - optimizer lr (default 5e-5, matching the remote mecka runs)

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# wandb online (key from apikey.txt); falls back to offline if absent.
if grep -q WANDB_API_KEY apikey.txt 2>/dev/null; then
  export WANDB_API_KEY="$(grep -oP 'WANDB_API_KEY=\K.*' apikey.txt)"
  export WANDB_MODE=online
elif grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
  export WANDB_MODE=online   # netrc-authenticated
else
  export WANDB_MODE=offline
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"

# Handoff from a still-running lower-GPU job: cancel it AT START TIME and
# resume from the freshest checkpoint it produced (resolved now, not at
# submission). Prefer the newest epoch_*.ckpt — it is written before
# last.ckpt at each save event, so it is complete once last.ckpt exists.
HANDOFF_JOB="${HANDOFF_JOB-}"
RESUME_DIR="${RESUME_DIR-}"
if [[ -n "$HANDOFF_JOB" ]]; then
  scancel "$HANDOFF_JOB" 2>/dev/null || true
  echo "[handoff] cancelled $HANDOFF_JOB; letting checkpoint writes settle"
  sleep 120
fi
if [[ -z "$CKPT_PATH" && -n "$RESUME_DIR" ]]; then
  CKPT_PATH=$(ls -t "$RESUME_DIR"/epoch_*.ckpt 2>/dev/null | head -1 || true)
  [[ -z "$CKPT_PATH" && -f "$RESUME_DIR/last.ckpt" ]] && CKPT_PATH="$RESUME_DIR/last.ckpt"
  echo "[handoff] resuming from: ${CKPT_PATH:-<none found - fresh start>}"
fi

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  echo "Run egomimic/scripts/norm_mecka_all_6d.sbatch first." >&2
  exit 1
fi

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=mecka_all_pi_6d
  model=pi0.5_bc_mecka_6d
  evaluator=eval_pi_wristframe_6d
  # Second eval head: spot-check metrics + videos on TRAIN data each val
  # (train_viz/ prefix, videos_train_viz/). Loader defined in the data config.
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d
  launch_params.gpus_per_node=8
  "model.optimizer.lr=$LR"
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  # cadence (val 50 / viz 100 / ckpt 100 top-3) is the config default now
  # (trainer/ddp_pi.yaml + callbacks/checkpoints.yaml). Checkpoint every 50
  # epochs (~4h at 8-GPU pace) to bound crash loss — same policy as the 4g run.
  callbacks.model_checkpoint.every_n_epochs=50
  name=pi05_mecka_all_6d
  description=pi05_mecka_all_6d_bs64x8_constlr_wrist6d
)
# Lightning names checkpoints "epoch_epoch=..." — quote so hydra's grammar
# doesn't choke on '='.
[[ -n "$CKPT_PATH" ]] && ARGS+=("ckpt_path='$CKPT_PATH'")

echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
