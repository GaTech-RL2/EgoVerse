#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=257875M
#SBATCH --gres=gpu:h200:4
#SBATCH --time=72:00:00
#SBATCH --job-name=pi05_mecka_4g
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/pi05_mecka_4g_%j.log

# Full pi0.5 BC on ALL SQL mecka episodes (cartesian_6d, 6D proprio), 4xH200.
#
# 4 GPUs, not 8: the 8-rank shape hits a pathological NCCL slowdown on these
# HGX nodes (12.6 s/step vs 3.6 s/step measured at 4 ranks on the same
# hardware/strategy — see legacy run obj_gen_..._9368273). bs64/GPU -> global
# batch 256.
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
  launch_params.gpus_per_node=4
  "model.optimizer.lr=$LR"
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  # cadence (val 50 / viz 100 / ckpt 100 top-3) is the config default now
  # (trainer/ddp_pi.yaml + callbacks/checkpoints.yaml).
  name=pi05_mecka_all_6d
  description=pi05_mecka_all_6d_bs64x4_constlr_wrist6d
)
# Lightning names checkpoints "epoch_epoch=..." — quote so hydra's grammar
# doesn't choke on '='.
[[ -n "$CKPT_PATH" ]] && ARGS+=("ckpt_path='$CKPT_PATH'")

echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
