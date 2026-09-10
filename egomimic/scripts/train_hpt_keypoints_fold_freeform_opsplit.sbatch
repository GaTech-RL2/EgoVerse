#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=hpt_kp_fold_freeform_opsplit
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/hpt_keypoints_fold_clothes_freeform_opsplit_%j.log

# HPT twin of train_keypoints_fold_freeform_opsplit.sbatch: ~300M HPT flow
# BC (+ frozen Qwen3-Embedding text stem) on the 138-D wrist-first MANO
# KEYPOINT action in wrist frame (keypoints_wristframe_pi: wrist pose (6) +
# 21 keypoints (63) per hand), mecka FREEFORM fold-clothes, held-out-OPERATOR
# split (412 train eps / 50 val eps from 31 held-out operators). Constant lr
# 5e-5 (model config), 64/GPU x 1 GPU. Cadence (epoch = limit_train_batches
# = 100 steps): val every 20 epochs, viz every 100 (both heads, <=10 batches),
# ckpt every 50 epochs. Eval = eval_hpt_keypoints_wrist (wrist -> head-frame
# keypoint revert, 126-D cam-frame keypoint MSE + overlay video) + train_viz
# head. Metric names match the pi twin's
# (Valid/human_bimanual_actions_keypoints_{paired,final,cam_paired,cam_final}_mse_avg).
#
# GPU: the repo rule puts non-pi runs on L40S (`-p gpu-l40s --gres=gpu:1`;
# `--gres=gpu:l40s:1` is rejected). For an H200 instead:
#   sbatch -p gpu-h200 --gres=gpu:h200:1 egomimic/scripts/train_hpt_keypoints_fold_freeform_opsplit.sbatch
#
# NORM STATS: the pi keypoint twin's precomputed stats are valid here — the
# norm-mode keymap strips camera keys, so both configs normalize the same
# {actions_keypoints, observations.state.keypoints, observations.state.ee_pose,
# ...} key set (MultiDataset's provenance check enforces it). Default below
# = the pi twin's file; norm_fold_freeform_opsplit_hpt_kp.sbatch recomputes.
#
# The Qwen stem loads Qwen/Qwen3-Embedding-0.6B from the HF cache; compute
# nodes have no internet, so pre-fetch it once on the login node:
#   emimic/bin/python -c "from transformers import AutoModel, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B'); AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')"
#
# Env knobs:
#   WT         - worktree to run from (default: the kp-wristframe-port worktree)
#   NORM_PATH  - precomputed norm_stats dir (default above)
#   CKPT_PATH  - resume from a Lightning checkpoint
#   LR         - optimizer lr (default 5e-5)
#   BS         - per-GPU batch size (default 64)

set -euo pipefail
WT="${WT-/storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/kp-wristframe-port}"
REPO=/storage/project/r-dxu345-0/agao81/EgoVerse
cd "$WT"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
  export WANDB_MODE=online   # netrc-authenticated
else
  export WANDB_MODE=offline
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Private compile caches per job (same-user jobs on one node otherwise share
# /tmp/torchinductor_<user>).
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID}"

NORM_PATH="${NORM_PATH-/storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_opsplit_kp/norm_stats}"
CKPT_PATH="${CKPT_PATH-}"
LR="${LR-5e-5}"
BS="${BS-64}"
LABEL=hpt300m_keypoints_fold_clothes_freeform_opsplit

if [[ ! -f "$NORM_PATH/norm_stats.json" ]]; then
  echo "ERROR: precomputed norm stats not found: $NORM_PATH/norm_stats.json" >&2
  echo "Run egomimic/scripts/norm_fold_freeform_opsplit_hpt_kp.sbatch first." >&2
  exit 1
fi

ARGS=(
  --config-name=train_zarr_mecka_kp_wrist_hpt
  launch_params.gpus_per_node=1
  "model.optimizer.lr=$LR"
  "data.train_dataloader_params.human_bimanual.batch_size=$BS"
  "data.valid_dataloader_params.human_bimanual.batch_size=$BS"
  "norm_stats.precomputed_norm_path=$NORM_PATH"
  trainer.check_val_every_n_epoch=20
  callbacks.model_checkpoint.every_n_epochs=50
  "name=$LABEL"
  "description=bs${BS}x1_const${LR}_kp_opsplit"
  "+logger.wandb.name=$LABEL"
  # Keep outputs (checkpoints, norm-stats cache) out of the worktree.
  'hydra.run.dir=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/${name}/${description}_${now:%Y-%m-%d_%H-%M-%S}'
)
# Lightning names checkpoints "epoch_epoch=..." — quote so hydra's grammar
# doesn't choke on '='.
[[ -n "$CKPT_PATH" ]] && ARGS+=("ckpt_path='$CKPT_PATH'")

echo "[train] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[train] srun python -u egomimic/trainHydra.py ${ARGS[*]}"
srun "$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
