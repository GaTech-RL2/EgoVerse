#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=smoke_mecka_6d
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/smoke_mecka_6d_%j.log

# 1xH200 smoke of the mecka cartesian-6d pi0.5 pipeline with trainer=debug
# (4 epochs, val every 2, 2 train / 3 val batches).
#
# viz gate check: evaluator.viz_every_n_epochs=4 with val at epochs 1 and 3
# means epoch 1's validation must NOT render video and epoch 3's must
# ((3+1) % 4 == 0) -> exactly one videos/epoch_3 dir proves the gate.
#
# Env knobs:
#   DEBUG_EPS  - limit resolver to N episodes (default 12; empty = all)
#   NORM_PATH  - precomputed norm_stats dir; empty = compute fresh at startup

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DEBUG_EPS="${DEBUG_EPS-12}"
NORM_PATH="${NORM_PATH-}"

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=mecka_all_pi_6d
  model=pi0.5_bc_mecka_6d
  evaluator=eval_pi_wristframe_6d
  trainer=debug
  launch_params.gpus_per_node=1
  evaluator.viz_every_n_epochs=4
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
  name=smoke_mecka_6d
  description=smoke_mecka_all_pi_6d
)
if [[ -n "$DEBUG_EPS" ]]; then
  ARGS+=(
    "++data.train_datasets.human_bimanual.resolver.debug=$DEBUG_EPS"
    # 12-episode subset: default valid_ratio 0.05 floors to 0 valid episodes.
    "data.train_datasets.human_bimanual.valid_ratio=0.25"
  )
fi
if [[ -n "$NORM_PATH" ]]; then
  ARGS+=("norm_stats.precomputed_norm_path=$NORM_PATH")
else
  # inline stats on a tiny fraction: the smoke only needs plausible
  # normalizers, not accurate ones — avoids the startup stats cost.
  ARGS+=("norm_stats.sample_frac=0.02")
fi

echo "[smoke] python -u egomimic/trainHydra.py ${ARGS[*]}"
"$PY" -u egomimic/trainHydra.py "${ARGS[@]}"
