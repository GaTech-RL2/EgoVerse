#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# Typed gres: an untyped --gres=gpu:1 over -p gpu-h200,gpu-h100 landed on a
# V100 node (atl1-1-03-006-35-0) on 2026-09-02 - no bf16, compiled sampler
# fell back to eager at ~2.6 min/batch.
#SBATCH -p gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0
#SBATCH --job-name=eval_palm_kp49
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/eval_palm_kp49_%j.log

# Eval-only pass of the keypoint run's epoch-49 checkpoint over the held-out-
# operator validation set (160 batches x 64, the same slice the training val
# sees across its 2 ranks): standard keypoint metrics + palm-origin cam-frame
# MSE (eval_pi_keypoints_palm), reverse-KL sampling off to halve the cost.

set -euo pipefail
WT=/storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/fold-freeform-kp
REPO=/storage/project/r-dxu345-0/agao81/EgoVerse
cd "$WT"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if grep -q api.wandb.ai ~/.netrc 2>/dev/null; then export WANDB_MODE=online; else export WANDB_MODE=offline; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CKPT="${CKPT-/storage/project/r-dxu345-0/agao81/EgoVerse/logs/keypoints_fold_clothes_freeform_fit_test_flat5e-5_opsplit/bs64x2_flat5e-5_kp_opsplit_2026-09-02_02-54-36/checkpoints/epoch_epoch=49.ckpt}"
LABEL="${LABEL-eval_palm_kp49}"

echo "[eval] cwd=$(pwd) commit=$(git rev-parse HEAD) ckpt=$CKPT"
"$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian_pi \
  data=mecka_fold_freeform_opsplit_pi_keypoints \
  model=pi0.5_bc_mecka_keypoints \
  evaluator=eval_pi_keypoints_palm \
  +evaluator.limit_val_batches=160 \
  launch_params.gpus_per_node=1 \
  ++mode=eval \
  "ckpt_path='$CKPT'" \
  norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_opsplit_kp/norm_stats \
  model.robomimic_model.config.pytorch_weight_path=$REPO/egomimic/algo/pi_checkpoints/pi05_base_pytorch \
  model.robomimic_model.config.reverse_kl_samples=1 \
  ~data.train_viz_datasets ~data.train_viz_dataloader_params \
  "name=$LABEL" description=ep49_valset_160batches \
  "+logger.wandb.name=$LABEL" \
  'hydra.run.dir=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/${name}/${description}_${now:%Y-%m-%d_%H-%M-%S}'
echo "[eval] exit=$?"
