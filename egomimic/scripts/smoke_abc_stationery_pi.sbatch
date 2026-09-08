#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=smoke_abc_stat
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/smoke_abc_stationery_%j.log
#SBATCH --exclude=atl1-1-02-012-16-0,atl1-1-02-014-9-0

# End-to-end smoke for the stationery pi0.5 path: 16-episode subset,
# trainer=debug (4 epochs, val@2, 2 train / 3 val batches), viz at epoch 3.
# Exercises the relabel-proof filter, the stock annotation pipeline (ABC's
# dense sub-task spans as prompts), the pi0.5-base load, the eva wristframe-6D
# revert and the video render — everything the 72 h run would otherwise
# discover late.

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/abc-stationery-pi
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/tr_$SLURM_JOB_ID

echo "=== commit=$(git rev-parse --short HEAD) node=$SLURMD_NODENAME $(date) ==="

"$PY" -u egomimic/trainHydra.py \
  --config-name=train_zarr_cartesian_pi \
  data=abc_stationery_pi_6d \
  model=pi0.5_bc_abc_eva_6d \
  evaluator=eval_pi_wristframe_6d \
  +evaluator@train_viz_evaluator=train_viz_pi_wristframe_6d \
  trainer=debug \
  launch_params.gpus_per_node=1 \
  evaluator.viz_every_n_epochs=4 \
  "++data.train_datasets.eva_bimanual.resolver.debug=16" \
  "data.train_datasets.eva_bimanual.valid_ratio=0.25" \
  "norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/abc_stationery_rl2yam/norm_stats" \
  model.robomimic_model.config.pytorch_weight_path=/storage/project/r-dxu345-0/agao81/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch \
  name=smoke_abc_stationery \
  description=smoke_abc_stationery_pi
echo "=== SMOKE EXIT $? ($(date)) ==="
