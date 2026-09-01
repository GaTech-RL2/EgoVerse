#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=norm_fold_freeform_cam6d
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/norm_fold_freeform_cam6d_%j.log

# CPU-only: 0.1-frac quantile norm stats for the mecka FREEFORM fold-clothes
# set (462 eps / 1.13M frames) in the cam6d pipeline
# (data=mecka_fold_freeform_pi_cam6d, model=pi0.5_bc_mecka_6d). Same procedure as
# norm_mecka_all_6d.sbatch, run from the aidan/fold-freeform-frames worktree.
#
# Output: /storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_cam6d/norm_stats/norm_stats.json
# Training uses:
#   norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_cam6d/norm_stats

set -euo pipefail
WT=/storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/fold-freeform-frames
cd "$WT"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[norm] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
"$PY" egomimic/scripts/precompute_norm_stats.py \
  --data mecka_fold_freeform_pi_cam6d \
  --model pi0.5_bc_mecka_6d \
  --sample-frac 0.1 \
  --num-workers 22 \
  --out /storage/project/r-dxu345-0/agao81/norm_stats/fold_freeform_cam6d
