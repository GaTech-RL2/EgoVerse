#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=norm_fold_flagship_opsplit_hpt_6d
#SBATCH --output=logs/norm_fold_flagship_opsplit_hpt_6d_%j.log

# CPU-only: 0.1-frac quantile norm stats for the mecka FLAGSHIP fold-clothes
# set, seen/unseen-OPERATOR split (train side: the 18 non-held-out operators,
# ~1346 eps / 271k frames after the seeded 95/5 episode split), from the HPT
# 6D data config (data=mecka_fold_flagship_opsplit_hpt_6d, model=
# hpt_bc_mecka_6d_300M). Key set = {actions_cartesian (18),
# observations.state.ee_pose (20)}; the freeform stats share it but not the
# distribution, so do not reuse them.
#
# Submit from the repo root (the #SBATCH --output path is relative to it;
# `mkdir -p logs` first). The first run syncs any flagship zarrs missing from
# the shared mirror from S3 (316k frames total).
#
# Output: $NORM_ROOT/fold_flagship_opsplit_hpt_6d/norm_stats/norm_stats.json
# Training uses:
#   NORM_PATH=$NORM_ROOT/fold_flagship_opsplit_hpt_6d/norm_stats
#
# Env knobs:
#   REPO       - checkout to run from (default: the submit directory)
#   NORM_ROOT  - where norm-stats dirs live (default: $REPO/norm_stats)
#   DATASET_DIR - zarr mirror (default: the shared PACE mirror /storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets)

set -euo pipefail
REPO="${REPO-$SLURM_SUBMIT_DIR}"
NORM_ROOT="${NORM_ROOT-$REPO/norm_stats}"
DATASET_DIR="${DATASET_DIR-/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/emimic/bin/python"
export PATH="$REPO/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "[norm] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
"$PY" egomimic/scripts/precompute_norm_stats.py \
  --config-name train_zarr_mecka_flagship_6d_hpt \
  --data mecka_fold_flagship_opsplit_hpt_6d \
  --model hpt_bc_mecka_6d_300M \
  --sample-frac 0.1 \
  --num-workers 22 \
  --out "$NORM_ROOT/fold_flagship_opsplit_hpt_6d" \
  "paths.dataset_dir=$DATASET_DIR"
