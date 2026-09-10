#!/bin/bash
#SBATCH --job-name=norm_flagship_hpt_6d
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/coc/flash7/rco3/EgoVerse2/logs/norm_fold_flagship_opsplit_hpt_6d_%j.log

# SKYNET, CPU-only: 0.1-frac quantile norm stats for the mecka FLAGSHIP
# fold-clothes set, seen/unseen-OPERATOR split, train side = the 18
# non-held-out operators after the seeded 95/5 episode split (~1346 eps /
# 271k frames), from the HPT 6D data config. Key set = {actions_cartesian (18),
# observations.state.ee_pose (20)}. Skynet twin of
# norm_fold_flagship_opsplit_hpt_6d.sbatch (PACE).
#
# First run also pulls every flagship train episode missing from DATA_DIR
# from S3 (s5cmd, ~8 MB/episode, ~11 GB total) — that is most of the wall
# time; the stats pass itself is minutes.
#
# Output: $NORM_ROOT/fold_flagship_opsplit_hpt_6d/norm_stats/norm_stats.json
# Training (train_hpt_6d_fold_flagship_opsplit_skynet.sbatch) uses:
#   NORM_PATH=$NORM_ROOT/fold_flagship_opsplit_hpt_6d/norm_stats
#
# Env knobs:
#   DATA_DIR   - zarr mirror (default: /coc/flash7/scratch/egoverseS3ZarrDataset = paths.dataset_dir)
#   NORM_ROOT  - where norm-stats dirs live (default: $REPO/norm_stats)

set -euo pipefail
case "$(hostname)" in sky1*|sky2*) echo "ABORT: on login node"; exit 1;; esac
hostname
REPO=/coc/flash7/rco3/EgoVerse2
DATA_DIR="${DATA_DIR-/coc/flash7/scratch/egoverseS3ZarrDataset}"
NORM_ROOT="${NORM_ROOT-$REPO/norm_stats}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/emimic/bin/python"
export PATH="$REPO/emimic/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$DATA_DIR" "$NORM_ROOT"

echo "[norm] cwd=$(pwd) commit=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) data=$DATA_DIR"
"$PY" egomimic/scripts/precompute_norm_stats.py \
  --config-name train_zarr_mecka_flagship_6d_hpt \
  --data mecka_fold_flagship_opsplit_hpt_6d \
  --model hpt_bc_mecka_6d_300M \
  --sample-frac 0.1 \
  --num-workers 22 \
  --out "$NORM_ROOT/fold_flagship_opsplit_hpt_6d" \
  "paths.dataset_dir=$DATA_DIR"
echo "[norm] exit=$?"
