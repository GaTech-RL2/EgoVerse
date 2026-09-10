#!/bin/bash
#SBATCH --job-name=norm_freeform_hpt_kp
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/coc/flash7/rco3/EgoVerse2/logs/norm_fold_freeform_opsplit_hpt_kp_%j.log

# SKYNET, CPU-only: 0.1-frac quantile norm stats for the mecka FREEFORM
# fold-clothes set (462 eps / 1.13M frames), held-out-OPERATOR split, train
# side = the 147 non-held-out operators (412 eps), from the HPT KEYPOINT data
# config (data=mecka_fold_freeform_opsplit_hpt_keypoints, keypoints_wristframe_ypr).
# Key set = {actions_keypoints (138), observations.state.keypoints (138)}:
# per hand [wrist xyz+ypr (6) | 21 MANO keypoints (63)] x {left, right}. The
# 6D recipes' stats carry a different key set and fail the provenance check,
# so this recipe needs its own directory.
#
# The parent data config hard-codes the PACE mirror as folder_path, so the
# trailing hydra override points the resolver at the Skynet mirror instead
# (valid/train_viz interpolate the train resolver, one override covers all).
# First run also pulls every freeform train episode missing from DATA_DIR
# from S3 — freeform episodes are long (~2.4k frames each), so expect a much
# bigger first sync than the flagship set; the stats pass itself is minutes.
#
# Output: $NORM_ROOT/fold_freeform_opsplit_hpt_kp/norm_stats/norm_stats.json
# Training (train_hpt_kp_fold_freeform_opsplit_skynet.sbatch) uses:
#   NORM_PATH=$NORM_ROOT/fold_freeform_opsplit_hpt_kp/norm_stats
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
  --config-name train_zarr_mecka_kp_wrist_hpt \
  --data mecka_fold_freeform_opsplit_hpt_keypoints \
  --model hpt_bc_keypoints_wrist_300M \
  --sample-frac 0.1 \
  --num-workers 22 \
  --out "$NORM_ROOT/fold_freeform_opsplit_hpt_kp" \
  "paths.dataset_dir=$DATA_DIR" \
  "data.train_datasets.human_bimanual.resolver.folder_path=$DATA_DIR"
echo "[norm] exit=$?"
