#!/bin/bash
#SBATCH --job-name=trunk_latents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH --output=logs/trunk_latents/%j.out
#SBATCH --error=logs/trunk_latents/%j.err
#
# Partition + GRES intentionally NOT set as SBATCH directives so callers can
# route to whichever GPU pool has capacity via `--partition` + `--gres` on
# the submit line. Recommended patterns (h100 is default when unspecified):
#
#   sbatch --partition=gpu-h100 --gres=gpu:h100:1 scripts/visualize_trunk_latents.sbatch
#   sbatch --partition=gpu-a100 --gres=gpu:a100:1 scripts/visualize_trunk_latents.sbatch
#   sbatch --partition=gpu-h200 --gres=gpu:h200:1 scripts/visualize_trunk_latents.sbatch
#   sbatch --partition=gpu-l40s --gres=gpu:l40s:1 scripts/visualize_trunk_latents.sbatch
#
# `--export=ALL,PARTITION=...` env-var fallbacks below preserve the old
# behavior (default h100) when neither `--partition` nor `--gres` is passed.
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1

set -eo pipefail

REPO=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc
cd "$REPO"
mkdir -p logs/trunk_latents

source ~/.bashrc || true
source "$REPO/.venv/bin/activate"

RUN_DIR=${RUN_DIR:-/storage/home/hcoda1/5/acheluva3/r-dxu345-0/Egoverse-arc/logs/arc_tests_cotrain_arctok/h100_1gpu_bs128_constlr3e4_val30_2026-07-27_22-28-30/0}
CKPT=${CKPT:-$RUN_DIR/checkpoints/last.ckpt}
POOL=${POOL:-mean}
NSAMPLES=${NSAMPLES:-500}
BATCH=${BATCH:-32}
TAG=${TAG:-}
[ -n "$TAG" ] && TAG="_$TAG"
OUT=${OUT:-$REPO/logs/trunk_latents/trunk_latents_tsne2d${TAG}_${SLURM_JOB_ID}.html}

echo "[$(date -Is)] run_dir=$RUN_DIR"
echo "[$(date -Is)] ckpt=$CKPT"
echo "[$(date -Is)] pool=$POOL nsamples=$NSAMPLES batch=$BATCH"
echo "[$(date -Is)] out=$OUT"
nvidia-smi | head -20

python scripts/visualize_trunk_latents.py \
    --run-dir "$RUN_DIR" \
    --ckpt   "$CKPT" \
    --num-samples "$NSAMPLES" \
    --batch-size  "$BATCH" \
    --pool "$POOL" \
    --out "$OUT"

echo "[$(date -Is)] done: $OUT"
ls -lh "$OUT" || true
