#!/bin/bash
#SBATCH --job-name=arc_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p cpu-small
#SBATCH --output=logs/embedding/arc_%j.out
#SBATCH --error=logs/embedding/arc_%j.err

set -euo pipefail

REPO=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc
cd "$REPO"
mkdir -p logs/embedding out

source ~/.bashrc || true
source "$REPO/.venv/bin/activate"

# Defaults: aria + eva RL2 fold_clothes cotraining set, SQL-backed
# S3EpisodeResolver — lazy-downloads any missing episodes into DATA_DIR.
# Override at submit time, e.g.:
#   sbatch --export=ALL,DATA=aria_fold_clothes_lab arc_embedding.sbatch
#   sbatch --export=ALL,EMB=tsne_3d,MAX=3000 arc_embedding.sbatch
DATA=${DATA:-aria_eva_fold_clothes_rl2}
EMB=${EMB:-tsne_2d}
DATA_DIR=${DATA_DIR:-/storage/project/r-dxu345-0/shared/aria_fold}
# Naming: arc_embed_<data-config>_<embedder>_<jobid>.html — data & embedder in
# the name so multiple concurrent runs on different data don't collide, and
# so you can eyeball what a file is without opening it.
OUT=${OUT:-$REPO/out/arc_embed_${DATA}_${EMB}_${SLURM_JOB_ID}.html}
MAX=${MAX:-1500}
SPLIT=${SPLIT:-train}

echo "[$(date -Is)] repo=$REPO"
echo "[$(date -Is)] data=$DATA embedder=$EMB split=$SPLIT max=$MAX"
echo "[$(date -Is)] data_dir=$DATA_DIR"
echo "[$(date -Is)] out=$OUT"

python -m egomimic.scripts.arc_embedding_viz \
    data="$DATA" \
    embedder="$EMB" \
    paths.dataset_dir="$DATA_DIR" \
    output_html="$OUT" \
    split="$SPLIT" \
    embedder.max_samples_per_dataset="$MAX"

echo "[$(date -Is)] done: $OUT"
ls -lh "$OUT"
