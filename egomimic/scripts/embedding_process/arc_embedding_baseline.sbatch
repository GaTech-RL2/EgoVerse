#!/bin/bash
#SBATCH --job-name=arc_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p cpu-small
#SBATCH --output=logs/embedding/arc_base_%j.out
#SBATCH --error=logs/embedding/arc_base_%j.err

set -euo pipefail

REPO=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc
cd "$REPO"
mkdir -p logs/embedding out

source ~/.bashrc || true
source "$REPO/.venv/bin/activate"

# Classic time-based action-chunk baseline (no arc tokenizer). Same tabbed
# HTML viewer + click-image panel as arc_embedding_sweep.
DATA=${DATA:-human_mecka_eva_rl2_fold_clothes}
DATA_DIR=${DATA_DIR:-/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets}
OUT=${OUT:-$REPO/out/arc_baseline_${DATA}_${SLURM_JOB_ID}.html}
MAX=${MAX:-1500}
SPLIT=${SPLIT:-train}

echo "[$(date -Is)] repo=$REPO data=$DATA split=$SPLIT max=$MAX data_dir=$DATA_DIR"
echo "[$(date -Is)] out=$OUT"

python -m egomimic.scripts.arc_embedding_baseline \
    data="$DATA" \
    paths.dataset_dir="$DATA_DIR" \
    output_html="$OUT" \
    split="$SPLIT" \
    max_samples_per_dataset="$MAX"

echo "[$(date -Is)] done: $OUT"
ls -lh "$OUT"
