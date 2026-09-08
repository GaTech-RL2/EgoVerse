#!/bin/bash
#SBATCH --job-name=arc_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p cpu-small
#SBATCH --output=logs/embedding/arc_sweep_%j.out
#SBATCH --error=logs/embedding/arc_sweep_%j.err

set -euo pipefail

REPO=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc
cd "$REPO"
mkdir -p logs/embedding out

source ~/.bashrc || true
source "$REPO/.venv/bin/activate"

# Defaults sweep the aria + eva RL2 fold_clothes cotraining set.
DATA=${DATA:-aria_eva_fold_clothes_rl2}
DATA_DIR=${DATA_DIR:-/storage/project/r-dxu345-0/shared/aria_fold}
# Naming: arc_sweep_<data-config>_<jobid>.html — data-config in the name so
# multiple concurrent sweeps on different data don't collide, and so you can
# eyeball what a file is without opening it.
OUT=${OUT:-$REPO/out/arc_sweep_${DATA}_${SLURM_JOB_ID}.html}
MAX=${MAX:-1500}
SPLIT=${SPLIT:-train}
FEATURE=${FEATURE:-positions_delta_only}

echo "[$(date -Is)] repo=$REPO"
echo "[$(date -Is)] data=$DATA split=$SPLIT max=$MAX data_dir=$DATA_DIR feature=$FEATURE"
echo "[$(date -Is)] out=$OUT"

EXTRA=${EXTRA:-}
python -m egomimic.scripts.arc_embedding_sweep \
    data="$DATA" \
    paths.dataset_dir="$DATA_DIR" \
    output_html="$OUT" \
    split="$SPLIT" \
    max_samples_per_dataset="$MAX" \
    feature="$FEATURE" \
    $EXTRA

echo "[$(date -Is)] done: $OUT"
ls -lh "$OUT"
