#!/bin/bash
#SBATCH --job-name=abc_convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=720
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
set -euo pipefail

# Download + convert ABC-130k episodes until a FRAME budget is met.
#
#   sbatch scripts/abc_build_subset.sbatch [task] [target_frames] [out_dir]
#
# Defaults build a t-shirt-folding set matching the mecka fold_clothes frame
# count (316068 frames over 1607 episodes; see the DB filter in
# hydra_configs/data/mecka_foldclothes_*).
#
# CPU only -- this is video decode + JPEG encode, no GPU. Frame budget rather
# than episode count because ABC episode lengths vary ~3x and roughly a third of
# t-shirt episodes ship no EE pose and cannot be converted.
#
# Re-running is incremental: existing zarrs count toward the budget and are
# skipped, so a requeue resumes rather than restarting.

TASK="${1:-fold_and_stack_the_t_shirts}"
TARGET_FRAMES="${2:-316068}"
OUT_DIR="${3:-/coc/flash7/scratch/acheluva3/abc_data}"

cd /coc/flash7/acheluva3/egoverse-arc

# The Hub token must sit at $HF_HOME/token; huggingface_hub does not look in
# ~/.cache/huggingface when HF_HOME is set elsewhere.
export HF_HOME="${HF_HOME:-/coc/flash7/scratch/acheluva3/.cache}"

echo "task=$TASK  target_frames=$TARGET_FRAMES  out=$OUT_DIR"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  node=$(hostname)"

srun .venv/bin/python -m egomimic.scripts.abc_process.build_abc_subset \
    --task "$TASK" \
    --target-frames "$TARGET_FRAMES" \
    --out "$OUT_DIR"
