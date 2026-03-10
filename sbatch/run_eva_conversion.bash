#!/bin/bash
#SBATCH --job-name=eva_conversion
#SBATCH --output=logs/eva_conversion_%j.out
#SBATCH --error=logs/eva_conversion_%j.err
#SBATCH --partition=rl2-lab
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G

# Load conda - try miniforge3 first, fallback to miniconda3
if [ -f /coc/flash7/mlin365/miniforge3/etc/profile.d/conda.sh ]; then
    source /coc/flash7/mlin365/miniforge3/etc/profile.d/conda.sh
elif [ -f /coc/flash7/mlin365/miniconda3/etc/profile.d/conda.sh ]; then
    source /coc/flash7/mlin365/miniconda3/etc/profile.d/conda.sh
fi

# Activate environment
conda activate egowm

# Navigate to script directory
cd /coc/flash7/mlin365/EgoVerse/egomimic/scripts/eva_process

# Run the conversion
echo "Starting eva conversion at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

python run_eva_conversion.py --no-sql --skip-if-done

echo ""
echo "Conversion completed at $(date)"
echo "Exit code: $?"
