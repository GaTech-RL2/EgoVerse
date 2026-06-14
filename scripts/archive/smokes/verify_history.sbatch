#!/bin/bash
#SBATCH --job-name=verifyhist
#SBATCH --time=0:20:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/verifyhist_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/verifyhist_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. REPO=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
srun python verify_history.py
echo "VERIFY_EXIT=$?"
