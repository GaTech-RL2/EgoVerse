#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=profile_dl
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/profile_%j.out
#SBATCH -e /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/profile_%j.err
set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=.

python scripts/profile_dataloader.py
