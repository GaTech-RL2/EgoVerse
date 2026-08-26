#!/bin/bash
#SBATCH --job-name=srcdiag
#SBATCH -A rl2-lab
#SBATCH -p rl2-lab
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/srcdiag_%j.out

cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
export PYTHONPATH=/coc/flash7/paphiwetsa3/projects/EgoVerse2
echo "node=$(hostname)"
../EgoVerse7/.venv/bin/python scripts/src_cov_diag.py --n-src 200 --n-scripted 100 --max-steps 400
echo "SBATCH-DONE"
