#!/bin/bash
#SBATCH --job-name=genyield
#SBATCH -A rl2-lab
#SBATCH -p rl2-lab
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/genyield_%j.out

cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
export PYTHONPATH=/coc/flash7/paphiwetsa3/projects/EgoVerse2
echo "node=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"
../EgoVerse7/.venv/bin/python scripts/gen_yield.py \
  --ckpt /coc/flash7/paphiwetsa3/fpo_snapshots/gmmBC_c3000_ep1799_FINAL.ckpt \
  --seeds 0-39 --n-stoch 8 --max-steps 400 --thr 0.9
echo "SBATCH-DONE"
