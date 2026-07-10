#!/bin/bash
#SBATCH --job-name=bf_map_ckpt
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --qos=short
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=bf_map_ckpt_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=.
OLD=/coc/flash7/paphiwetsa3/projects/EgoVerse-gmm-dualstream/eval_indomain/prdec_h200_624eval_parsnap.ckpt
TPL=$(ls -t logs/debug/bf_overfit_prdec_*/checkpoints/last.ckpt | head -1)
srun .venv/bin/python scripts/map_old_ckpt.py "$OLD" "$TPL" bf_prdec_mapped624.ckpt
