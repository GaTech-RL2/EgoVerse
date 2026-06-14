#!/bin/bash
#SBATCH -A rl2-lab
#SBATCH --partition=rl2-lab
#SBATCH --qos=short
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=zarr_decode
#SBATCH -o /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/zarr_decode_%j.out
#SBATCH -e /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/hpt_data_ablation/zarr_decode_%j.err
set -euxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate

python scripts/zarr_decode_images.py \
  /coc/flash7/paphiwetsa3/datasets/new_circle_3 \
  /coc/flash7/paphiwetsa3/datasets/new_circle_3_fast
