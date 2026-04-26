#!/bin/bash
#SBATCH --job-name=dinov3_zarr_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH -A gts-dxu345-rl2
#SBATCH --output=logs/embedding/dinov3_%j.out
#SBATCH --error=logs/embedding/dinov3_%j.err

set -euo pipefail

mkdir -p logs/embedding

cd /storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/projects/EgoVerse2
source ~/.bashrc
source .venv/bin/activate && export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"

python egomimic/scripts/embedding_process/zarr_embedding.py \
    --transform dinov3 \
    --dataset-config-path egomimic/hydra_configs/data/cotrain_pi_lang.yaml \
    --input-keys images.front_1 \
    --batch-size 512 \
    --overwrite
