#!/bin/bash
#SBATCH --job-name=ray_bucket_to_zarr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12         # Ray will automatically see and use these cores
#SBATCH --mem-per-cpu=8G           # 96GB total RAM
#SBATCH --time=8:00:00
#SBATCH -A gts-dxu345-rl2
#SBATCH --output=logs/conversion/ray_bucket_to_zarr_%j.out

cd /storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/projects/EgoVerse3
source ~/.bashrc
source .venv/bin/activate && export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"

# --num-cpus-per-task defaults to 1, meaning 12 episodes process at the same time.
python egomimic/scripts/language_process/bucket_to_zarr_annotation_parallel.py \
    --dataset-config-path egomimic/hydra_configs/data/eva_pi_lang.yaml \
    --bucket s3://rldb/processed_annotations \
    --overwrite
