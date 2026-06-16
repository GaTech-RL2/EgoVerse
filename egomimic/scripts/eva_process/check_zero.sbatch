#!/bin/bash
#SBATCH --job-name=ray_zarr_convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12         # Ray will automatically see and use these 24 cores
#SBATCH --mem-per-cpu=8G           # 192GB total RAM to handle 24 concurrent conversions
#SBATCH --time=8:00:00
#SBATCH -A gts-dxu345-rl2
#SBATCH --output=logs/check_zero/%j.out

cd /storage/home/hcoda1/4/paphiwetsa3/r-dxu345-0/projects/EgoVerse
source ~/.bashrc
source .venv/bin/activate && export PATH="/storage/project/r-dxu345-0/paphiwetsa3/install/bin:$PATH"

python egomimic/scripts/eva_process/check_zero.py --config-name=train_zarr_cartesian_pi data=cotrain_pi_lang norm_stats.sample_frac=0.001