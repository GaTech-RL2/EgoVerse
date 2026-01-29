#!/bin/bash
#SBATCH --partition=hoffman-lab
#SBATCH --account=hoffman-lab
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/process_image/%j.out
cd /nethome/paphiwetsa3/flash/projects/EgoVerse

source .venv/bin/activate

srun --cpus-per-task=12 python egomimic/scripts/visualization_process/process_image.py --data-config /nethome/paphiwetsa3/flash/projects/EgoVerse/egomimic/hydra_configs/data/clothe_eva_aria_mecka.yaml --out-dir egomimic/scripts/visualization_process/fold_clothes_aria_eva_mecka_all
