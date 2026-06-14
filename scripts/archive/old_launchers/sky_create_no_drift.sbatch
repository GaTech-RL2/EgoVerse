#!/bin/bash
#SBATCH --job-name=create_no_drift
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/create_no_drift_%j.log
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/create_no_drift_%j.log
#SBATCH --partition=overcap
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=overcap

source /coc/flash7/paphiwetsa3/projects/EgoVerse2/.venv/bin/activate
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

/coc/flash7/paphiwetsa3/projects/EgoVerse2/.venv/bin/python3.11 \
  /coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/create_no_drift.py
