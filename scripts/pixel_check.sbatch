#!/bin/bash
#SBATCH --job-name=pixel_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p cpu-small
#SBATCH --output=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc/out/pixel_check_%j.out
#SBATCH --error=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc/out/pixel_check_%j.err
set -euo pipefail
cd /storage/project/r-dxu345-0/acheluva3/Egoverse-arc
source .venv/bin/activate
python scripts/pixel_check.py
