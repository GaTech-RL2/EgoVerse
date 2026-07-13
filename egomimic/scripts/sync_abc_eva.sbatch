#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=sync_abc_eva
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/sync_abc_eva_%j.log

# Stage 1 of the abc-eva finetune chain: mirror the 18,456 abc eva episodes
# into the shared dataset folder (skips already-present episodes).

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"

"$PY" -u egomimic/scripts/sync_abc_eva.py
