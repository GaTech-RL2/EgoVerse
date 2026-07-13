#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=norm_abc_eva_6d
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/norm_abc_eva_6d_%j.log

# Stage 2 of the abc-eva finetune chain: per-(step,dim) quantile norm stats
# over the wristframe-6D abc eva data (20D native with grippers — no proprio
# padding patch needed, unlike mecka). Output consumed by
# train_abc_eva_ft.sbatch.

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"

OUT=/storage/project/r-dxu345-0/agao81/norm_stats/abc_eva_6d
mkdir -p "$OUT"

"$PY" -u egomimic/scripts/precompute_norm_stats.py \
  --data abc_eva_pi_6d \
  --model pi0.5_ft_abc_eva_6d \
  --sample-frac 0.1 \
  --max-samples 2000000 \
  --num-workers 22 \
  --out "$OUT"
