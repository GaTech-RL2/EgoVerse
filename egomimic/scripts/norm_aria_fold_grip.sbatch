#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=norm_aria_fold_grip
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/norm_aria_fold_grip_%j.log

# Stage 2 of the abc-eva finetune chain: per-(step,dim) quantile norm stats
# over the wristframe-6D aria fold data (20D incl. the keypoint pseudo-gripper).
# padding patch needed, unlike mecka). Output consumed by
# train_aria_fold_grip.sbatch.

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"

OUT=/storage/project/r-dxu345-0/agao81/norm_stats/aria_fold_grip_6d
mkdir -p "$OUT"

"$PY" -u egomimic/scripts/precompute_norm_stats.py \
  --data aria_fold_pi_6d_grip \
  --model pi0.5_ft_aria_fold_grip \
  --sample-frac 0.1 \
  --max-samples 2000000 \
  --num-workers 22 \
  --out "$OUT"
