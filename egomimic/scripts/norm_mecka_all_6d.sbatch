#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=norm_mecka_all_6d
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/norm_mecka_all_6d_%j.log

# CPU-only: compute 0.1-frac quantile norm stats over all SQL lab=mecka
# episodes (shared PACE mirror already holds the data; the resolver syncs
# only the handful of missing episodes). cartesian_6d: 18D action + proprio.
#
# Output: /storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d/norm_stats/norm_stats.json
# Training uses:
#   norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d/norm_stats

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$PY" egomimic/scripts/precompute_norm_stats.py \
  --data mecka_all_pi_6d \
  --model pi0.5_bc_mecka_6d \
  --sample-frac 0.1 \
  --num-workers 22 \
  --out /storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d
