#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=norm_abc_stat
#SBATCH --output=/storage/project/r-dxu345-0/agao81/EgoVerse/logs/norm_abc_stationery_rl2yam_%j.log

# Norm stats for the stationery BC run, with the rl2 lab YAM station's overhead
# extrinsics (extrinsics_key=rl2yam) baked into the proprio conversion. The
# wristframe ACTION stats are extrinsics-invariant; the cam-frame proprio
# ee_pose stats are NOT, so neither norm_stats/abc_eva_6d (x5Dec13_2) nor
# norm_stats/abc_eva_6d_foldviz (abc_fold_viz) is the right distribution here.
# Computed over exactly the run's data config (all 2,066 stationery episodes).
# CPU only — the norm-mode keymap strips camera and annotation keys.

set -euo pipefail
cd /storage/project/r-dxu345-0/agao81/EgoVerse/.claude/worktrees/abc-stationery-pi
PY=/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin/python
export PATH="/storage/home/hcoda1/5/agao81/r-dxu345-0/EgoVerse/emimic/bin:$PATH"
export PYTHONPATH="$PWD"
OUT=/storage/project/r-dxu345-0/agao81/norm_stats/abc_stationery_rl2yam
mkdir -p "$OUT"

echo "=== commit=$(git rev-parse --short HEAD) node=$SLURMD_NODENAME $(date) ==="
"$PY" -u egomimic/scripts/precompute_norm_stats.py \
  --data abc_stationery_pi_6d \
  --model pi0.5_bc_abc_eva_6d \
  --sample-frac 0.2 \
  --max-samples 2000000 \
  --num-workers 22 \
  --out "$OUT"
echo "=== NORM EXIT $? ($(date)) ==="
ls -la "$OUT/norm_stats"
