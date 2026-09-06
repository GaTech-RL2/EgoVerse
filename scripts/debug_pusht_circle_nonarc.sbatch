#!/bin/bash
#SBATCH --job-name=pusht_circle_debug_nonarc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -p gpu-l40s
set -euo pipefail
cd /storage/project/r-dxu345-0/acheluva3/Egoverse-arc
source .venv/bin/activate
export PYTHONPATH=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc:${PYTHONPATH:-}

python egomimic/trainHydra.py \
  name=pusht_circle_debug \
  description=nonarc \
  model=pusht/circle \
  data=pushshapes/pusht/circle \
  evaluator=hpt/pusht \
  trainer=ddp/debug \
  logger=csv/base \
  norm_stats.sample_frac=0.2 \
  norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc/logs/pusht_circle_debug/nonarc_2026-08-14_23-51-42/norm_stats/norm_stats.json \
  launch_params.gpus_per_node=1 \
  launch_params.nodes=1
