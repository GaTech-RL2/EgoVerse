#!/bin/bash
#SBATCH --job-name=pusht_circle_debug_arc
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
  description=arc \
  model=pusht_arc/circle \
  data=pushshapes/pusht/circle_arc \
  evaluator=hpt/pusht_arc \
  trainer=ddp/debug \
  logger=csv/base \
  norm_stats.sample_frac=0.2 \
  norm_stats.precomputed_norm_path=/storage/project/r-dxu345-0/acheluva3/Egoverse-arc/logs/pusht_circle_debug/arc_2026-08-15_00-07-38/norm_stats/norm_stats.json \
  launch_params.gpus_per_node=1 \
  launch_params.nodes=1
