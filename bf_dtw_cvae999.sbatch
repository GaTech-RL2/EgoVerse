#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --exclude=omgwth,ig-88,baymax
#SBATCH --output=bf_dtw_cvae999_%j.out
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow
export PYTHONPATH=. MUJOCO_GL=egl
# PINNED ep999 snap (provenance rule) — never last.ckpt
.venv/bin/python scripts/bf_dtw_latent.py \
  --ckpt eval_bf/bf_cvae_zs_var4_ep999_snap.ckpt \
  --config-path eval_bf/dtw/bf_cvae_zs_var4_dtwcfg.yaml \
  --n-episodes 20 --evr 0.95 --good-eps 3,7,0
echo DTW_CVAE999_DONE rc=$?
