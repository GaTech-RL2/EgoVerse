#!/bin/bash
#SBATCH --job-name=dagger_smoke
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/dagger_smoke_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/dagger_smoke_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
export DAGGER_SANITY=1
export DESC=chunkB_dagger_smoke
export MODEL=hnet_pushshapes_fused_windowed_resnet
export OBS=sim
export CHUNK_ROLLOUT=true
export CHUNK_SIZE=32
export DAGGER=true
export ROLLOUT_MODE=chunk_openloop
export EVAL_CHUNK_K=32
export SMOKE=1
export EPOCHS=2
export VALEVERY=2
export MAXW=8
bash scripts/train_windowed.sh
