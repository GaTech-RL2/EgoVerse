#!/bin/bash
#SBATCH --job-name=chunkB_dagger
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkB_dagger_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/chunkB_dagger_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
export DESC=chunkB_dagger
export MODEL=hnet_pushshapes_fused_windowed_resnet
export OBS=sim
export CHUNK_ROLLOUT=true
export CHUNK_SIZE=32
export DAGGER=true
export ROLLOUT_MODE=chunk_openloop
export EVAL_CHUNK_K=32
export KSCHED='[[0,1],[100,2],[250,4],[400,6]]'
export EPOCHS=600
export VALEVERY=200
export MAXW=128
bash scripts/train_windowed.sh
