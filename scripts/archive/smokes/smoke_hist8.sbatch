#!/bin/bash
#SBATCH --job-name=smokeHist8
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smokeHist8_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smokeHist8_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
# 2-epoch debug closed-loop smoke: obs-history N=8 + GMM on run-B config.
# Drives train_windowed_gmm.sh via env knobs. CHUNK_LOSS_TRACE prints the
# per-epoch GMM NLL to stdout (wandb buffers on_epoch metrics).
export WANDB_MODE=offline
export CHUNK_LOSS_TRACE=1
export OBS=sim
export CHUNK_ROLLOUT=true
export CHUNK_SIZE=32
export MODEL=hnet_pushshapes_fused_windowed_resnet
export ACTION_HEAD=gmm
export GMM_MODES=5
export N_OBS_HISTORY=8
export ROLLOUT_MODE=chunk_openloop
export EVAL_CHUNK_K=32
export MWS=96
export MAXW=16
export EPOCHS=2
export VALEVERY=2
export DESC=smoke_chunkC_hist8_gmm
# Short fixed schedule so K stays small (fast) but >1 (exercises multi-chunk
# history). KFIX=2 -> 2 chunks/window every epoch.
export KFIX=2
bash scripts/train_windowed_gmm.sh
echo "SMOKE_EXIT=$?"
