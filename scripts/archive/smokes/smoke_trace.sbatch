#!/bin/bash
#SBATCH --job-name=smoke_trace
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smoke_trace_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/smoke_trace_%j.err
set -uxo pipefail
export CHUNK_LOSS_TRACE=1
export WANDB_MODE=offline
COMMON="OBS=sim CHUNK_ROLLOUT=true CHUNK_SIZE=32 MODEL=hnet_pushshapes_fused_windowed_resnet KFIX=1 SMOKE=1 EPOCHS=6 VALEVERY=100 MAXW=16"
echo "================ GMM TRACE ================"
env $COMMON ACTION_HEAD=gmm GMM_MODES=5 DESC=trace_gmm \
  bash /coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/train_windowed_gmm.sh
echo "================ CONTINUOUS TRACE ================"
env $COMMON DESC=trace_cont \
  bash /coc/flash7/paphiwetsa3/projects/EgoVerse2/scripts/train_windowed_gmm.sh
echo "================ TRACE DONE ================"
