#!/bin/bash
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=100G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/workspace/users/ani-cheluva/EgoVerse

set -euo pipefail
OPS="${1:?rung required}"
MANIFEST="${2:?checkpoint manifest required}"
UNSEEN_DIR="${3:?private unseen validation folder required}"
FLAGSHIP="${4:?reviewed flagship episode required}"
UNSEEN_PREVIEW="${5:?unseen preview episode required}"
case "$OPS" in 4|8|12|16) ;; *) exit 2 ;; esac
test -s "$MANIFEST"
test -d "$UNSEEN_DIR"

echo "offline validation sweep job=$SLURM_JOB_ID op=$OPS manifest=$MANIFEST"
echo "unseen_dir=$UNSEEN_DIR unseen_video=$UNSEEN_PREVIEW flagship_video=$FLAGSHIP"
source emimic/bin/activate
python -m py_compile egomimic/scripts/offline_opscale_sweep/record_metrics.py
bash -n egomimic/scripts/offline_opscale_sweep/worker.sh
srun --ntasks=8 --gpus-per-task=h100:1 --gpu-bind=single:1 --cpu-bind=cores \
  bash egomimic/scripts/offline_opscale_sweep/worker.sh \
  "$OPS" "$MANIFEST" "$UNSEEN_DIR" "$FLAGSHIP" "$UNSEEN_PREVIEW"
