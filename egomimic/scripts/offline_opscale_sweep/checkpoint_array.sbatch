#!/bin/bash
#SBATCH --account=loaner
#SBATCH --partition=rl2
#SBATCH --exclude=gpu21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=100G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --chdir=/workspace/users/ani-cheluva/EgoVerse
set -euo pipefail
source emimic/bin/activate
bash -n egomimic/scripts/offline_opscale_sweep/worker.sh
srun --ntasks=1 --cpu-bind=cores \
  bash egomimic/scripts/offline_opscale_sweep/worker.sh "$@"
