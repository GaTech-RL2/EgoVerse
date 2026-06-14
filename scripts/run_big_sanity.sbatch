#!/bin/bash
#SBATCH --job-name=bigSanity
#SBATCH --time=00:40:00
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bigSanity_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/bigSanity_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export CKPT=${CKPT:-/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetCo_nc3/bc_rnn_hnet_cotrain_2026-06-06_02-55-53/checkpoints/ep1200.ckpt}
export OUTDIR=${OUTDIR:-/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/bcrnnHnetCo_nc3/bc_rnn_hnet_cotrain_2026-06-06_02-55-53/videos/small_tf_overlay}
export DBG=1; srun python scripts/tf_overlay_big.py
echo "OVL_EXIT=$?"
ls -la $OUTDIR
