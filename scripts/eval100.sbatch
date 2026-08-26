#!/bin/bash
#SBATCH --job-name=ppoFIXeval
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/ppoFIXeval_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/ppoFIXeval_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1

# Clean 100-seed eval (fixed target = default goal) of a given checkpoint, to get
# a trustworthy det/stoch coverage vs the noisy 40-seed in-run spikes.
RESUME=${RESUME:?set RESUME to the ppo ckpt to eval}
NSEEDS=${NSEEDS:-100}
BC=/coc/flash7/paphiwetsa3/fpo_snapshots/gmmBCpaper_ppo_bootstrap.ckpt

echo "EVAL ckpt=$RESUME nseeds=$NSEEDS"
srun --kill-on-bad-exit=1 python scripts/train_ppo_gmm.py \
  --ckpt "$BC" --resume "$RESUME" --out /tmp/ppofixeval_$SLURM_JOB_ID \
  --eval-only --eval-nseeds "$NSEEDS" --eval-det-seed 1234
echo "EVAL_EXIT=$?"
