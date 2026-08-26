#!/bin/bash
#SBATCH --job-name=ppoC3kSW
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=scavenger_qos
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=ig-88
#SBATCH --output=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/ppogmm_%x_%j.out
#SBATCH --error=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/sbatch/ppogmm_%x_%j.err
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1

# Corrected fair-retest sweep. Holds EVERYTHING from train_ppo_c3000.sh constant
# EXCEPT the two diagnosed throttles: LR (step size) and ANCHOR (BC restraint).
# Required env: ARM (label/out-subdir), LR, ANCHOR, ITERS.
ARM=${ARM:?set ARM}
LR=${LR:?set LR}
ANCHOR=${ANCHOR:?set ANCHOR}
ITERS=${ITERS:-300}
# logp is MEANed over the 8 chunk positions, so target_kl is on the per-position
# (geometric-mean) KL; true joint per-decision KL ~ 8x larger. target_kl=0.015 ->
# joint trust region ~0.12 (a sane PPO step), vs the old 0.05 (joint ~0.4, too loose).
TKL=${TKL:-0.015}

SNAP=/coc/flash7/paphiwetsa3/fpo_snapshots/gmmBC_c3000_ppo_bootstrap.ckpt
if [ ! -f "$SNAP" ]; then echo "MISSING BC bootstrap $SNAP"; exit 1; fi

OUT=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/ppoSW_${ARM}
mkdir -p "$OUT"
# requeue-safe resume from this ARM's own latest iter on preemption.
RESUME=""
LAST=$(ls -t ${OUT}/ppo_iter*.pt 2>/dev/null | head -1)
if [ -n "$LAST" ]; then RESUME="--resume $LAST"; echo "RESUMING $ARM from $LAST"; fi

srun --kill-on-bad-exit=1 python scripts/train_ppo_gmm.py \
  --ckpt "$SNAP" --out "$OUT" \
  --iters "$ITERS" --seeds-per-iter 48 --max-steps 400 \
  --epochs 4 --minibatch 64 --lr "$LR" --vlr 1e-3 \
  --eps-clip 0.1 --ent-coef 0.0 --vcoef 0.5 --target-kl "$TKL" \
  --bc-anchor-coef "$ANCHOR" \
  --curriculum --cur-r-start 50 --cur-r-end 260 --cur-anneal-iters 70 \
  --shape-w 0.0 --gamma 0.99 --lam 0.95 --critic-warmup 4 \
  --goal "" --eval-every 5 --eval-nseeds 40 --eval-det-seed 1234 \
  $RESUME
echo "PPO_EXIT=$? ARM=$ARM LR=$LR ANCHOR=$ANCHOR"
