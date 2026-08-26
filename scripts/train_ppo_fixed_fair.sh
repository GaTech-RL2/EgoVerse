#!/bin/bash
#SBATCH --job-name=ppoFIXfair
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

# FAIR PPO on the FIXED-TARGET task (the original goal's task; BC baseline ~0.72,
# needs only +0.13 to clear >0.85 — unlike circle_3000's 0.44). Applies the
# corrected methodology (controlled lr 4e-6, joint trust region ~0.12, light
# exploration, det+stoch eval) that was previously only run on circle_3000, and
# turns the ORIENTATION curriculum ON (anneal rotation error 0.4->pi) to train the
# diagnosed bottleneck: the ~13 mis-rotated "stubborn" seeds. NB the prior
# fixed-target "0.785 ceiling" came from THROTTLED lr=1e-6 runs — never fair-tuned.
# Required env: ARM. Optional: CKPT, RESUME, LR, ANCHOR, ENT, TKL, ITERS.
ARM=${ARM:?set ARM}
CKPT=${CKPT:-/coc/flash7/paphiwetsa3/fpo_snapshots/gmmBCpaper_ppo_bootstrap.ckpt}
LR=${LR:-4e-6}
ANCHOR=${ANCHOR:-0.1}   # protect the ~27/40 already-solved seeds while hard seeds improve
ENT=${ENT:-0.002}
TKL=${TKL:-0.015}
ITERS=${ITERS:-300}
CURR=${CURR:-1}            # 1=orientation reverse-curriculum on; 0=fixed full-difficulty
START_RESUME=${RESUME:-}   # optional: warm-start policy+value from a prior PPO ckpt

if [ ! -f "$CKPT" ]; then echo "MISSING fixed-target BC ckpt $CKPT"; exit 1; fi

OUT=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/ppoFIX_${ARM}
mkdir -p "$OUT"
# requeue-safe: prefer this ARM's own latest iter; else the optional warm-start.
RES=""
LAST=$(ls -t ${OUT}/ppo_iter*.pt 2>/dev/null | head -1)
if [ -n "$LAST" ]; then RES="--resume $LAST"; echo "RESUMING $ARM from $LAST";
elif [ -n "$START_RESUME" ]; then RES="--resume $START_RESUME"; echo "WARM-START $ARM from $START_RESUME"; fi

# NOTE: NO --goal override -> default fixed goal (256,256,0.785). eval = 40 fixed
# seeds (det+stoch). CURR=1 -> orientation reverse-curriculum (trains hard rotation).
CUR=""
if [ "$CURR" = "1" ]; then
  CUR="--curriculum --cur-r-start 50 --cur-r-end 260 --cur-anneal-iters 70 --cur-ori-start 0.4 --cur-ori-end 3.14159"
fi
srun --kill-on-bad-exit=1 python scripts/train_ppo_gmm.py \
  --ckpt "$CKPT" --out "$OUT" \
  --iters "$ITERS" --seeds-per-iter 48 --max-steps 400 \
  --epochs 4 --minibatch 64 --lr "$LR" --vlr 1e-3 \
  --eps-clip 0.1 --ent-coef "$ENT" --vcoef 0.5 --target-kl "$TKL" \
  --bc-anchor-coef "$ANCHOR" \
  $CUR \
  --shape-w 0.0 --gamma 0.99 --lam 0.95 --critic-warmup 4 \
  --eval-every 5 --eval-nseeds 40 --eval-det-seed 1234 \
  $RES
echo "PPO_EXIT=$? ARM=$ARM CKPT=$CKPT LR=$LR ANCHOR=$ANCHOR ENT=$ENT RESUME=$START_RESUME"
