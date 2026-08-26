#!/bin/bash
#SBATCH --job-name=ppoC3kLONG
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

# DEFINITIVE long-horizon test: does on-policy PPO EVER improve coverage given
# (a) much longer horizon (~10M env frames), (b) real EXPLORATION (entropy bonus,
# the suspected bottleneck — a sharp BC explores too narrowly to find the hard
# rotation maneuver), and (c) NO curriculum (FIXED full-difficulty random init ->
# train_cov is a CLEAN, comparable "is RL improving?" signal, not confounded by a
# ramping task). Watch whether train_cov / eval ever trend UP.
# Required env: ARM, LR, ANCHOR, ENT. Optional: ITERS, TKL.
ARM=${ARM:?set ARM}
LR=${LR:?set LR}
ANCHOR=${ANCHOR:?set ANCHOR}
ENT=${ENT:?set ENT (entropy coef for exploration)}
ITERS=${ITERS:-500}
TKL=${TKL:-0.015}

SNAP=/coc/flash7/paphiwetsa3/fpo_snapshots/gmmBC_c3000_ppo_bootstrap.ckpt
if [ ! -f "$SNAP" ]; then echo "MISSING BC bootstrap $SNAP"; exit 1; fi

OUT=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/ppoLONG_${ARM}
mkdir -p "$OUT"
RESUME=""
LAST=$(ls -t ${OUT}/ppo_iter*.pt 2>/dev/null | head -1)
if [ -n "$LAST" ]; then RESUME="--resume $LAST"; echo "RESUMING $ARM from $LAST"; fi

# NOTE: no --curriculum -> fixed full-difficulty varied-goal random init.
srun --kill-on-bad-exit=1 python scripts/train_ppo_gmm.py \
  --ckpt "$SNAP" --out "$OUT" \
  --iters "$ITERS" --seeds-per-iter 48 --max-steps 400 \
  --epochs 4 --minibatch 64 --lr "$LR" --vlr 1e-3 \
  --eps-clip 0.1 --ent-coef "$ENT" --vcoef 0.5 --target-kl "$TKL" \
  --bc-anchor-coef "$ANCHOR" \
  --shape-w 0.0 --gamma 0.99 --lam 0.95 --critic-warmup 4 \
  --goal "" --eval-every 10 --eval-nseeds 40 --eval-det-seed 1234 \
  $RESUME
echo "PPO_EXIT=$? ARM=$ARM LR=$LR ANCHOR=$ANCHOR ENT=$ENT"
