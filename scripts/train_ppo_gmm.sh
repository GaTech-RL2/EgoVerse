#!/bin/bash
#SBATCH --job-name=ppoGMM
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

# BC checkpoint to bootstrap PPO from (snapshot to a stable =-free path so training
# can't overwrite it mid-run; the '=' in epoch_epoch=N.ckpt also breaks hydra-free
# torch.load paths). Override CKPT_SRC at submit time:
#   sbatch --export=ALL,CKPT_SRC=/path/to/bc.ckpt scripts/train_ppo_gmm.sh
CKPT_SRC=${CKPT_SRC:?set CKPT_SRC to the BC checkpoint to bootstrap from}
SNAP=/coc/flash7/paphiwetsa3/fpo_snapshots/gmmBCpaper_ppo_bootstrap.ckpt
if [ ! -f "$SNAP" ]; then cp "$CKPT_SRC" "$SNAP"; echo "snapshotted $CKPT_SRC -> $SNAP"; fi

OUT=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/ppoGMM
mkdir -p "$OUT"
# requeue-safe: resume policy+value from the latest saved PPO iter on preemption.
RESUME=""
LAST=$(ls -t ${OUT}/ppo_iter*.pt 2>/dev/null | head -1)
if [ -n "$LAST" ]; then RESUME="--resume $LAST"; echo "RESUMING ppo from $LAST"; fi

srun --kill-on-bad-exit=1 python scripts/train_ppo_gmm.py \
  --ckpt "$SNAP" --out "$OUT" \
  --iters 200 --seeds-per-iter 48 --max-steps 400 \
  --epochs 4 --minibatch 64 --lr 1e-6 --vlr 1e-3 \
  --eps-clip 0.1 --ent-coef 0.0 --vcoef 0.5 --target-kl 0.05 \
  --bc-anchor-coef 1.0 \
  --curriculum --cur-r-start 50 --cur-r-end 260 --cur-anneal-iters 70 \
  --shape-w 0.0 --gamma 0.99 --lam 0.95 --critic-warmup 4 \
  --eval-every 5 --eval-nseeds 40 --eval-det-seed 1234 \
  $RESUME
echo "PPO_EXIT=$?"
