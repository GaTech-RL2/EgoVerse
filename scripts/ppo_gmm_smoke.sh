#!/bin/bash
set -uxo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
CK=/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/gmmBCpaperSmoke/gmm_bc_paper_smoke_2026-06-18_18-09-18/checkpoints/last.ckpt
python scripts/train_ppo_gmm.py --ckpt "$CK" \
  --out /coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/ppo_gmm_smoke \
  --iters 2 --seeds-per-iter 4 --eval-nseeds 4 --eval-every 1 \
  --critic-warmup 1 --max-steps 400 --minibatch 32
echo "PPO_SMOKE_EXIT=$?"
