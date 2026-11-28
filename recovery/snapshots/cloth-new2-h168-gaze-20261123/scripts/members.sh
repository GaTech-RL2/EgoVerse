#!/bin/bash
set -euo pipefail
export GAZE_ROOT=/coc/flash7/cxin34/cloth-new2-h168-gaze-20261123 TORCH_HOME=/coc/flash7/cxin34/cloth-new2-h168-gaze-20261123/torch PYTHONPATH=/coc/flash7/cxin34/cloth-new2-h168-gaze-20261123/scripts PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd /coc/flash7/cxin34/cloth-new2-h168-gaze-20261123
test -s reports/first-independent-audit.json
SEED=$((42+SLURM_ARRAY_TASK_ID))
/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python scripts/train.py --split splits/split_final.json --seed "$SEED" --epochs 30 --out runs/s"$SEED"
