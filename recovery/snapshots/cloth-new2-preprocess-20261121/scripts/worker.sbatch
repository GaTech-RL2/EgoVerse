#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset VIRTUAL_ENV PYTHONPATH
exec /coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python /coc/flash7/cxin34/cloth-new2-scratch-20261121/scripts/worker.py "${SLURM_ARRAY_TASK_ID:-0}"
