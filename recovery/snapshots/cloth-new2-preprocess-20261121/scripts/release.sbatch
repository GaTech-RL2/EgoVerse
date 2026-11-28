#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
unset VIRTUAL_ENV PYTHONPATH
R=/coc/flash7/cxin34/cloth-new2-scratch-20261121
PY=/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python
"$PY" "$R/scripts/assemble_release.py"
"$PY" "$R/scripts/visual_qc.py"
