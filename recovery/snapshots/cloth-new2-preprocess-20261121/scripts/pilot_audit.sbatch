#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
unset PYTHONPATH VIRTUAL_ENV
P=/coc/flash7/cxin34/cloth-new2-scratch-20261121
PY=/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python
"$PY" "$P/scripts/audit_item.py" 0
"$PY" "$P/scripts/visual_qc.py"
