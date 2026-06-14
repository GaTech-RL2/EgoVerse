#!/bin/bash
set -uo pipefail
cd /coc/flash7/paphiwetsa3/projects/EgoVerse2
source /coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/activate
export PYTHONPATH=. MUJOCO_GL=egl HYDRA_FULL_ERROR=1
echo "===== proof_suite.py ====="
python scratch/proof_suite.py
echo "RC_SUITE=$?"
echo "===== proof_params.py ====="
python scratch/proof_params.py
echo "RC_PARAMS=$?"
