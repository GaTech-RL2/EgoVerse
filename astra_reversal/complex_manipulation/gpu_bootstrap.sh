#!/usr/bin/env bash
set -euo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libglvnd0 libglib2.0-0
python3 -m venv --system-site-packages /tmp/emimic
source /tmp/emimic/bin/activate
python -m pip install --quiet boto3
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYNPUT_BACKEND=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export OPENPI_DATA_HOME=/opt/astra-complex/model-cache
export PYTHONUNBUFFERED=1
python /tmp/worker.py \
    --stage astra-complex-20261006-stage-1 \
    --manifest /tmp/release_manifest.json \
    --project /opt/astra-project \
    --output /opt/astra-results \
    --smoke-actions 5 \
    --maximum-worker-seconds 790
