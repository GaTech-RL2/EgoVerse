#!/usr/bin/env bash
set -euo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libglvnd0 libglib2.0-0
python3 -m venv --system-site-packages /tmp/emimic
source /tmp/emimic/bin/activate
python -m pip install --quiet boto3
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=/opt/astra-project
python -m astra_reversal.complex_manipulation.xiaomi_selector_worker \
    --stage "$1" --manifest /tmp/xiaomi_manifest.json \
    --protocol /tmp/xiaomi_observation_protocol.json --evaluator observation_audit \
    --maximum-worker-seconds 2400
