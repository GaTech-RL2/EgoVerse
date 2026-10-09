#!/usr/bin/env bash
set -euo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libglvnd0 libglib2.0-0
python3 -m venv --system-site-packages /tmp/emimic
source /tmp/emimic/bin/activate
python -m pip install --quiet uv boto3 huggingface_hub
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export GIT_LFS_SKIP_SMUDGE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/opt/astra-project
export UV_CACHE_DIR=/tmp/astra-xiaomi-uv-cache
if [[ "${1:-}" == "stage" ]]; then
    python -m astra_reversal.complex_manipulation.xiaomi_stage --manifest /tmp/xiaomi_manifest.json
elif [[ "${1:-}" == "evaluate" ]]; then
    python -m astra_reversal.complex_manipulation.xiaomi_worker \
        --stage "$2" --manifest /tmp/xiaomi_manifest.json --protocol /tmp/xiaomi_protocol.json
else
    echo "Expected stage or evaluate mode" >&2
    exit 2
fi
