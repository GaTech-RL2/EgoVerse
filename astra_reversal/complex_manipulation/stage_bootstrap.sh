#!/usr/bin/env bash
set -euo pipefail
set +x
python3 -m venv --system-site-packages /tmp/emimic
source /tmp/emimic/bin/activate
python -m pip install --quiet uv boto3 huggingface_hub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export GIT_LFS_SKIP_SMUDGE=1
export UV_CACHE_DIR=/tmp/astra-complex-uv-cache
python /tmp/stage_robocasa.py --manifest /tmp/release_manifest.json --upload
