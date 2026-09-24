#!/usr/bin/env bash
set -Eeuo pipefail
set +x
debug_failure() {
    local astra_exit=$?
    trap - ERR
    echo "Development worker failed with code $astra_exit; retaining it for up to 15 minutes for diagnosis."
    for astra_attempt in $(seq 1 180); do
        if [[ -f /tmp/astra-recovery-exit.status ]]; then
            exit "$(cat /tmp/astra-recovery-exit.status)"
        fi
        sleep 5
    done
    exit "$astra_exit"
}
trap debug_failure ERR
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
unset PIP_CONSTRAINT UV_CONSTRAINT UV_BUILD_CONSTRAINT PIP_BUILD_CONSTRAINT
mkdir -p /workspace/astra-reversal /osmo/run/workspace
cd /workspace/astra-reversal
exec > >(tee /workspace/astra-reversal/bootstrap.log) 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
apt-get update -qq
apt-get install -y -qq --no-install-recommends git curl ca-certificates libgl1 libegl1 libglib2.0-0 libosmesa6 ffmpeg
curl -LsSf https://astral.sh/uv/0.10.4/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Source and the already-authorized tokenizer arrive through OSMO rsync.
# Wait for a checksum match so a partially uploaded archive is never executed.
for attempt in $(seq 1 900); do
    if [[ -f /osmo/run/workspace/payload.tar.gz ]] && \
       [[ "$(sha256sum /osmo/run/workspace/payload.tar.gz | cut -d' ' -f1)" == "$PAYLOAD_SHA256" ]]; then
        break
    fi
    if [[ "$attempt" == 900 ]]; then exit 2; fi
    sleep 2
done
tar xzf /osmo/run/workspace/payload.tar.gz
uv venv --python 3.11 emimic
source emimic/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r astra_reversal/requirements-libero.txt \
    accelerate==1.10.1 datasets==4.8.5 draccus==0.10.0 einops==0.8.2 \
    huggingface-hub==0.36.2 safetensors==0.8.0 sentencepiece==0.2.2 \
    pillow==12.3.0 pyyaml packaging typing-extensions termcolor gymnasium \
    deepdiff boto3 imageio matplotlib h5py pytest 'diffusers>=0.35.1,<0.36' pydantic omegaconf rich \
    pyserial==3.5 jsonlines==4.0.0 'av>=15,<16' 'wandb>=0.24,<0.25' \
    'transformers @ https://github.com/huggingface/transformers/archive/dcddb970176382c0fcf4521b0c0e6fc15894dfe0.tar.gz'

clone_revision() {
    git init -q "$3"
    git -C "$3" remote add origin "$1"
    git -C "$3" fetch --depth 1 origin "$2"
    git -C "$3" checkout --detach FETCH_HEAD
    test "$(git -C "$3" rev-parse HEAD)" = "$2"
}
clone_revision https://github.com/huggingface/lerobot.git 8fff0fde7c79f23a93d845d1a50e985de01f8b8a astra_reversal/.deps/lerobot
uv pip install --no-deps -e astra_reversal/.deps/lerobot
clone_revision https://github.com/Lifelong-Robot-Learning/LIBERO.git f78abd68ee283de9f9be3c8f7e2a9ad60246e95c astra_reversal/.deps/libero
if [[ "${ASTRA_INCLUDE_OOD:-0}" == "1" ]]; then
    clone_revision https://github.com/QuanyiLi/pi0-text-latent.git 587a6cbf64f16c7b87fa5805dc0ed934192239a4 astra_reversal/.deps/libero-ood
fi
clone_revision https://github.com/Physical-Intelligence/openpi.git 981483dca0fd9acba698fea00aa6e52d56a66c58 external/openpi

# Give source snapshots a real, local revision for the existing run manifests.
git init -q
git add astra_reversal
git -c user.name='Astra experiment snapshot' -c user.email='snapshot@localhost' commit -qm 'Immutable uploaded experiment source'
export PYTHONPATH="$PWD/astra_reversal/.deps/lerobot/src:$PWD/external/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
python -m "${ASTRA_ENTRY_MODULE:-astra_reversal.osmo.experiment}"
