#!/usr/bin/env bash
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p /workspace/hardware-study /osmo/run/workspace
cd /workspace/hardware-study
exec > >(tee bootstrap.log) 2>&1
apt-get update -qq
apt-get install -y -qq --no-install-recommends git libgl1 libegl1 libglib2.0-0 libosmesa6 libseccomp2 ffmpeg
for attempt in $(seq 1 900); do
    if [[ -f /osmo/run/workspace/payload.tar.gz ]] && [[ "$(sha256sum /osmo/run/workspace/payload.tar.gz | cut -d' ' -f1)" == "$PAYLOAD_SHA256" ]]; then
        break
    fi
    if [[ "$attempt" == 900 ]]; then exit 2; fi
    sleep 2
done
tar xzf /osmo/run/workspace/payload.tar.gz
python3 -m venv emimic
source emimic/bin/activate
python -m pip install pip==24.0 setuptools==68.2.2 wheel==0.41.3
git init -q upstream-libero
git -C upstream-libero remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C upstream-libero fetch --depth 1 origin f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
git -C upstream-libero checkout --detach FETCH_HEAD
test "$(git -C upstream-libero rev-parse HEAD)" = f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
# LIBERO uses torch only to load official reset tensors here. Preserve 1.11.0;
# the CPU wheel is an explicit runtime variant of the upstream CUDA training setup.
python -m pip install -c experiments/libero_hardware_interface/constraints.txt -r upstream-libero/requirements.txt \
    jsonschema==4.19.2 pytest==7.4.4 boto3==1.34.162 imageio==2.31.6 imageio-ffmpeg==0.4.9 h5py==3.8.0 \
    torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e upstream-libero
mkdir -p artifacts/runtime
python -m pip freeze --all > artifacts/runtime/pip-freeze.txt
python --version > artifacts/runtime/python.txt
python -m pip --version >> artifacts/runtime/python.txt
dpkg-query -W > artifacts/runtime/dpkg.txt
nvidia-smi --query-gpu=name,driver_version --format=csv > artifacts/runtime/gpu.txt
git -C upstream-libero submodule status --recursive > artifacts/runtime/libero-submodules.txt
git -C upstream-libero status --porcelain > artifacts/runtime/libero-dirty.txt
# Archive exact installable wheels, not only a version listing. Editable LIBERO
# is supplied by its clean source SHA and never by actor-visible mounts.
sed '/^-e /d' artifacts/runtime/pip-freeze.txt > artifacts/runtime/wheel-requirements.txt
python -m pip wheel --no-deps -r artifacts/runtime/wheel-requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu -w artifacts/wheelhouse
sha256sum artifacts/wheelhouse/* > artifacts/runtime/wheels.sha256
python -m pytest --confcutdir=tests/unit/hardware_interface tests/unit/hardware_interface -q \
    --junitxml=artifacts/runtime/unit-tests.xml > artifacts/runtime/unit-tests.log 2>&1
python -m astra_reversal.hardware_interface.osmo_worker
