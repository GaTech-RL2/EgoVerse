#!/usr/bin/env bash
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONHASHSEED=0
mkdir -p /workspace/hardware-study/artifacts/runtime /osmo/run/workspace
cd /workspace/hardware-study
exec > >(tee artifacts/runtime/bootstrap.log) 2>&1
bootstrap_failure() {
    local result=$?
    if [[ "$result" != 0 ]]; then
        echo "RoboCasa bootstrap/worker exited with code $result"
        for log in apt-install.log dependency-install.log wheel-build.log unit-tests.log; do
            if [[ -f "artifacts/runtime/$log" ]]; then
                echo "Last diagnostic lines from $log"
                tail -n 60 "artifacts/runtime/$log"
            fi
        done
        if [[ "${HARDWARE_STAGE:-commission}" != staging-check && ! -f artifacts/worker-entered.json && -f experiments/robocasa_hardware_interface/tools/archive_failure.py ]]; then
            python experiments/robocasa_hardware_interface/tools/archive_failure.py || true
        fi
    fi
    exit "$result"
}
trap bootstrap_failure EXIT
base64 --decode /tmp/hardware-payload.tar.gz.b64 > /osmo/run/workspace/payload.tar.gz
echo "$PAYLOAD_SHA256  /osmo/run/workspace/payload.tar.gz" | sha256sum --check --status
tar xzf /osmo/run/workspace/payload.tar.gz
if [[ "${HARDWARE_STAGE:-commission}" != commission ]]; then
    # Secret volumes can contain symlinks and be read-only. The provider uses
    # O_NOFOLLOW and requires a regular 0600 file outside archives and scratch.
    install -m 600 /run/hardware-inference/inference_api_key "$HARDWARE_API_KEY_FILE"
fi
python3 - <<'PY'
import json, os, stat
from pathlib import Path
stage = os.environ.get('HARDWARE_STAGE', 'commission')
if stage != 'commission':
    fd = os.open(os.environ['HARDWARE_API_KEY_FILE'], os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise RuntimeError('api_key_file_not_private')
        key = stream.read(4097).strip()
        if not key or len(key) > 4096:
            raise RuntimeError('api_key_file_invalid')
receipt = {'event': 'inputs_ready', 'source_commit': os.environ['HARDWARE_SOURCE_COMMIT'],
           'payload_sha256': os.environ['PAYLOAD_SHA256'], 'source_materialized': True,
           'inference_key_ready': stage != 'commission', 'stage': stage,
           'model_calls': 0}
Path('artifacts/runtime/input-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
PY
if [[ "${HARDWARE_STAGE:-commission}" == staging-check ]]; then exit 0; fi
# The digest-pinned image contains the 2025-09-29 Debian base. An older
# snapshot provides libc6-dev deb12u10 against the image's libc6 deb12u13.
# Both package resolutions were checked on OSMO before selecting this date.
cat > /tmp/hardware-apt.list <<'APT'
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20251001T000000Z/ bookworm main
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20251001T000000Z/ bookworm-security main
APT
APT_OPTIONS=(-o Dir::Etc::sourcelist=/tmp/hardware-apt.list -o Dir::Etc::sourceparts=- -o Acquire::Retries=3)
apt-get "${APT_OPTIONS[@]}" update -qq > artifacts/runtime/apt-install.log 2>&1
apt-get "${APT_OPTIONS[@]}" install -y --no-install-recommends git cmake g++ libgl1 libegl1 libglib2.0-0 libosmesa6 libseccomp2 ffmpeg unzip >> artifacts/runtime/apt-install.log 2>&1
python3 -m venv emimic
source emimic/bin/activate
python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1 boto3==1.34.162 pyyaml==6.0.2
git init -q upstream-robosuite
git -C upstream-robosuite remote add origin https://github.com/ARISE-Initiative/robosuite.git
git -C upstream-robosuite fetch --depth 1 origin 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
git -C upstream-robosuite checkout --detach FETCH_HEAD
git init -q upstream-robocasa
git -C upstream-robocasa remote add origin https://github.com/robocasa/robocasa.git
git -C upstream-robocasa fetch --depth 1 origin 456174f62b89b8fca99eaaf33949c29fec9cfc2a
git -C upstream-robocasa checkout --detach FETCH_HEAD
echo 'Installing the official RoboCasa dependencies (CPU PyTorch; Astra inference is remote)'
python -m pip install torch==2.7.1+cpu torchvision==0.22.1+cpu --index-url https://download.pytorch.org/whl/cpu > artifacts/runtime/dependency-install.log 2>&1
python -m pip install -e upstream-robosuite -e upstream-robocasa jsonschema==4.23.0 pytest==8.3.5 matplotlib==3.10.1 imageio-ffmpeg==0.6.0 >> artifacts/runtime/dependency-install.log 2>&1 || { tail -n 70 artifacts/runtime/dependency-install.log; exit 1; }
python -m robocasa.scripts.setup_macros
python experiments/robocasa_hardware_interface/tools/prepare_assets.py
cp /tmp/hardware-apt.list artifacts/runtime/apt-sources.list
python -m pip freeze --all > artifacts/runtime/pip-freeze.txt
python --version > artifacts/runtime/python.txt
dpkg-query -W > artifacts/runtime/dpkg.txt
nvidia-smi --query-gpu=name,driver_version --format=csv > artifacts/runtime/gpu.txt
sed '/^-e /d' artifacts/runtime/pip-freeze.txt > artifacts/runtime/wheel-requirements.txt
python -m pip wheel --no-deps -r artifacts/runtime/wheel-requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu -w artifacts/wheelhouse > artifacts/runtime/wheel-build.log 2>&1 || { tail -n 60 artifacts/runtime/wheel-build.log; exit 1; }
sha256sum artifacts/wheelhouse/* > artifacts/runtime/wheels.sha256
python -m pytest --confcutdir=tests/unit/hardware_interface tests/unit/hardware_interface -q --junitxml=artifacts/runtime/unit-tests.xml > artifacts/runtime/unit-tests.log 2>&1 || { cat artifacts/runtime/unit-tests.log; exit 1; }
python -m astra_reversal.hardware_interface.robocasa_worker
