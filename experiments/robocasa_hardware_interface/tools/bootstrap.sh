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
cat > /tmp/hardware-apt.list <<'APT'
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20250901T000000Z/ bookworm main
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20250901T000000Z/ bookworm-security main
APT
APT_OPTIONS=(-o Dir::Etc::sourcelist=/tmp/hardware-apt.list -o Dir::Etc::sourceparts=- -o Acquire::Retries=3)
apt-get "${APT_OPTIONS[@]}" update -qq > artifacts/runtime/apt-install.log 2>&1
apt-get "${APT_OPTIONS[@]}" install -y -qq --no-install-recommends git cmake g++ libgl1 libegl1 libglib2.0-0 libosmesa6 libseccomp2 ffmpeg unzip >> artifacts/runtime/apt-install.log 2>&1 || { tail -n 50 artifacts/runtime/apt-install.log; exit 1; }
for attempt in $(seq 1 900); do
    if [[ -f /osmo/run/workspace/payload.tar.gz ]] && [[ "$(sha256sum /osmo/run/workspace/payload.tar.gz | cut -d' ' -f1)" == "$PAYLOAD_SHA256" ]]; then break; fi
    if [[ "$attempt" == 900 ]]; then exit 2; fi
    sleep 2
done
tar xzf /osmo/run/workspace/payload.tar.gz
python3 -m venv emimic
source emimic/bin/activate
python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1 boto3==1.34.162 pyyaml==6.0.2
bootstrap_failure() {
    local result=$?
    if [[ "$result" != 0 && ! -f artifacts/worker-entered.json ]]; then
        python experiments/robocasa_hardware_interface/tools/archive_failure.py || true
    fi
    exit "$result"
}
trap bootstrap_failure EXIT
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
if [[ "${HARDWARE_STAGE:-commission}" != commission ]]; then
    for attempt in $(seq 1 300); do
        if [[ -s "$HARDWARE_API_KEY_FILE" ]]; then chmod 600 "$HARDWARE_API_KEY_FILE"; break; fi
        if [[ "$attempt" == 300 ]]; then exit 3; fi
        sleep 2
    done
fi
python -m astra_reversal.hardware_interface.robocasa_worker
