#!/usr/bin/env bash
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${HARDWARE_RENDER_BACKEND:-egl}" PYOPENGL_PLATFORM="${HARDWARE_RENDER_BACKEND:-egl}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p /workspace/threading-study/artifacts/runtime /osmo/run/workspace
cd /workspace/threading-study
if [[ "${HARDWARE_BOOTSTRAP_FILE_LOG:-0}" == 1 ]]; then
    # The native chroot deliberately has no host /proc or /dev/fd.
    exec > bootstrap.log 2>&1
else
    exec > >(tee bootstrap.log) 2>&1
fi
base64 --decode /tmp/hardware-payload.tar.gz.b64 > /osmo/run/workspace/threading-payload.tar.gz
echo "$PAYLOAD_SHA256  /osmo/run/workspace/threading-payload.tar.gz" | sha256sum --check --status
tar xzf /osmo/run/workspace/threading-payload.tar.gz
if [[ "${HARDWARE_STAGE:-commission}" == pilot ]]; then
    install -m 600 /run/hardware-inference/inference_api_key "$HARDWARE_API_KEY_FILE"
fi
cat > /tmp/threading-apt.list <<'APT'
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20240901T000000Z/ bullseye main
deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20240901T000000Z/ bullseye-security main
APT
APT_OPTIONS=(-o Dir::Etc::sourcelist=/tmp/threading-apt.list -o Dir::Etc::sourceparts=- -o Acquire::Retries=3)
apt-get "${APT_OPTIONS[@]}" update -qq > apt-install.log 2>&1
apt-get "${APT_OPTIONS[@]}" install -y -qq --no-install-recommends git libgl1 libegl1 libglib2.0-0 libosmesa6 libseccomp2 ffmpeg >> apt-install.log 2>&1 || { tail -n 60 apt-install.log; exit 1; }
python3 -m venv emimic
source emimic/bin/activate
python -m pip install pip==24.0 setuptools==68.2.2 wheel==0.41.3 > dependency-install.log 2>&1
python -m pip install -r experiments/threading_hardware_interface/requirements.txt >> dependency-install.log 2>&1 || { tail -n 80 dependency-install.log; exit 1; }
git init -q upstream-robosuite
git -C upstream-robosuite remote add origin https://github.com/ARISE-Initiative/robosuite.git
git -C upstream-robosuite fetch --depth 1 origin c848ca848020d0c4ccdd10c5056bd06f2a195ba2
git -C upstream-robosuite checkout --detach FETCH_HEAD
git init -q upstream-mimicgen
git -C upstream-mimicgen remote add origin https://github.com/NVlabs/mimicgen.git
git -C upstream-mimicgen fetch --depth 1 origin 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb
git -C upstream-mimicgen checkout --detach FETCH_HEAD
python -m pip install --no-deps -e upstream-robosuite -e upstream-mimicgen >> dependency-install.log 2>&1 || { tail -n 80 dependency-install.log; exit 1; }
python --version > artifacts/runtime/python.txt
python -m pip freeze --all > artifacts/runtime/pip-freeze.txt
if [[ "$MUJOCO_GL" == osmesa ]]; then
    echo 'CPU rendering: OSMesa; no simulator GPU allocation.' > artifacts/runtime/gpu.txt
else
    nvidia-smi > artifacts/runtime/gpu.txt
fi
python -m pytest --confcutdir=tests/unit/hardware_interface tests/unit/hardware_interface -q --junitxml=artifacts/runtime/unit-tests.xml > artifacts/runtime/unit-tests.log 2>&1 || { cat artifacts/runtime/unit-tests.log; exit 1; }
if [[ "${HARDWARE_STAGE:-commission}" == pilot ]]; then
    python -m experiments.threading_hardware_interface.campaign
else
    python -m astra_reversal.hardware_interface.threading_worker commission
fi
