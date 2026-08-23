#!/usr/bin/env bash
set -euo pipefail

ARX5_COMMIT="a8890c9bae94464abd1cb7c5e4da7c4a62104a3a"
OUTPUT_DIR="${1:-dist/arx5}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "ARX5 production wheels must be built on Linux x86_64; current host is $(uname -s) $(uname -m)." >&2
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to build the manylinux ARX5 wheel." >&2
    exit 2
fi

BUILD_DIR="$(mktemp -d)"
cleanup() {
    docker rm -f egomimic-arx5-py311-copy >/dev/null 2>&1 || true
    rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

git clone https://github.com/real-stanford/arx5-sdk.git "${BUILD_DIR}/arx5-sdk"
git -C "${BUILD_DIR}/arx5-sdk" checkout --detach "${ARX5_COMMIT}"
git -C "${BUILD_DIR}/arx5-sdk" apply \
    "${REPO_ROOT}/egomimic/robot/eva/arx5_patches/0001-wait-for-gripper-calibration-readback.patch"

docker build \
    --file "${BUILD_DIR}/arx5-sdk/wheels/Dockerfile.single_ver_x86_64" \
    --build-arg PYTHON_VERSION=cp311-cp311 \
    --tag egomimic-arx5-py311 \
    "${BUILD_DIR}/arx5-sdk"
docker rm -f egomimic-arx5-py311-copy >/dev/null 2>&1 || true
docker create --name egomimic-arx5-py311-copy egomimic-arx5-py311
mkdir -p "${REPO_ROOT}/${OUTPUT_DIR}"
docker cp \
    egomimic-arx5-py311-copy:/root/arx5-sdk/wheelhouse/. \
    "${REPO_ROOT}/${OUTPUT_DIR}/"

WHEEL="$(find "${REPO_ROOT}/${OUTPUT_DIR}" -maxdepth 1 -type f -name '*cp311*manylinux*x86_64.whl' -print -quit)"
if [[ -z "${WHEEL}" ]]; then
    echo "Build completed but no CPython 3.11 x86_64 wheel was produced." >&2
    exit 1
fi
echo "Built ${WHEEL}"
sha256sum "${WHEEL}"
