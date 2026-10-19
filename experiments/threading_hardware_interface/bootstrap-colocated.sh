#!/usr/bin/env bash
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
mkdir -p /workspace/deepseek-serving
cd /workspace/deepseek-serving
base64 --decode /tmp/hardware-payload.tar.gz.b64 > /tmp/serving-payload.tar.gz
echo "$PAYLOAD_SHA256  /tmp/serving-payload.tar.gz" | sha256sum --check --status
tar xzf /tmp/serving-payload.tar.gz
mkdir -p "$HARDWARE_SERVING_CACHE/colocated/$HARDWARE_WORKFLOW"
apt-get update -qq > "$HARDWARE_SERVING_CACHE/colocated/$HARDWARE_WORKFLOW/host-apt.log" 2>&1
apt-get install -y -qq --no-install-recommends skopeo umoci >> "$HARDWARE_SERVING_CACHE/colocated/$HARDWARE_WORKFLOW/host-apt.log" 2>&1 || { tail -n 40 "$HARDWARE_SERVING_CACHE/colocated/$HARDWARE_WORKFLOW/host-apt.log"; exit 1; }
ulimit -l unlimited 2>/dev/null || true
python3 -m experiments.threading_hardware_interface.colocated
