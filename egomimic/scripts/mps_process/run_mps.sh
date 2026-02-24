#!/bin/bash
set -euo pipefail

# --- Cron-safe environment ---
export HOME="/home/ubuntu"
export USER="ubuntu"

# Cron PATH is minimal; include where aria_mps lives.
export PATH="/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PYTHON_BIN="/usr/bin/python3"

echo "[$(date -Is)] run_mps.sh starting"
echo "[$(date -Is)] PATH=$PATH"
echo "[$(date -Is)] python3=$(which python3)"
echo "[$(date -Is)] aria_mps=$(command -v aria_mps || echo 'NOT_FOUND')"

# Fail fast if aria_mps isn't visible (prevents silent batch failures)
if ! command -v aria_mps >/dev/null 2>&1; then
  echo "[$(date -Is)] ERROR: aria_mps not found in PATH"
  exit 1
fi

# --- Credentials ---
export MPS_USER="georgiat_zb658p"
export MPS_PASSWORD="georgiat0001"

# --- Local directory (avoid ~ in cron for safety) ---
LOCAL_DIR="/home/ubuntu/local"
mkdir -p "$LOCAL_DIR"

# --- Run ---
"$PYTHON_BIN" /home/ubuntu/EgoVerse/egomimic/scripts/mps_process/s3_parallel_processor.py \
  --bucket rldb \
  --s3-prefix raw_v2/aria \
  --local-dir "$LOCAL_DIR" \
  --target-size-gb 150 \
  --features HAND_TRACKING SLAM
