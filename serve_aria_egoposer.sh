#!/usr/bin/env bash
set -euo pipefail

# Serve the aria_egoposer whole-body image policy on this machine's GPU.
# Loads the baked DataSchematic + norm stats from the .ckpt; num_inference_steps is
# forced to 10 at serving. Pushes metadata (camera_keys / proprio_keys / action_dim 49 /
# action_horizon 32) to the receiver on connect.
#
# Usage:
#   bash serve_aria_egoposer.sh                 # hier (recommended), port 8000
#   VARIANT=vanilla bash serve_aria_egoposer.sh
#   CKPT=checkpoints/aria_egoposer/hier/checkpoints/epoch_epoch=599.ckpt bash serve_aria_egoposer.sh
#   PORT=8001 bash serve_aria_egoposer.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VARIANT="${VARIANT:-hier}"
PORT="${PORT:-8000}"
CKPT="${CKPT:-checkpoints/aria_egoposer/${VARIANT}/checkpoints/last.ckpt}"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  echo "Run:  bash pull_aria_egoposer.sh   (downloads from Skynet)"
  exit 1
fi

# Use the uv venv (not conda); avoid conda lib injection.
source emimic/bin/activate
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Serving $CKPT on ws://0.0.0.0:${PORT}"
exec python egomimic/scripts/serve_policy.py --checkpoint "$CKPT" --port "$PORT"
