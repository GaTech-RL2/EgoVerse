#!/usr/bin/env bash
set -euo pipefail

# Serve the aria_egoposer whole-body image policy on this machine's GPU.
# Loads the baked DataSchematic + norm stats from the .ckpt; num_inference_steps is
# forced to 10 at serving. Pushes metadata (camera_keys / proprio_keys / action_dim 49 /
# action_horizon 32) to the receiver on connect.
#
# Usage:
#   bash serve_aria_egoposer.sh                      # legacy hier, port 8000
#   VARIANT=vanilla bash serve_aria_egoposer.sh      # legacy vanilla
#   VARIANT=firm_v1 bash serve_aria_egoposer.sh      # clean _firm V1 (recommended)
#   VARIANT=firm_v2 PORT=8001 bash serve_aria_egoposer.sh   # V2 hist+traj on :8001
#   VARIANT=crop100_2k bash serve_aria_egoposer.sh         # R4 primary (needs updated hpt.py)
#   VARIANT=dino100_2k PORT=8001 bash serve_aria_egoposer.sh # R4 DINO A/B
#   CKPT=path/to/last.ckpt bash serve_aria_egoposer.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VARIANT="${VARIANT:-hier}"
PORT="${PORT:-8000}"

if [[ -z "${CKPT:-}" ]]; then
  case "$VARIANT" in
    hier|vanilla)
      CKPT="checkpoints/aria_egoposer/${VARIANT}/checkpoints/last.ckpt"
      ;;
    firm_v1|firm|firm_vanilla|v1)
      CKPT="checkpoints/aria_egoposer_firm/vanilla/checkpoints/last.ckpt"
      ;;
    firm_v2|v2|v2_hist_traj|hist_traj)
      CKPT="checkpoints/aria_egoposer_firm_v2/v2_hist_traj/checkpoints/last.ckpt"
      ;;
    crop100_2k|crop100|r4_primary)
      CKPT="checkpoints/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt"
      ;;
    dino100_2k|dino100)
      CKPT="checkpoints/aria_egoposer_firm/dino100_2k/checkpoints/last.ckpt"
      ;;
    dino_neck_2k|dino_neck)
      CKPT="checkpoints/aria_egoposer_firm/dino_neck_2k/checkpoints/last.ckpt"
      ;;
    dino_lora_2k|dino_lora)
      CKPT="checkpoints/aria_egoposer_firm/dino_lora_2k/checkpoints/last.ckpt"
      ;;
    *)
      CKPT="checkpoints/aria_egoposer/${VARIANT}/checkpoints/last.ckpt"
      ;;
  esac
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  case "$VARIANT" in
    firm_v1|firm|firm_vanilla|v1|firm_v2|v2|v2_hist_traj|hist_traj|crop100_2k|crop100|r4_primary|dino100_2k|dino100|dino_neck_2k|dino_neck|dino_lora_2k|dino_lora)
      echo "Run:  bash pull_aria_egoposer_firm.sh   (downloads from Skynet)"
      ;;
    *)
      echo "Run:  bash pull_aria_egoposer.sh   (downloads from Skynet)"
      ;;
  esac
  exit 1
fi

# Use the uv venv (not conda); avoid conda lib injection.
source emimic/bin/activate
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Serving $CKPT on ws://0.0.0.0:${PORT}"
exec python egomimic/scripts/serve_policy.py --checkpoint "$CKPT" --port "$PORT"
