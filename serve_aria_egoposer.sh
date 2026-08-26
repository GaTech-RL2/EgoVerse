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
#   VARIANT=d3lora bash serve_aria_egoposer.sh             # FPP A (newest epoch on disk)
#   VARIANT=resnet PORT=8003 bash serve_aria_egoposer.sh   # FPP D (newest epoch on disk)
#   CKPT=path/to/epoch_epoch=1699.ckpt bash serve_aria_egoposer.sh   # pin an exact epoch
#
# FPP variants A-F auto-select the NEWEST epoch_epoch=<N>.ckpt present locally
# (pulled via: TARGET_EPOCH=99999 VARIANTS="A B C D E F" bash pull_fpp_same_epoch.sh).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VARIANT="${VARIANT:-hier}"
PORT="${PORT:-8000}"

# Pick the newest epoch_epoch=<N>.ckpt in a checkpoints dir (highest N). Falls back to
# last.ckpt if no numbered epoch files exist. Prints empty if the dir is missing.
_latest_epoch_ckpt() {
  local ckpt_dir="$1"
  local best="" best_n=-1 f n
  for f in "$ckpt_dir"/epoch_epoch=*.ckpt; do
    [[ -e "$f" ]] || continue
    n="${f##*epoch_epoch=}"; n="${n%.ckpt}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if (( n > best_n )); then best_n="$n"; best="$f"; fi
  done
  if [[ -z "$best" && -f "$ckpt_dir/last.ckpt" ]]; then best="$ckpt_dir/last.ckpt"; fi
  printf '%s' "$best"
}

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
    # FPP HD-era round 2026-07-21 (proprio-dropout-0.9, vision-driven). Exact gate-verified
    # epochs from the guide §1 (higher epoch is NOT necessarily better). Override with CKPT=...
    hd_wam3|hd_wam3_2k|hdA)
      CKPT="checkpoints/aria_fullpp_wam3/fpp_hd_wam3_2k/checkpoints/epoch_epoch=${EPOCH:-1399}.ckpt"
      ;;
    hd_resnet|hd_resnet_2k|hdB)
      CKPT="checkpoints/aria_fullpp/fpp_hd_resnet_2k/checkpoints/epoch_epoch=${EPOCH:-1499}.ckpt"
      ;;
    hd_c|wam3_baseline|wam3@1599)
      CKPT="checkpoints/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=${EPOCH:-1599}.ckpt"
      ;;
    # FPP Round 2026-07-20 — default to the NEWEST epoch on disk (override with CKPT=... or EPOCH=<N>).
    A|a|d3lora|d3lora_2k|d3lora@99)
      CKPT="$(_latest_epoch_ckpt checkpoints/aria_fullpp/fpp_d3lora_2k/checkpoints)"
      ;;
    A399|d3lora@399)
      CKPT="checkpoints/aria_fullpp/fpp_d3lora_2k/checkpoints/epoch_epoch=399.ckpt"
      ;;
    B|b|d3conv|d3conv_2k|d3conv@399)
      CKPT="$(_latest_epoch_ckpt checkpoints/aria_fullpp/fpp_d3conv_2k/checkpoints)"
      ;;
    C|c|wam3|wam3_2k|wam3@999)
      CKPT="$(_latest_epoch_ckpt checkpoints/aria_fullpp_wam3/fpp_wam3_2k/checkpoints)"
      ;;
    D|d|resnet|resnet_2k|resnet@1599)
      CKPT="$(_latest_epoch_ckpt checkpoints/aria_fullpp/fpp_resnet_2k/checkpoints)"
      ;;
    E|e|bare|bare_2k|bare@1399)
      CKPT="$(_latest_epoch_ckpt checkpoints/exp1_bare/fpp_bare_2k/checkpoints)"
      ;;
    F|f|glove|glove_2k|glove@699)
      CKPT="$(_latest_epoch_ckpt checkpoints/exp1_glove/fpp_glove_2k/checkpoints)"
      ;;
    *)
      CKPT="checkpoints/aria_egoposer/${VARIANT}/checkpoints/last.ckpt"
      ;;
  esac
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  case "$VARIANT" in
    hd_wam3|hd_wam3_2k|hdA|hd_resnet|hd_resnet_2k|hdB|hd_c|wam3_baseline|wam3@1599)
      echo "Run:  bash pull_fpp_hd.sh   (downloads HD-era FPP ckpts from Skynet)"
      ;;
    A|a|d3lora|d3lora_2k|d3lora@99|A399|d3lora@399|B|b|d3conv|d3conv_2k|d3conv@399|C|c|wam3|wam3_2k|wam3@999|D|d|resnet|resnet_2k|resnet@1599|E|e|bare|bare_2k|bare@1399|F|f|glove|glove_2k|glove@699)
      echo "Run:  bash pull_fpp_round.sh   (downloads FPP ckpts from Skynet)"
      ;;
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
