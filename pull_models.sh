#!/usr/bin/env bash
set -euo pipefail

# ====== config (edit these) ======
REMOTE_USER_HOST="bli678@sky2.cc.gatech.edu"
REMOTE_PATH="/coc/flash7/bli678/Projects/EgoVerse/logs/cup_saucer/hpt_indomain2h_cotrain_2026-03-24_16-26-29"
LOCAL_PATH="./egomimic/robot/models/cup_saucer/hpt_indomain2h_cotrain_2026-03-24_16-26-29"
# =================================

mkdir -p "$LOCAL_PATH"

# Prefer system rsync to avoid OpenSSL/conda mismatch.
# If rsync isn't available in this environment, fall back to tar-over-ssh.
RSYNC_BIN="/usr/bin/rsync"
if [[ ! -x "$RSYNC_BIN" ]]; then
  RSYNC_BIN="$(command -v rsync || true)"
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)

if [[ -n "${RSYNC_BIN:-}" && -x "${RSYNC_BIN}" ]]; then
  echo "Using rsync to pull models..."
  # Run rsync without Conda/mamba library injection
  env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
    "$RSYNC_BIN" -avh --progress --partial --inplace \
    --exclude='***/0/videos/***' \
    --exclude='***/0/wandb/***' \
    "${REMOTE_USER_HOST}:${REMOTE_PATH%/}/" \
    "${LOCAL_PATH%/}/"
else
  echo "rsync not found; falling back to tar over ssh..."
  # Fail fast with a clear message if SSH auth isn't set up.
  if ! ssh "${SSH_OPTS[@]}" "${REMOTE_USER_HOST}" 'echo ssh_ok' >/dev/null 2>&1; then
    echo "SSH auth failed (permission denied). Set up SSH keys/access for ${REMOTE_USER_HOST}." >&2
    echo "Then re-run this script." >&2
    exit 1
  fi
  # Exclusions are approximate equivalents of the original rsync patterns.
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER_HOST}" \
    "cd \"${REMOTE_PATH%/}\" && tar cf - \
      --exclude='*/0/videos/*' \
      --exclude='*/0/wandb/*' \
      ." \
    | tar xf - -C "${LOCAL_PATH%/}/"
fi
