#!/usr/bin/env bash
set -euo pipefail

# Pull the LATEST checkpoints of ALL pcd/depth policy runs from Skynet.
#
# Unlike pull_dp3_adapt3r.sh (fixed epoch list from the guide), this first
# LISTS what actually exists remotely (training may have continued past the
# guide), then pulls per run directory:
#   * the highest epoch_epoch=N.ckpt
#   * last.ckpt when present
# across logs/RBY1_dp3_pcd1024, logs/RBY1_dp3_pcd (2048), logs/RBY1_adapt3r*,
# plus any other run family whose path contains 'pcd' or 'adapt3r'.
# Also pulls the rby1_teleop_pcd2048_glass dataset (needed to eval DP3-2048:
# GT modes, replay test, dataset-avg reset).
#
# Usage:
#   cd ~/RB_Y1_workspace/EgoVerse
#   read -rs -p "Skynet pw: " SKYNET_PASS; export SKYNET_PASS; echo
#   bash pull_pcd_latest.sh
#
#   LIST_ONLY=1 bash pull_pcd_latest.sh    # just show what exists remotely
#   ALL_EPOCHS=1 bash pull_pcd_latest.sh   # pull every ckpt, not only latest

REMOTE_USER_HOST="${REMOTE_USER_HOST:-czhang883@sky2.cc.gatech.edu}"
REMOTE_REPO="${REMOTE_REPO:-/coc/flash7/czhang883/Documents/EgoVerse}"
REMOTE_SSH_PORT="${REMOTE_SSH_PORT:-22}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"
LIST_ONLY="${LIST_ONLY:-0}"
ALL_EPOCHS="${ALL_EPOCHS:-0}"

RSYNC_BIN="/usr/bin/rsync"
[[ -x "$RSYNC_BIN" ]] || RSYNC_BIN="$(command -v rsync)"

SSH_OPTS=(-p "${REMOTE_SSH_PORT}" -i $HOME/.ssh/id_ed25519_czhang_skynet -o IdentitiesOnly=yes -o PreferredAuthentications=publickey -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

_ssh() {
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" ssh "${SSH_OPTS[@]}" ${SSH_EXTRA_OPTS} \
      "$REMOTE_USER_HOST" "$@"
  else
    ssh "${SSH_OPTS[@]}" ${SSH_EXTRA_OPTS} "$REMOTE_USER_HOST" "$@"
  fi
}

_rsync() {
  local src="$1" dst="$2"; shift 2
  mkdir -p "$(dirname "$dst")"
  local ssh_cmd="ssh -p ${REMOTE_SSH_PORT} -i $HOME/.ssh/id_ed25519_czhang_skynet -o IdentitiesOnly=yes -o PreferredAuthentications=publickey -o ServerAliveInterval=30 -o ServerAliveCountMax=6 ${SSH_EXTRA_OPTS}"
  if [[ -n "${SKYNET_PASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      sshpass -p "$SKYNET_PASS" \
      "$RSYNC_BIN" -avh --progress --partial --inplace "$@" \
      -e "$ssh_cmd" "${REMOTE_USER_HOST}:${src}" "$dst"
  else
    env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
      "$RSYNC_BIN" -avh --progress --partial --inplace "$@" \
      -e "$ssh_cmd" "${REMOTE_USER_HOST}:${src}" "$dst"
  fi
}

echo "=== 1. Listing remote pcd/adapt3r checkpoints ==="
CKPT_LIST="$(_ssh "cd ${REMOTE_REPO} && find logs -mindepth 3 -maxdepth 5 -name '*.ckpt' \
  \( -ipath '*pcd*' -o -ipath '*adapt3r*' \) 2>/dev/null | sort")"
if [[ -z "$CKPT_LIST" ]]; then
  echo "No remote checkpoints found (or listing failed)." >&2
  exit 1
fi
echo "$CKPT_LIST" | sed 's/^/  remote: /'

# Select per run-dir: highest epoch + last.ckpt (or everything with ALL_EPOCHS=1).
SELECTED="$(echo "$CKPT_LIST" | ALL_EPOCHS="$ALL_EPOCHS" python3 -c '
import os, re, sys
all_epochs = os.environ.get("ALL_EPOCHS") == "1"
best, lasts, other = {}, {}, []
for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    if all_epochs:
        other.append(path)
        continue
    d, f = os.path.dirname(path), os.path.basename(path)
    if f == "last.ckpt":
        lasts[d] = path
        continue
    m = re.fullmatch(r"epoch_epoch=(\d+)\.ckpt", f)
    if m:
        ep = int(m.group(1))
        if ep >= best.get(d, (-1, ""))[0]:
            best[d] = (ep, path)
    else:
        other.append(path)
out = sorted({p for _, p in best.values()} | set(lasts.values()) | set(other))
print("\n".join(out))
')"

echo ""
echo "=== 2. Selected (latest per run) ==="
echo "$SELECTED" | sed 's/^/  pull: /'

if [[ "$LIST_ONLY" == "1" ]]; then
  echo "(LIST_ONLY=1 — nothing pulled)"
  exit 0
fi

echo ""
echo "=== 3. Pulling checkpoints ==="
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  local_path="checkpoints/${rel#logs/}"
  if [[ -f "$local_path" ]]; then
    echo "  SKIP (exists): $local_path"
    continue
  fi
  echo "  --- $rel ---"
  _rsync "${REMOTE_REPO}/${rel}" "./${local_path}"
done <<< "$SELECTED"

echo ""
echo "=== 4. Datasets (pcd2048 for the 2048 eval; others resynced cheaply) ==="
for ds in rby1_teleop_pcd2048_glass rby1_teleop_pcd1024_glass rby1_teleop_slamrect_rgbd; do
  echo "  --- datasets/${ds} ---"
  _rsync "${REMOTE_REPO}/datasets/${ds}/" "./datasets/${ds}/"
done

echo ""
echo "=== Done. Local pcd/depth checkpoints: ==="
find checkpoints/RBY1_dp3_pcd1024 checkpoints/RBY1_dp3_pcd checkpoints/RBY1_adapt3r_slamrect \
  -name '*.ckpt' -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $5 "  " $9}'
du -sh datasets/rby1_teleop_pcd2048_glass 2>/dev/null || true
