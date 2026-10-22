#!/bin/bash
# Copy the committed code (no .git, no local edits) to SkyNet as one revision and make it
# the one the watcher runs. Run on a laptop or workstation, from the repo:
#
#     egomimic/ingest/skynet/deploy.sh [host] [INGEST_ROOT]
#
# Defaults: host sky1, INGEST_ROOT /coc/flash7/scratch/mcap_pipeline. The next watcher pass
# picks it up; a pass already running finishes on the old revision.
set -euo pipefail

HOST="${1:-sky1}"
ROOT="${2:-/coc/flash7/scratch/mcap_pipeline}"
cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain -- egomimic)" ]; then
    echo "egomimic/ has uncommitted changes: commit them, so the deployed code is a revision" >&2
    exit 1
fi
REV="$(git rev-parse HEAD)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git archive HEAD egomimic pyproject.toml uv.lock | tar -x -C "$TMP"
echo "$REV" > "$TMP/egomimic/ingest/REVISION"   # stage.py pins it: there is no .git

ssh "$HOST" "mkdir -p '$ROOT/code/$REV' '$ROOT/logs'"
rsync -a --delete "$TMP/" "$HOST:$ROOT/code/$REV/"
ssh "$HOST" "ln -sfn '$REV' '$ROOT/code/current'"
echo "deployed $REV to $HOST:$ROOT/code/current"
