#!/bin/bash
set -euo pipefail
source /coc/flash7/cxin34/human-ondevice-pilot-20261110/scripts/env.sh
E="$PREP_ROOT/episodes/$1"
export ARIA_HANDS=ondevice
export ONDEVICE_AUDIT="$E/ondevice-stream-audit.json"
unset MPS_ROOT
"$ARIA_PY" "$PREP_ROOT/scripts/export_ondevice.py" "$1" >"$E/logs/stream-v2.log" 2>&1
cd "$SEW_REPO"
bash projects/xhand_teleop/aria_egoposer/vrs_to_all.sh "$E/input/episode" "$E/retarget" --njobs 1 --name "$1" >"$E/logs/retarget.log" 2>&1
test -s "$E/retarget/episode.hdf5"
/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python "$PREP_ROOT/scripts/finish.py" "$1"
