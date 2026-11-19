#!/usr/bin/env bash
# One cell on a cluster without the Pace data copy (ICE): stage the cell's
# episodes from R2 to node-local disk, then score the final checkpoint and
# then every other retained checkpoint (receipted steps are skipped).
#
# Usage: ice_evaluate.sh <cell> <opscale_root>
#   <opscale_root>/private-unseen-clean-20261107  metadata; canonical_zarr -> $STAGE
#   <opscale_root>/index                          _episode_index*.json
#   <opscale_root>/stage_inventory.json           url, md5, size per file
#   <opscale_root>/runs/manifest.json             from `runtime prepare`
# Runs the runtime from this checkout, not the commit `prepare` ran from.
set -euo pipefail
cell=$1
root=$2
STAGE=/tmp/acheluva3-opscale/canonical_zarr

set -a
. ~/.egoverse_env
set +a
# s5cmd reads AWS_* only; map the R2 keys and drop ~/.aws (AWS, not R2).
export AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY
export AWS_REGION=auto AWS_DEFAULT_REGION=auto
export AWS_SHARED_CREDENTIALS_FILE=/dev/null AWS_CONFIG_FILE=/dev/null
unset AWS_SESSION_TOKEN AWS_PROFILE

here=$(dirname "$0")
echo "[ice] $(date) $(hostname) stage start"
df -h /tmp
python -u "$here/stage_episodes.py" "$root/stage_inventory.json" \
  "$root/private-unseen-clean-20261107/splits/fold_moving_$cell.json" "$STAGE" "$root/index"
echo "[ice] $(date) stage done"
cd "$here/../../.."
for mode in evaluate-final evaluate-all; do
  python -u -m experiments.fold_clothes_moving_opscale.lambda_async.runtime \
    "$mode" --manifest "$root/runs/manifest.json" --cell "$cell"
  echo "[ice] $(date) $mode done"
done
echo "[ice] $(date) all done"
