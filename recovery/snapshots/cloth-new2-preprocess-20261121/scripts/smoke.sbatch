#!/bin/bash
set -euo pipefail
source /coc/flash7/cxin34/cloth-new2-scratch-20261121/scripts/env.sh
"$CAMPAIGN_PYTHON" "$CAMPAIGN_ROOT/scripts/photometric_audit.py"
export SMOKE=1 RUN_ID=cloth-new2-B-msc-scratch-s42
exec bash "$CAMPAIGN_ROOT/scripts/train.sbatch"
