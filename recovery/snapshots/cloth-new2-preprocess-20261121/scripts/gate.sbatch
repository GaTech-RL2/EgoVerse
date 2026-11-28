#!/bin/bash
set -euo pipefail
source /coc/flash7/cxin34/cloth-new2-scratch-20261121/scripts/env.sh
exec "$CAMPAIGN_PYTHON" "$CAMPAIGN_ROOT/scripts/smoke_gate.py" cloth-new2-B-msc-scratch-s42
