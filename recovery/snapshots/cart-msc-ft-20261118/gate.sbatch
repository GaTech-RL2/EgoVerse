#!/bin/bash
set -euo pipefail
source "${CAMPAIGN_ROOT:?}/scripts/env.sh"
exec "$CAMPAIGN_PYTHON" "$CAMPAIGN_ROOT/scripts/gate.py" "${RUN_ID:?}"
