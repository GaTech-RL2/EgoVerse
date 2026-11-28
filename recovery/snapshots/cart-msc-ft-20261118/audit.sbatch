#!/bin/bash
set -euo pipefail
source "${CAMPAIGN_ROOT:?}/scripts/env.sh"
exec "$CAMPAIGN_PYTHON" "$CAMPAIGN_ROOT/scripts/audit.py"
