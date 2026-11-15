#!/usr/bin/env bash
# Forced command for the dedicated uploader key. No shell or forwarding access.
set -euo pipefail
cd "$HOME/leact-tracker"
source emimic/bin/activate
case "${SSH_ORIGINAL_COMMAND:-}" in
  leact-import-snapshot)
    exec python current/server.py import --db data/experiments.sqlite3 --snapshot - ;;
  leact-import-records)
    exec python current/server.py import-records --db data/experiments.sqlite3 --snapshot - ;;
  leact-health)
    exec curl --fail --silent --max-time 5 http://127.0.0.1:8126/api/health ;;
  *)
    echo 'This key accepts only LeAct tracker imports and health checks.' >&2
    exit 64 ;;
esac
