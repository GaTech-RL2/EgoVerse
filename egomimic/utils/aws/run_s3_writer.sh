#!/usr/bin/env bash
set -euo pipefail

# Path to your environment file written by setup_rds_secret.sh
ENV_FILE="/home/ubuntu/.egoverse_env"
PY_SCRIPT="/home/ubuntu/EgoVerse/egomimic/utils/aws/add_raw_data_to_table.py"
LOG_FILE="/home/ubuntu/add_raw_data.log"

# Load environment variables if the file exists
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
else
  echo "[$(date)] ERROR: Env file $ENV_FILE not found!" >> "$LOG_FILE"
  exit 1
fi

# Sanity check — optional
echo "[$(date)] Running add_raw_data_to_table.py with SECRETS_ARN=$SECRETS_ARN" >> "$LOG_FILE"

# Run the Python job
/usr/bin/python3 "$PY_SCRIPT" >> "$LOG_FILE" 2>&1
