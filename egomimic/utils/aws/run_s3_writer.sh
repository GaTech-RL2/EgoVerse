#!/usr/bin/env bash
set -euo pipefail

# Path to your environment file written by setup_rds_secret.sh
ENV_FILE="/home/ubuntu/.egoverse_env"
PY_SCRIPT="/home/ubuntu/EgoVerse/egomimic/utils/aws/add_raw_data_to_table.py"
LOG_FILE="/home/ubuntu/add_raw_data.log"
RAW_V2_PATH="s3://rldb/raw_v2"
PYTHON_BIN="/usr/bin/python3"

# Load environment variables if the file exists
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
else
  echo "[$(date)] ERROR: Env file $ENV_FILE not found!" > "$LOG_FILE"
  exit 1
fi

RAW_V2_ENDPOINT_URL="${R2_ENDPOINT_URL:-${S3_ENDPOINT_URL:-${AWS_ENDPOINT_URL_S3:-}}}"

# Backward-compat for old env files that stored R2 keys in AWS_* vars.
if [ -n "${RAW_V2_ENDPOINT_URL:-}" ]; then
  if [ -z "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
    export R2_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
  fi
  if [ -z "${R2_SECRET_ACCESS_KEY:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    export R2_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
  fi
fi

# Sanity check — optional
echo "[$(date)] Running add_raw_data_to_table.py with SECRETS_ARN=$SECRETS_ARN RAW_V2_PATH=$RAW_V2_PATH ENDPOINT_URL=${RAW_V2_ENDPOINT_URL:-<none>}" > "$LOG_FILE"

# Run the Python job
if [ -n "$RAW_V2_ENDPOINT_URL" ]; then
  "$PYTHON_BIN" "$PY_SCRIPT" "$RAW_V2_PATH" --endpoint-url "$RAW_V2_ENDPOINT_URL" >> "$LOG_FILE" 2>&1
else
  "$PYTHON_BIN" "$PY_SCRIPT" "$RAW_V2_PATH" >> "$LOG_FILE" 2>&1
fi
