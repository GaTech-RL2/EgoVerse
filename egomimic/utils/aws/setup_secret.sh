#!/usr/bin/env bash
set -euo pipefail

# Configuration (override with environment variables as needed)
REGION="${REGION:-us-east-2}"
DB_SECRET_NAME="${DB_SECRET_NAME:-rds/appdb/appuser}"
R2_SECRET_NAME="${R2_SECRET_NAME:-r2/rldb/credentials}"
ENV_FILE="${ENV_FILE:-$HOME/.egoverse_env}"
BUCKET="${BUCKET:-rldb}"

echo "=== Bootstrapping EgoVerse env from Secrets Manager (read-only) ==="
echo "Region: $REGION"
echo "DB secret: $DB_SECRET_NAME"
echo "R2 secret: $R2_SECRET_NAME"

SECRET_ARN="$(
  aws secretsmanager describe-secret \
    --secret-id "$DB_SECRET_NAME" \
    --region "$REGION" \
    --query 'ARN' \
    --output text
)"

R2_SECRET_JSON="$(
  aws secretsmanager get-secret-value \
    --secret-id "$R2_SECRET_NAME" \
    --region "$REGION" \
    --query 'SecretString' \
    --output text
)"

read -r R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY AWS_ENDPOINT_URL_S3 < <(
  R2_SECRET_JSON="$R2_SECRET_JSON" python3 - <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["R2_SECRET_JSON"])
access = payload.get("access_key_id", "")
secret = payload.get("secret_access_key", "")
endpoint = payload.get("endpoint_url", "")

if not access or not secret or not endpoint:
    print("Missing required keys in R2 secret JSON.", file=sys.stderr)
    sys.exit(1)

print(access, secret, endpoint)
PY
)

{
  printf "SECRETS_ARN=%q\n" "$SECRET_ARN"
  printf "R2_ACCESS_KEY_ID=%q\n" "$R2_ACCESS_KEY_ID"
  printf "R2_SECRET_ACCESS_KEY=%q\n" "$R2_SECRET_ACCESS_KEY"
  printf "AWS_ENDPOINT_URL_S3=%q\n" "$AWS_ENDPOINT_URL_S3"
  printf "R2_ENDPOINT_URL=%q\n" "$AWS_ENDPOINT_URL_S3"
  printf "S3_ENDPOINT_URL=%q\n" "$AWS_ENDPOINT_URL_S3"
  printf "AWS_DEFAULT_REGION=%q\n" "$REGION"
  printf "BUCKET=%q\n" "$BUCKET"
} >"$ENV_FILE"

chmod 600 "$ENV_FILE"
echo "✅ Wrote runtime environment to $ENV_FILE"
