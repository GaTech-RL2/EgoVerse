#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# setup_secret.sh — write credential files for EgoVerse
#
# Two modes, writing to DIFFERENT files:
#
#   NEW (direct) mode  — Mecka R2 + DigitalOcean keys → ~/.egoverse_env
#   Used by: training, data processing, Modal training jobs
#   Required env vars:
#     NEW_R2_ACCESS_KEY_ID      Cloudflare R2 access key (new Mecka account)
#     NEW_R2_SECRET_ACCESS_KEY  Cloudflare R2 secret key
#     NEW_R2_ENDPOINT_URL       R2 endpoint URL  (bucket: data)
#     NEW_DATABASE_URL          DigitalOcean PostgreSQL connection string
#     NEW_MONGODB_URI           (optional) MongoDB URI
#
#   OLD (AWS Secrets Manager) mode — legacy EgoVerse rldb keys → ~/.egoverse_env_old
#   Used by: ingest_zarr.py ONLY (download from old rldb R2 bucket into Modal volume)
#   Requires `aws configure` to have been run first:
#     AccessKeyId:     AKIAYDKH4BNCAYHE5NG2
#     SecretAccessKey: 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
#     Default region:  us-east-2
#   The script fetches R2 and DB credentials from AWS Secrets Manager.
#   NOTE: these credentials give access to the legacy rldb R2 bucket and the
#   old AWS RDS database.  Only ingest_zarr.py should use them for training.
# ---------------------------------------------------------------------------

REGION="${REGION:-us-east-2}"
DB_SECRET_NAME="${DB_SECRET_NAME:-rds/appdb/appuser}"
PUBLIC_DB_SECRET_NAME="${PUBLIC_DB_SECRET_NAME:-rds/appdb/appuser-readonly}"
R2_SECRET_NAME="${R2_SECRET_NAME:-r2/rldb/credentials}"
PUBLIC_R2_SECRET_NAME="${PUBLIC_R2_SECRET_NAME:-r2/rldb/public/credentials}"
BUCKET="${BUCKET:-rldb}"

# ---------------------------------------------------------------------------
# NEW (direct-credentials) mode  →  ~/.egoverse_env
# ---------------------------------------------------------------------------
if [[ -n "${NEW_R2_ACCESS_KEY_ID:-}" ]]; then
  ENV_FILE="${ENV_FILE:-$HOME/.egoverse_env}"
  echo "=== New Mecka credentials mode → $ENV_FILE ==="

  _r2_access="${NEW_R2_ACCESS_KEY_ID}"
  _r2_secret="${NEW_R2_SECRET_ACCESS_KEY:?'Set NEW_R2_SECRET_ACCESS_KEY'}"
  _r2_endpoint="${NEW_R2_ENDPOINT_URL:?'Set NEW_R2_ENDPOINT_URL'}"
  _db_url="${NEW_DATABASE_URL:-}"
  _mongo_uri="${NEW_MONGODB_URI:-}"

  {
    printf "R2_ACCESS_KEY_ID=%q\n"     "$_r2_access"
    printf "R2_SECRET_ACCESS_KEY=%q\n" "$_r2_secret"
    printf "AWS_ENDPOINT_URL_S3=%q\n"  "$_r2_endpoint"
    printf "R2_ENDPOINT_URL=%q\n"      "$_r2_endpoint"
    printf "S3_ENDPOINT_URL=%q\n"      "$_r2_endpoint"
    printf "AWS_DEFAULT_REGION=%q\n"   "auto"
    printf "BUCKET=%q\n"               "data"
    if [[ -n "$_db_url" ]]; then
      printf "DATABASE_URL='%s'\n" "$_db_url"
    fi
    if [[ -n "$_mongo_uri" ]]; then
      printf "MONGODB_URI='%s'\n" "$_mongo_uri"
    fi
  } >"$ENV_FILE"

  chmod 600 "$ENV_FILE"
  echo "✅ Wrote new Mecka credentials to $ENV_FILE"
  exit 0
fi

# ---------------------------------------------------------------------------
# OLD (AWS Secrets Manager) mode  →  ~/.egoverse_env_old
# ---------------------------------------------------------------------------
ENV_FILE="${ENV_FILE:-$HOME/.egoverse_env_old}"
echo "=== Old EgoVerse credentials mode → $ENV_FILE ==="
echo "NOTE: these credentials access the legacy rldb R2 bucket and old AWS RDS."
echo "      Used by ingest_zarr.py ONLY — do NOT use for training."
echo ""

SECRET_ARN=""
EFFECTIVE_DB_SECRET_NAME=""
if SECRET_ARN="$(
  aws secretsmanager describe-secret \
    --secret-id "$DB_SECRET_NAME" \
    --region "$REGION" \
    --query 'ARN' \
    --output text 2>/dev/null
)"; then
  EFFECTIVE_DB_SECRET_NAME="$DB_SECRET_NAME"
elif [[ "$PUBLIC_DB_SECRET_NAME" != "$DB_SECRET_NAME" ]] && SECRET_ARN="$(
  aws secretsmanager describe-secret \
    --secret-id "$PUBLIC_DB_SECRET_NAME" \
    --region "$REGION" \
    --query 'ARN' \
    --output text 2>/dev/null
)"; then
  EFFECTIVE_DB_SECRET_NAME="$PUBLIC_DB_SECRET_NAME"
fi

if R2_SECRET_JSON="$(
  aws secretsmanager get-secret-value \
    --secret-id "$R2_SECRET_NAME" \
    --region "$REGION" \
    --query 'SecretString' \
    --output text 2>/dev/null
)"; then
  :
elif [[ "$PUBLIC_R2_SECRET_NAME" != "$R2_SECRET_NAME" ]] && R2_SECRET_JSON="$(
  aws secretsmanager get-secret-value \
    --secret-id "$PUBLIC_R2_SECRET_NAME" \
    --region "$REGION" \
    --query 'SecretString' \
    --output text 2>/dev/null
)"; then
  R2_SECRET_NAME="$PUBLIC_R2_SECRET_NAME"
else
  echo "⚠️  Could not fetch R2 secret from Secrets Manager."
  echo "   R2 credentials will NOT be written to $ENV_FILE."
  echo "   DB secret (SECRETS_ARN) will still be written if available."
  R2_SECRET_JSON=""
fi

CREDENTIAL_MODE="admin"
if [[ "$R2_SECRET_NAME" == "$PUBLIC_R2_SECRET_NAME" ]] || [[ "$EFFECTIVE_DB_SECRET_NAME" == "$PUBLIC_DB_SECRET_NAME" ]]; then
  CREDENTIAL_MODE="public"
fi

if [[ "$CREDENTIAL_MODE" == "public" ]]; then
  echo "Credential level: public (read-only)"
else
  echo "Credential level: admin"
fi
if [[ -n "$EFFECTIVE_DB_SECRET_NAME" ]]; then
  echo "DB secret: $EFFECTIVE_DB_SECRET_NAME"
fi

{
  if [[ -n "$SECRET_ARN" ]]; then
    printf "SECRETS_ARN=%q\n" "$SECRET_ARN"
  fi

  if [[ -n "$R2_SECRET_JSON" ]]; then
    read -r R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_SESSION_TOKEN AWS_ENDPOINT_URL_S3 < <(
      R2_SECRET_JSON="$R2_SECRET_JSON" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["R2_SECRET_JSON"])
access = payload.get("access_key_id", "")
secret = payload.get("secret_access_key", "")
session = payload.get("session_token", "")
endpoint = payload.get("endpoint_url", "")
if not access or not secret or not endpoint:
    print("Missing required keys in R2 secret JSON.", file=sys.stderr)
    sys.exit(1)
print(access, secret, session or "__EMPTY__", endpoint)
PY
    )
    printf "R2_ACCESS_KEY_ID=%q\n"    "$R2_ACCESS_KEY_ID"
    printf "R2_SECRET_ACCESS_KEY=%q\n" "$R2_SECRET_ACCESS_KEY"
    if [[ "$R2_SESSION_TOKEN" != "__EMPTY__" ]]; then
      printf "R2_SESSION_TOKEN=%q\n" "$R2_SESSION_TOKEN"
    fi
    printf "AWS_ENDPOINT_URL_S3=%q\n" "$AWS_ENDPOINT_URL_S3"
    printf "R2_ENDPOINT_URL=%q\n"     "$AWS_ENDPOINT_URL_S3"
    printf "S3_ENDPOINT_URL=%q\n"     "$AWS_ENDPOINT_URL_S3"
  fi

  printf "AWS_DEFAULT_REGION=%q\n" "$REGION"
  printf "BUCKET=%q\n"             "$BUCKET"

  if [[ -n "${DATABASE_URL:-}" ]]; then
    printf "DATABASE_URL='%s'\n" "$DATABASE_URL"
  fi
  if [[ -n "${MONGODB_URI:-}" ]]; then
    printf "MONGODB_URI='%s'\n" "$MONGODB_URI"
  fi
} >"$ENV_FILE"

chmod 600 "$ENV_FILE"
echo "✅ Wrote old EgoVerse credentials to $ENV_FILE"
