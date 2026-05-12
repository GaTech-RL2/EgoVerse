#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# setup_secret.sh — interactive credential setup for EgoVerse
#
# Writes TWO files in one run:
#   ~/.egoverse_env      new Mecka R2 + WandB  (training, Modal jobs)
#   ~/.egoverse_env_old  legacy rldb R2 + AWS RDS  (ingest_zarr.py ONLY)
#
# Non-interactive / CI mode: pre-set the env vars below and run with
#   NONINTERACTIVE=1 ./setup_secret.sh
# ---------------------------------------------------------------------------

NEW_ENV_FILE="${NEW_ENV_FILE:-$HOME/.egoverse_env}"
OLD_ENV_FILE="${OLD_ENV_FILE:-$HOME/.egoverse_env_old}"
REGION="${REGION:-us-east-2}"
DB_SECRET_NAME="${DB_SECRET_NAME:-rds/appdb/appuser}"
PUBLIC_DB_SECRET_NAME="${PUBLIC_DB_SECRET_NAME:-rds/appdb/appuser-readonly}"
R2_SECRET_NAME="${R2_SECRET_NAME:-r2/rldb/credentials}"
PUBLIC_R2_SECRET_NAME="${PUBLIC_R2_SECRET_NAME:-r2/rldb/public/credentials}"

# Detect whether we're running interactively
INTERACTIVE=1
if [[ "${NONINTERACTIVE:-0}" == "1" ]] || [[ ! -t 0 ]]; then
  INTERACTIVE=0
fi

# ---------------------------------------------------------------------------
# Helper: prompt for a value, with optional silent input and default
# ---------------------------------------------------------------------------
prompt() {
  local var_name="$1"
  local label="$2"
  local silent="${3:-0}"
  local default="${4:-}"
  local current="${!var_name:-}"

  if [[ -n "$current" ]]; then
    echo "  $label: (using env var)"
    return
  fi

  if [[ "$INTERACTIVE" == "0" ]]; then
    if [[ -z "$default" ]]; then
      echo "ERROR: $var_name must be set in non-interactive mode." >&2
      exit 1
    fi
    printf -v "$var_name" "%s" "$default"
    return
  fi

  local prompt_str="  $label"
  [[ -n "$default" ]] && prompt_str+=" [${default}]"
  prompt_str+=": "

  if [[ "$silent" == "1" ]]; then
    read -rsp "$prompt_str" input
    echo
  else
    read -rp "$prompt_str" input
  fi

  if [[ -z "$input" && -n "$default" ]]; then
    input="$default"
  fi
  printf -v "$var_name" "%s" "$input"
}

# ---------------------------------------------------------------------------
# PART 1 — New Mecka credentials  →  ~/.egoverse_env
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PART 1 — New Mecka R2 + WandB  →  $NEW_ENV_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Used by: training, Modal jobs, data processing"
echo ""

prompt NEW_R2_ACCESS_KEY_ID     "New Mecka R2 Access Key ID"
prompt NEW_R2_SECRET_ACCESS_KEY "New Mecka R2 Secret Access Key" 1
prompt NEW_R2_ENDPOINT_URL      "New Mecka R2 Endpoint URL"
prompt NEW_MONGODB_URI          "MongoDB URI (optional, press Enter to skip)"
prompt WANDB_API_KEY            "WandB API Key (from wandb.ai/settings)" 1

{
  printf "R2_ACCESS_KEY_ID=%q\n"     "$NEW_R2_ACCESS_KEY_ID"
  printf "R2_SECRET_ACCESS_KEY=%q\n" "$NEW_R2_SECRET_ACCESS_KEY"
  printf "AWS_ENDPOINT_URL_S3=%q\n"  "$NEW_R2_ENDPOINT_URL"
  printf "R2_ENDPOINT_URL=%q\n"      "$NEW_R2_ENDPOINT_URL"
  printf "S3_ENDPOINT_URL=%q\n"      "$NEW_R2_ENDPOINT_URL"
  printf "AWS_DEFAULT_REGION=%q\n"   "auto"
  printf "BUCKET=%q\n"               "data"
  if [[ -n "$NEW_MONGODB_URI" ]]; then
    printf "MONGODB_URI='%s'\n" "$NEW_MONGODB_URI"
  fi
  if [[ -n "$WANDB_API_KEY" ]]; then
    printf "WANDB_API_KEY=%q\n" "$WANDB_API_KEY"
  fi
} >"$NEW_ENV_FILE"

chmod 600 "$NEW_ENV_FILE"
echo ""
echo "✅ Wrote new Mecka credentials to $NEW_ENV_FILE"

# ---------------------------------------------------------------------------
# PART 2 — Legacy rldb credentials via AWS Secrets Manager  →  ~/.egoverse_env_old
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PART 2 — Legacy rldb R2 + AWS RDS  →  $OLD_ENV_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Used by: ingest_zarr.py ONLY"
echo "  Requires AWS CLI configured with EgoVerse account credentials."
echo ""

# Check if AWS CLI is available and configured
if ! command -v aws &>/dev/null; then
  echo "⚠️  aws CLI not found — skipping legacy credential setup."
  echo "   Install it and run \`aws configure\` then re-run this script."
  exit 0
fi

if ! aws sts get-caller-identity --region "$REGION" &>/dev/null; then
  echo "  AWS CLI is not configured. Enter the legacy EgoVerse AWS credentials:"
  echo ""
  prompt AWS_LEGACY_ACCESS_KEY_ID     "AWS Access Key ID"
  prompt AWS_LEGACY_SECRET_ACCESS_KEY "AWS Secret Access Key" 1
  aws configure set aws_access_key_id     "$AWS_LEGACY_ACCESS_KEY_ID"
  aws configure set aws_secret_access_key "$AWS_LEGACY_SECRET_ACCESS_KEY"
  aws configure set default.region        "$REGION"
  echo ""
  echo "  AWS CLI configured."
fi

# Fetch DB secret ARN
SECRET_ARN=""
EFFECTIVE_DB_SECRET_NAME=""
if SECRET_ARN="$(aws secretsmanager describe-secret \
    --secret-id "$DB_SECRET_NAME" --region "$REGION" \
    --query 'ARN' --output text 2>/dev/null)"; then
  EFFECTIVE_DB_SECRET_NAME="$DB_SECRET_NAME"
elif SECRET_ARN="$(aws secretsmanager describe-secret \
    --secret-id "$PUBLIC_DB_SECRET_NAME" --region "$REGION" \
    --query 'ARN' --output text 2>/dev/null)"; then
  EFFECTIVE_DB_SECRET_NAME="$PUBLIC_DB_SECRET_NAME"
fi

# Fetch R2 secret
R2_SECRET_JSON=""
if R2_SECRET_JSON="$(aws secretsmanager get-secret-value \
    --secret-id "$R2_SECRET_NAME" --region "$REGION" \
    --query 'SecretString' --output text 2>/dev/null)"; then
  :
elif R2_SECRET_JSON="$(aws secretsmanager get-secret-value \
    --secret-id "$PUBLIC_R2_SECRET_NAME" --region "$REGION" \
    --query 'SecretString' --output text 2>/dev/null)"; then
  R2_SECRET_NAME="$PUBLIC_R2_SECRET_NAME"
else
  echo "⚠️  Could not fetch R2 secret from Secrets Manager — R2 keys will be omitted."
  R2_SECRET_JSON=""
fi

{
  [[ -n "$SECRET_ARN" ]] && printf "SECRETS_ARN=%q\n" "$SECRET_ARN"

  if [[ -n "$R2_SECRET_JSON" ]]; then
    read -r R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_SESSION_TOKEN AWS_ENDPOINT_URL_S3 < <(
      R2_SECRET_JSON="$R2_SECRET_JSON" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["R2_SECRET_JSON"])
access   = payload.get("access_key_id", "")
secret   = payload.get("secret_access_key", "")
session  = payload.get("session_token", "")
endpoint = payload.get("endpoint_url", "")
if not access or not secret or not endpoint:
    print("Missing required keys in R2 secret JSON.", file=sys.stderr)
    sys.exit(1)
print(access, secret, session or "__EMPTY__", endpoint)
PY
    )
    printf "R2_ACCESS_KEY_ID=%q\n"     "$R2_ACCESS_KEY_ID"
    printf "R2_SECRET_ACCESS_KEY=%q\n" "$R2_SECRET_ACCESS_KEY"
    [[ "$R2_SESSION_TOKEN" != "__EMPTY__" ]] && printf "R2_SESSION_TOKEN=%q\n" "$R2_SESSION_TOKEN"
    printf "AWS_ENDPOINT_URL_S3=%q\n"  "$AWS_ENDPOINT_URL_S3"
    printf "R2_ENDPOINT_URL=%q\n"      "$AWS_ENDPOINT_URL_S3"
    printf "S3_ENDPOINT_URL=%q\n"      "$AWS_ENDPOINT_URL_S3"
  fi

  printf "AWS_DEFAULT_REGION=%q\n" "$REGION"
  printf "BUCKET=%q\n"             "rldb"
} >"$OLD_ENV_FILE"

chmod 600 "$OLD_ENV_FILE"
echo "✅ Wrote legacy credentials to $OLD_ENV_FILE"
echo ""
echo "Done. Both credential files are configured."
