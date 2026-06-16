#!/usr/bin/env bash
set -euo pipefail

# Cross-account migration for an unencrypted RDS PostgreSQL instance.
# Source and destination accounts must both be accessible via local AWS CLI profiles.

SRC_PROFILE="${SRC_PROFILE:-default}"
DST_PROFILE="${DST_PROFILE:-dst-556885871428}"
REGION="${REGION:-us-east-2}"
SRC_ACCOUNT="${SRC_ACCOUNT:-654654140494}"
DST_ACCOUNT="${DST_ACCOUNT:-556885871428}"
SRC_DB="${SRC_DB:-lowuse-pg-east2}"
DST_DB="${DST_DB:-lowuse-pg-east2}"
SECRET_NAME="${SECRET_NAME:-rds/appdb/appuser}"
SUBNET_GROUP_NAME="${SUBNET_GROUP_NAME:-rds-subnets-east2}"
SG1_NAME="${SG1_NAME:-rds-pg-allow-myip}"
SG2_NAME="${SG2_NAME:-rds-sg-east2}"
DB_CLASS="${DB_CLASS:-db.t4g.micro}"
BACKUP_RETENTION="${BACKUP_RETENTION:-7}"
SKIP_APP_CHECK="${SKIP_APP_CHECK:-0}"

SNAP_ID="${SNAP_ID:-${SRC_DB}-xacct-$(date +%Y%m%d%H%M%S)}"
COPY_SNAP_ID="${COPY_SNAP_ID:-${SNAP_ID}-dstcopy}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

aws_account_for_profile() {
  aws sts get-caller-identity --profile "$1" --query Account --output text
}

get_or_create_sg() {
  local profile="$1"
  local region="$2"
  local vpc_id="$3"
  local sg_name="$4"
  local sg_id

  sg_id=$(
    aws ec2 describe-security-groups \
      --profile "$profile" \
      --region "$region" \
      --filters "Name=vpc-id,Values=$vpc_id" "Name=group-name,Values=$sg_name" \
      --query 'SecurityGroups[0].GroupId' \
      --output text
  )

  if [[ -z "$sg_id" || "$sg_id" == "None" ]]; then
    sg_id=$(
      aws ec2 create-security-group \
        --profile "$profile" \
        --region "$region" \
        --vpc-id "$vpc_id" \
        --group-name "$sg_name" \
        --description "RDS access group: $sg_name" \
        --query GroupId \
        --output text
    )
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "Created security group $sg_name -> $sg_id" >&2
  else
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "Using existing security group $sg_name -> $sg_id" >&2
  fi

  echo "$sg_id"
}

allow_sg_ingress_cidr_5432() {
  local profile="$1"
  local region="$2"
  local sg_id="$3"
  local cidr="$4"
  local desc="$5"

  aws ec2 authorize-security-group-ingress \
    --profile "$profile" \
    --region "$region" \
    --group-id "$sg_id" \
    --ip-permissions "IpProtocol=tcp,FromPort=5432,ToPort=5432,IpRanges=[{CidrIp=$cidr,Description=$desc}]" \
    >/dev/null 2>&1 || true
}

secret_field_from_string() {
  local secret_json="$1"
  local key="$2"
  python3 - "$secret_json" "$key" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
key = sys.argv[2]
val = data.get(key, "")
if val is None:
    val = ""
print(val)
PY
}

build_updated_secret_string() {
  local secret_json="$1"
  local host="$2"
  python3 - "$secret_json" "$host" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
host = sys.argv[2]
data["host"] = host
data["HOST"] = host
print(json.dumps(data, separators=(",", ":")))
PY
}

psql_query_from_secret_json() {
  local secret_json="$1"
  local sql="$2"

  local host dbname user password port
  host="$(secret_field_from_string "$secret_json" "host")"
  [[ -n "$host" ]] || host="$(secret_field_from_string "$secret_json" "HOST")"

  dbname="$(secret_field_from_string "$secret_json" "dbname")"
  [[ -n "$dbname" ]] || dbname="$(secret_field_from_string "$secret_json" "DBNAME")"
  [[ -n "$dbname" ]] || dbname="appdb"

  user="$(secret_field_from_string "$secret_json" "username")"
  [[ -n "$user" ]] || user="$(secret_field_from_string "$secret_json" "user")"
  [[ -n "$user" ]] || user="$(secret_field_from_string "$secret_json" "USER")"

  password="$(secret_field_from_string "$secret_json" "password")"
  [[ -n "$password" ]] || password="$(secret_field_from_string "$secret_json" "PASSWORD")"

  port="$(secret_field_from_string "$secret_json" "port")"
  [[ -n "$port" ]] || port="5432"

  PGPASSWORD="$password" psql \
    "host=$host port=$port user=$user dbname=$dbname sslmode=require" \
    -v ON_ERROR_STOP=1 -tAc "$sql"
}

log "Starting cross-account RDS migration"
log "Source: profile=$SRC_PROFILE account=$SRC_ACCOUNT db=$SRC_DB region=$REGION"
log "Dest:   profile=$DST_PROFILE account=$DST_ACCOUNT db=$DST_DB region=$REGION"
log "Snapshot IDs: source=$SNAP_ID destcopy=$COPY_SNAP_ID"

require_cmd aws
require_cmd python3
require_cmd psql
require_cmd curl

log "Verifying AWS profiles and account IDs"
actual_src_account="$(aws_account_for_profile "$SRC_PROFILE")"
actual_dst_account="$(aws_account_for_profile "$DST_PROFILE")"

if [[ "$actual_src_account" != "$SRC_ACCOUNT" ]]; then
  echo "Source profile account mismatch: expected $SRC_ACCOUNT, got $actual_src_account" >&2
  exit 1
fi
if [[ "$actual_dst_account" != "$DST_ACCOUNT" ]]; then
  echo "Destination profile account mismatch: expected $DST_ACCOUNT, got $actual_dst_account" >&2
  exit 1
fi

log "Checking destination DB collision"
dst_db_exists_count=$(
  aws rds describe-db-instances \
    --profile "$DST_PROFILE" \
    --region "$REGION" \
    --query "DBInstances[?DBInstanceIdentifier=='$DST_DB'] | length(@)" \
    --output text
)
if [[ "$dst_db_exists_count" != "0" ]]; then
  echo "Destination DB instance '$DST_DB' already exists. Aborting." >&2
  exit 1
fi

log "Preparing destination VPC resources"
dst_vpc_id=$(
  aws ec2 describe-vpcs \
    --profile "$DST_PROFILE" \
    --region "$REGION" \
    --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' \
    --output text
)
if [[ -z "$dst_vpc_id" || "$dst_vpc_id" == "None" ]]; then
  echo "No default VPC found in destination account/region $REGION" >&2
  exit 1
fi
log "Destination default VPC: $dst_vpc_id"

read -r -a dst_subnets <<<"$(
  aws ec2 describe-subnets \
    --profile "$DST_PROFILE" \
    --region "$REGION" \
    --filters "Name=vpc-id,Values=$dst_vpc_id" "Name=default-for-az,Values=true" \
    --query 'Subnets[].SubnetId' \
    --output text
)"
if (( ${#dst_subnets[@]} < 2 )); then
  echo "Need at least 2 default subnets in destination VPC for subnet group" >&2
  exit 1
fi

if ! aws rds describe-db-subnet-groups \
  --profile "$DST_PROFILE" \
  --region "$REGION" \
  --db-subnet-group-name "$SUBNET_GROUP_NAME" >/dev/null 2>&1; then
  aws rds create-db-subnet-group \
    --profile "$DST_PROFILE" \
    --region "$REGION" \
    --db-subnet-group-name "$SUBNET_GROUP_NAME" \
    --db-subnet-group-description "RDS subnets in $REGION" \
    --subnet-ids "${dst_subnets[@]}" >/dev/null
  log "Created DB subnet group: $SUBNET_GROUP_NAME"
else
  log "Using existing DB subnet group: $SUBNET_GROUP_NAME"
fi

DST_SG1="$(get_or_create_sg "$DST_PROFILE" "$REGION" "$dst_vpc_id" "$SG1_NAME")"
DST_SG2="$(get_or_create_sg "$DST_PROFILE" "$REGION" "$dst_vpc_id" "$SG2_NAME")"

allow_sg_ingress_cidr_5432 "$DST_PROFILE" "$REGION" "$DST_SG1" "0.0.0.0/0" "PostgreSQL-access-from-all-IPs"
allow_sg_ingress_cidr_5432 "$DST_PROFILE" "$REGION" "$DST_SG2" "0.0.0.0/0" "PostgreSQL-access-from-all-IPs"

my_ip="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')"
if [[ -n "$my_ip" ]]; then
  allow_sg_ingress_cidr_5432 "$DST_PROFILE" "$REGION" "$DST_SG1" "${my_ip}/32" "psql-from-home"
  log "Added/kept ${my_ip}/32 ingress on $DST_SG1"
fi

log "Creating source manual snapshot: $SNAP_ID"
aws rds create-db-snapshot \
  --profile "$SRC_PROFILE" \
  --region "$REGION" \
  --db-instance-identifier "$SRC_DB" \
  --db-snapshot-identifier "$SNAP_ID" >/dev/null

log "Waiting for source snapshot availability"
aws rds wait db-snapshot-available \
  --profile "$SRC_PROFILE" \
  --region "$REGION" \
  --db-snapshot-identifier "$SNAP_ID"

log "Sharing source snapshot with destination account"
aws rds modify-db-snapshot-attribute \
  --profile "$SRC_PROFILE" \
  --region "$REGION" \
  --db-snapshot-identifier "$SNAP_ID" \
  --attribute-name restore \
  --values-to-add "$DST_ACCOUNT" >/dev/null

SRC_SNAP_ARN=$(
  aws rds describe-db-snapshots \
    --profile "$SRC_PROFILE" \
    --region "$REGION" \
    --db-snapshot-identifier "$SNAP_ID" \
    --query 'DBSnapshots[0].DBSnapshotArn' \
    --output text
)
log "Copying shared snapshot into destination account: $COPY_SNAP_ID"
aws rds copy-db-snapshot \
  --profile "$DST_PROFILE" \
  --region "$REGION" \
  --source-db-snapshot-identifier "$SRC_SNAP_ARN" \
  --target-db-snapshot-identifier "$COPY_SNAP_ID" >/dev/null

log "Waiting for destination snapshot copy availability"
aws rds wait db-snapshot-available \
  --profile "$DST_PROFILE" \
  --region "$REGION" \
  --db-snapshot-identifier "$COPY_SNAP_ID"

log "Restoring destination DB instance from copied snapshot"
aws rds restore-db-instance-from-db-snapshot \
  --profile "$DST_PROFILE" \
  --region "$REGION" \
  --db-instance-identifier "$DST_DB" \
  --db-snapshot-identifier "$COPY_SNAP_ID" \
  --db-instance-class "$DB_CLASS" \
  --publicly-accessible \
  --db-subnet-group-name "$SUBNET_GROUP_NAME" \
  --vpc-security-group-ids "$DST_SG1" "$DST_SG2" \
  --no-multi-az >/dev/null

log "Waiting for destination DB instance availability"
aws rds wait db-instance-available \
  --profile "$DST_PROFILE" \
  --region "$REGION" \
  --db-instance-identifier "$DST_DB"

log "Setting destination backup retention to $BACKUP_RETENTION days"
aws rds modify-db-instance \
  --profile "$DST_PROFILE" \
  --region "$REGION" \
  --db-instance-identifier "$DST_DB" \
  --backup-retention-period "$BACKUP_RETENTION" \
  --apply-immediately >/dev/null

log "Replicating secret payload to destination account"
DST_HOST=$(
  aws rds describe-db-instances \
    --profile "$DST_PROFILE" \
    --region "$REGION" \
    --db-instance-identifier "$DST_DB" \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text
)

SRC_SECRET_JSON=$(
  aws secretsmanager get-secret-value \
    --profile "$SRC_PROFILE" \
    --region "$REGION" \
    --secret-id "$SECRET_NAME" \
    --query SecretString \
    --output text
)

NEW_SECRET_JSON="$(build_updated_secret_string "$SRC_SECRET_JSON" "$DST_HOST")"

if aws secretsmanager describe-secret \
  --profile "$DST_PROFILE" \
  --region "$REGION" \
  --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
  aws secretsmanager put-secret-value \
    --profile "$DST_PROFILE" \
    --region "$REGION" \
    --secret-id "$SECRET_NAME" \
    --secret-string "$NEW_SECRET_JSON" >/dev/null
  log "Updated existing destination secret: $SECRET_NAME"
else
  aws secretsmanager create-secret \
    --profile "$DST_PROFILE" \
    --region "$REGION" \
    --name "$SECRET_NAME" \
    --description "RDS credentials for appdb" \
    --secret-string "$NEW_SECRET_JSON" >/dev/null
  log "Created destination secret: $SECRET_NAME"
fi

log "Running validation checks"
src_count="$(psql_query_from_secret_json "$SRC_SECRET_JSON" "SELECT count(*) FROM app.episodes;")"
dst_count="$(psql_query_from_secret_json "$NEW_SECRET_JSON" "SELECT count(*) FROM app.episodes;")"
src_count="$(echo "$src_count" | tr -d '[:space:]')"
dst_count="$(echo "$dst_count" | tr -d '[:space:]')"

if [[ "$src_count" != "$dst_count" ]]; then
  echo "Row count mismatch: source=$src_count destination=$dst_count" >&2
  exit 1
fi
log "Row count check passed: $src_count rows"

schema_priv="$(psql_query_from_secret_json "$NEW_SECRET_JSON" "SELECT has_schema_privilege(current_user,'app','USAGE,CREATE');" | tr -d '[:space:]')"
table_priv="$(psql_query_from_secret_json "$NEW_SECRET_JSON" "SELECT has_table_privilege(current_user,'app.episodes','SELECT,INSERT,UPDATE,DELETE');" | tr -d '[:space:]')"
if [[ "$schema_priv" != "t" || "$table_priv" != "t" ]]; then
  echo "Privilege check failed: schema=$schema_priv table=$table_priv" >&2
  exit 1
fi
log "Privilege checks passed"

DST_SECRET_ARN=$(
  aws secretsmanager describe-secret \
    --profile "$DST_PROFILE" \
    --region "$REGION" \
    --secret-id "$SECRET_NAME" \
    --query ARN \
    --output text
)
log "Destination secret ARN: $DST_SECRET_ARN"
log "Destination DB host: $DST_HOST"

if [[ "$SKIP_APP_CHECK" == "1" ]]; then
  log "Skipping app import/engine check (SKIP_APP_CHECK=1)"
else
  log "Running app-level engine check using destination secret"
  SECRETS_ARN="$DST_SECRET_ARN" python3 - <<'PY'
from egomimic.utils.aws.aws_sql import create_default_engine

engine = create_default_engine()
with engine.connect() as conn:
    conn.exec_driver_sql("SELECT 1")
print("ok")
PY
  log "App-level engine check passed"
fi

log "Migration completed successfully."
log "Next cutover action: point workloads to destination account credentials and SECRETS_ARN=$DST_SECRET_ARN"
