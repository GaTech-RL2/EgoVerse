#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: create_s3_user.sh --user <iam_user> [--bucket <bucket>] [--prefix <prefix>] [--profile <aws_profile>] [--allow-delete]

Creates/updates an IAM user with S3 read/write access to a bucket prefix,
creates an access key, and writes credentials to:
  egomimic/utils/aws/created_users/<iam_user>.csv

Defaults:
  --bucket rldb
  --prefix rldb/   (folder inside bucket; override if needed)
  --allow-delete   (include s3:DeleteObject)

Examples:
  ./egomimic/utils/aws/create_s3_user.sh --user alice
  ./egomimic/utils/aws/create_s3_user.sh --user bob --prefix processed_v2/eva/
  ./egomimic/utils/aws/create_s3_user.sh --user carol --bucket rldb --prefix ""
  ./egomimic/utils/aws/create_s3_user.sh --user dave --allow-delete
USAGE
}

bucket="rldb"
prefix="rldb/"
profile=""
user=""
allow_delete="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user|-u)
      user="$2"
      shift 2
      ;;
    --bucket|-b)
      bucket="$2"
      shift 2
      ;;
    --prefix|-p)
      prefix="$2"
      shift 2
      ;;
    --profile)
      profile="$2"
      shift 2
      ;;
    --allow-delete)
      allow_delete="true"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$user" ]]; then
  read -r -p "IAM user name: " user
fi

if [[ -z "$user" ]]; then
  echo "User name is required." >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found in PATH." >&2
  exit 1
fi

aws_cli=(aws)
if [[ -n "$profile" ]]; then
  aws_cli+=(--profile "$profile")
fi

# Normalize prefix: allow empty (full bucket), otherwise ensure trailing '/'
if [[ -n "$prefix" && "${prefix: -1}" != "/" ]]; then
  prefix="${prefix}/"
fi

# Create IAM user if it doesn't exist
if ! "${aws_cli[@]}" iam get-user --user-name "$user" >/dev/null 2>&1; then
  "${aws_cli[@]}" iam create-user --user-name "$user" >/dev/null
  echo "Created IAM user: $user"
else
  echo "IAM user already exists: $user"
fi

# Build policy document
policy_file="$(mktemp)"
delete_action=""
if [[ "$allow_delete" == "true" ]]; then
  delete_action='        "s3:DeleteObject",'
fi

if [[ -n "$prefix" ]]; then
  object_arn="arn:aws:s3:::${bucket}/${prefix}*"
  cat >"$policy_file" <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::${bucket}",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["${prefix}", "${prefix}*"]
        }
      }
    },
    {
      "Sid": "ObjectReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
${delete_action}
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": "${object_arn}"
    }
  ]
}
POLICY
else
  object_arn="arn:aws:s3:::${bucket}/*"
  cat >"$policy_file" <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::${bucket}"
    },
    {
      "Sid": "ObjectReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
${delete_action}
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": "${object_arn}"
    }
  ]
}
POLICY
fi

policy_name="s3-${bucket}-rw"
"${aws_cli[@]}" iam put-user-policy \
  --user-name "$user" \
  --policy-name "$policy_name" \
  --policy-document "file://${policy_file}"

rm -f "$policy_file"

# Ensure we can create a new access key
key_count=$("${aws_cli[@]}" iam list-access-keys --user-name "$user" --query 'length(AccessKeyMetadata)' --output text)
if [[ "$key_count" -ge 2 ]]; then
  echo "User already has 2 access keys. Delete one before creating a new key." >&2
  exit 1
fi

read -r access_key secret_key <<<"$("${aws_cli[@]}" iam create-access-key --user-name "$user" --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text)"

if [[ -z "$access_key" || -z "$secret_key" ]]; then
  echo "Failed to create access key." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${script_dir}/created_users"
mkdir -p "$output_dir"

output_file="${output_dir}/${user}.csv"
umask 077
{
  echo "User name,Access key ID,Secret access key"
  echo "${user},${access_key},${secret_key}"
} >"$output_file"

chmod 600 "$output_file"

echo "Wrote credentials to: $output_file"
echo "Access key ID: $access_key"
