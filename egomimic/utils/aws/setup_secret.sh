#!/bin/bash
set -e

# Configuration
REGION="us-east-2"
SECRET_NAME="rds/appdb/appuser"
HOST="lowuse-pg-east2.claua8sacyu5.us-east-2.rds.amazonaws.com"
DBNAME="appdb"
USER="appuser"
PASSWORD="APPUSER_STRONG_PW"
PORT=5432

echo "=== Setting up Secrets Manager Secret for RDS ==="

# Check if secret exists
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$REGION" 2>/dev/null; then
  echo "Secret already exists: $SECRET_NAME"
  echo "Updating secret..."
  
  # Create the secret JSON
  SECRET_JSON=$(cat <<EOF
{
  "host": "$HOST",
  "port": $PORT,
  "dbname": "$DBNAME",
  "username": "$USER",
  "password": "$PASSWORD"
}
EOF
)
  
  aws secretsmanager update-secret \
    --secret-id "$SECRET_NAME" \
    --secret-string "$SECRET_JSON" \
    --region "$REGION" \
    > /dev/null
  
  echo "Secret updated successfully"
else
  echo "Creating new secret: $SECRET_NAME"
  
  # Create the secret JSON
  SECRET_JSON=$(cat <<EOF
{
  "host": "$HOST",
  "port": $PORT,
  "dbname": "$DBNAME",
  "username": "$USER",
  "password": "$PASSWORD"
}
EOF
)
  
  aws secretsmanager create-secret \
    --name "$SECRET_NAME" \
    --description "RDS credentials for appdb" \
    --secret-string "$SECRET_JSON" \
    --region "$REGION" \
    > /dev/null
  
  echo "Secret created successfully"
fi

# Get the ARN
SECRET_ARN=$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$REGION" --query 'ARN' --output text)
echo ""
echo "Secret ARN: $SECRET_ARN"
echo ""
echo "You can now use this ARN in the Lambda environment variable SECRETS_ARN"

