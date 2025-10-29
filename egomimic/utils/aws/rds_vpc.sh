# --- set your region and DB identifier ---
REGION=us-east-2
DBID=lowuse-pg-east2

# 1) Inspect the RDS instance: VPC, subnets, and attached SGs
aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DBID" \
  --query 'DBInstances[0].{VpcId:DBSubnetGroup.VpcId,Subnets:DBSubnetGroup.Subnets[].SubnetIdentifier,SgIds:VpcSecurityGroups[].VpcSecurityGroupId}' \
  --output table

# 2) Capture values to variables
VPC_ID=$(aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DBID" \
  --query 'DBInstances[0].DBSubnetGroup.VpcId' --output text)

# Grab two subnets from the RDS subnet group (good for Lambda in same VPC/AZs)
SUBNET1=$(aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DBID" \
  --query 'DBInstances[0].DBSubnetGroup.Subnets[0].SubnetIdentifier' --output text)
SUBNET2=$(aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DBID" \
  --query 'DBInstances[0].DBSubnetGroup.Subnets[1].SubnetIdentifier' --output text)
SUBNET_IDS="$SUBNET1,$SUBNET2"
echo "VPC_ID=$VPC_ID"
echo "SUBNET_IDS=$SUBNET_IDS"

# 3) Get the *actual* SGs attached to the DB (authoritative)
RDS_SG_IDS=$(aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DBID" \
  --query 'DBInstances[0].VpcSecurityGroups[].VpcSecurityGroupId' --output text)
echo "RDS_SG_IDS=$RDS_SG_IDS"

# 3a) (Optional) Check if your earlier SG 'sg-0e61fa3cb22ae2c23' is attached
if echo "$RDS_SG_IDS" | grep -q 'sg-0e61fa3cb22ae2c23'; then
  echo "Yes: sg-0e61fa3cb22ae2c23 is attached to the RDS instance."
else
  echo "No: sg-0e61fa3cb22ae2c23 is NOT attached to the RDS instance."
fi

# 4) Create (or reuse) a Lambda SG in the same VPC
LAMBDA_SG_NAME=lambda-to-rds
LAMBDA_SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=group-name,Values="$LAMBDA_SG_NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [ -z "$LAMBDA_SG_ID" ] || [ "$LAMBDA_SG_ID" = "None" ]; then
  LAMBDA_SG_ID=$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$LAMBDA_SG_NAME" --description "Lambda egress to RDS" \
    --vpc-id "$VPC_ID" --query GroupId --output text)
fi
echo "LAMBDA_SG_ID=$LAMBDA_SG_ID"

# 5) Allow Lambda SG -> RDS SG(s) on port 5432 (idempotent)
for sg in $RDS_SG_IDS; do
  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$sg" \
    --ip-permissions "IpProtocol=tcp,FromPort=5432,ToPort=5432,UserIdGroupPairs=[{GroupId=$LAMBDA_SG_ID,Description=from-lambda}]" \
    2>/dev/null || true
done
