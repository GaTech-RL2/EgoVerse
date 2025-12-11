REGION=us-east-2
DBID=lowuse-pg
DBUSER=POSTGRES_ADMIN
DBPASS='STRONG_PASSWORD_HERE'
SG_NAME=rds-pg-lowuse-sg

echo "Getting VPC ID"
VPC_ID=$(aws ec2 describe-vpcs --region $REGION \
  --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)

echo "Trying to find existing SG with that name *in this VPC*"
SG_ID=$(aws ec2 describe-security-groups --region $REGION \
  --filters Name=group-name,Values=$SG_NAME Name=vpc-id,Values=$VPC_ID \
  --query 'SecurityGroups[0].GroupId' --output text)

echo "If not found, creating SG"
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group --region $REGION \
    --group-name "$SG_NAME" --description "Allow psql from my IP" \
    --vpc-id "$VPC_ID" --query GroupId --output text)
fi

echo "Adding/ensuring inbound 5432 from *your* IP (ignore duplicate-rule error)"
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region $REGION \
  --group-id "$SG_ID" \
  --ip-permissions "IpProtocol=tcp,FromPort=5432,ToPort=5432,IpRanges=[{CidrIp=${MY_IP}/32,Description=psql-from-home}]" \
  2>/dev/null || true

echo "Creating RDS instance"
aws rds create-db-instance --region $REGION \
  --db-instance-identifier "$DBID" \
  --engine postgres \
  --engine-version 15 \
  --db-instance-class db.t4g.micro \
  --allocated-storage 20 \
  --storage-type gp3 \
  --master-username "$DBUSER" \
  --master-user-password "$DBPASS" \
  --publicly-accessible \
  --backup-retention-period 7 \
  --vpc-security-group-ids "$SG_ID" \
  --no-multi-az

echo "Waiting for RDS instance to be available"
aws rds wait db-instance-available --region $REGION --db-instance-identifier "$DBID"
aws rds describe-db-instances --region $REGION --db-instance-identifier "$DBID" \
  --query 'DBInstances[0].{Host:Endpoint.Address,Port:Endpoint.Port,Public:PubliclyAccessible}' --output table

