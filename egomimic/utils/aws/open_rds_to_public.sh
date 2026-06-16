#!/bin/bash
set -e

REGION=us-east-2
DBID=lowuse-pg-east2

echo "=== Opening RDS instance to external connections (0.0.0.0/0) ==="

# Get security groups attached to the RDS instance
SG_IDS=$(aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DBID" \
  --query 'DBInstances[0].VpcSecurityGroups[].VpcSecurityGroupId' --output text)

echo "Found security groups: $SG_IDS"
echo ""

# Add 0.0.0.0/0 to each security group
for SG_ID in $SG_IDS; do
  echo "Adding 0.0.0.0/0 to security group $SG_ID..."
  
  aws ec2 authorize-security-group-ingress \
    --region "$REGION" \
    --group-id "$SG_ID" \
    --ip-permissions "IpProtocol=tcp,FromPort=5432,ToPort=5432,IpRanges=[{CidrIp=0.0.0.0/0,Description=PostgreSQL-access-from-all-IPs}]" \
    2>/dev/null && echo "✅ Successfully added 0.0.0.0/0 to $SG_ID" || echo "⚠️  Rule may already exist for $SG_ID (this is OK)"
done

echo ""
echo "=== Verifying RDS Public Access ==="
aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DBID" \
  --query 'DBInstances[0].{PubliclyAccessible:PubliclyAccessible,Endpoint:Endpoint.Address}' --output table

echo ""
echo "=== Done ==="
echo "The RDS instance should now be accessible from any IP address on port 5432."
echo ""
echo "⚠️  SECURITY WARNING: The database is now open to the internet."
echo "   Make sure you have strong passwords and consider restricting access if possible."
