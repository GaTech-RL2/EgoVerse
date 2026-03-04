set -a
source ~/.egoverse_env
set +a

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="auto"
export AWS_REGION="auto"

# aria download id
ID=1768600563948
path="/home/ubuntu/aria_download"


s5cmd --endpoint-url "$R2_ENDPOINT_URL" sync \
  "s3://rldb/raw_v2/aria/mps_${ID}_vrs/**" \
  "${path}/mps_${ID}_vrs/"

s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/aria/${ID}.json" $path
s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/aria/${ID}.vrs" $path
s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/aria/${ID}_metadata.json" $path