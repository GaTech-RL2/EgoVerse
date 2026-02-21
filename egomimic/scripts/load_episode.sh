set -a
source ~/.egoverse_env
set +a

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="auto"
export AWS_REGION="auto"

ID=1764285228498
path="/home/ubuntu/download_aria"

# s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/aria/${ID}.json" $path
# s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/aria/${ID}.vrs" $path
# s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp "s3://rldb/raw_v2/aria/${ID}_metadata.json" $path

s5cmd --endpoint-url "$R2_ENDPOINT_URL" sync \
  "s3://rldb/processed_v3/proc_test_aria/${ID}.zarr/**" \
  "${path}/${ID}.zarr/"

# s5cmd --endpoint-url "$R2_ENDPOINT_URL" sync \
#   "s3://rldb/processed_v3/test_eva/1764215784190.zarr/**" \
#   "${path}/1764215784190.zarr/"