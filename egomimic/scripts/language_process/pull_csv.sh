set -a
source ~/.egoverse_env
set +a

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="auto"
export AWS_REGION="auto"

# get path from argument
R2_PATH=$1
DEST_PATH=$2

s5cmd --endpoint-url "$R2_ENDPOINT_URL" cp \
    "$R2_PATH" \
    "$DEST_PATH"