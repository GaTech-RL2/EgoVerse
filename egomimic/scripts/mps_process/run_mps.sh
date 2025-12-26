# Set credentials
export MPS_USER="georgiat_zb658p"
export MPS_PASSWORD="georgiat0001"

# Create local directory
mkdir -p ~/local

# Run
python3 /home/ubuntu/EgoVerse/egomimic/scripts/mps_process/s3_parallel_processor.py \
	--bucket rldb \
	--s3-prefix raw_v2/aria \
	--local-dir ~/local \
	--target-size-gb 375 \
    --features HAND_TRACKING SLAM \
    --retry-failed \
    --include-failed-recordings

