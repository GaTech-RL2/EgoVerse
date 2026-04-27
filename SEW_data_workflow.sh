
source /coc/flash7/zhenyang/EgoVerse/emimic/bin/activate

# Define the base name for this dataset (unified naming)
# DATASET_NAME="RBY1_0418_rotatebox"
# RAW_DATA_PATH="../datasets/0418_ye_rotatebox_raw/robot_data_0418_rotate_no_mobile.hdf5"
DATASET_NAME="RBY1_0423_cart_mobile_v6"
RAW_DATA_PATH="../datasets/0423_zhenyang_pushing_cart/0423_v6/robot_data_0423_cart_mobile_fix_aprtag_v6.hdf5" # 0423_converted/robot_data_0423_cart_mobile_v3.hdf5"
BLACK_IMAGE=true
FPS=10

# Step 1: Convert raw SEW-format HDF5 data to LeRobot folder using lowdim config.
python egomimic/rldb/scripts/robomimic_hd5.py \
    --name "${DATASET_NAME}_raw" \
    --raw-path ${RAW_DATA_PATH} \
    --dataset-repo-id "${DATASET_NAME}_raw" \
    --config-path ./egomimic/rldb/configs/RBY1_SEW_lowdim_HDF5_config.json \
    --output-dir ./datasets/${DATASET_NAME}_lerobot_raw \
    --fps ${FPS} \
    --ignore_episode_keys

# Step 2: Extract arm/hand splits and reformat for training.
python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py \
    ./datasets/${DATASET_NAME}_lerobot_raw/ \
    --output-path ./datasets/${DATASET_NAME}_human_data \
    $( [ "${BLACK_IMAGE}" = true ] && echo "--black-image" )