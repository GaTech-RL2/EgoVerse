
source /home/aloha/RB_Y1_workspace/EgoVerse/emimic/bin/activate

# Define the base name for this dataset (unified naming)
# DATASET_NAME="RBY1_robot_data_mink_elbow"
# RAW_DATA_PATH="../SEW-Geometric-Teleop/projects/collected_data/robot_data_mink_elbow.hdf5"
DATASET_NAME="RBY1_robot_data_mink_elbow"
RAW_DATA_PATH="../SEW-Geometric-Teleop/projects/collected_data/robot_data_mink_elbow.hdf5"
BLACK_IMAGE=true
DELTA_ACTIONS=false
FPS=10

# Step 1: Convert raw SEW-format HDF5 data to LeRobot folder using lowdim config.
python egomimic/rldb/scripts/robomimic_hd5.py \
    --name "${DATASET_NAME}_raw" \
    --raw-path ${RAW_DATA_PATH} \
    --dataset-repo-id "${DATASET_NAME}_raw" \
    --config-path ./egomimic/rldb/configs/RBY1_SEW_lowdim_no_apriltag_HDF5_config.json \
    --output-dir ./datasets/${DATASET_NAME}_lerobot_raw \
    --fps ${FPS} \
    --ignore_episode_keys

# Step 2: Extract arm/hand splits and reformat for training.
python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py \
    ./datasets/${DATASET_NAME}_lerobot_raw/ \
    --output-path ./datasets/${DATASET_NAME}_human_data \
    $( [ "${BLACK_IMAGE}" = true ] && echo "--black-image" ) \
    $( [ "${DELTA_ACTIONS}" = true ] && echo "--add-delta-actions" )
