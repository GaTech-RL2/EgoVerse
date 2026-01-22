#!/bin/bash
#
# Run script for YAM robot teleoperation Docker container
#
# This script:
# - Mounts the EgoVerse source code for live editing
# - Provides access to USB devices (cameras, VR controller)
# - Provides access to CAN bus for robot control
# - Maps user ID for proper file permissions
#
# Usage:
#   ./run_yam_docker.sh              # Interactive bash shell
#   ./run_yam_docker.sh <command>    # Run specific command
#
# Examples:
#   ./run_yam_docker.sh
#   ./run_yam_docker.sh python egomimic/robot/collect_demo.py --robot-type yam --dry-run

set -e

# Configuration
IMAGE_NAME="egoverse-yam"
CONTAINER_NAME="egoverse-yam-teleop"

# Get the directory where this script is located (EgoVerse root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Host paths to mount
EGOVERSE_PATH="${SCRIPT_DIR}"
DEMOS_PATH="${SCRIPT_DIR}/demos"

# Create demos directory if it doesn't exist
mkdir -p "${DEMOS_PATH}"

# Get current user ID and group ID for proper file permissions
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=$(whoami)

# Get plugdev group ID for USB/ADB device access
PLUGDEV_GID=$(getent group plugdev | cut -d: -f3 2>/dev/null || echo "")

# Build the image if it doesn't exist or if --build flag is passed
if [[ "$1" == "--build" ]] || [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "Building Docker image: ${IMAGE_NAME}"
    docker build \
        -f "${SCRIPT_DIR}/Dockerfile.yam" \
        --build-arg USER_ID=${USER_ID} \
        --build-arg GROUP_ID=${GROUP_ID} \
        --build-arg USERNAME=robot \
        -t ${IMAGE_NAME} \
        "${SCRIPT_DIR}"
    
    # If --build was the only argument, exit after building
    if [[ "$1" == "--build" ]] && [[ $# -eq 1 ]]; then
        echo "Build complete!"
        exit 0
    fi
    
    # Remove --build from arguments if present
    if [[ "$1" == "--build" ]]; then
        shift
    fi
fi

# Remove existing container if it exists
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Collect all USB devices for passthrough
USB_DEVICES=""
for dev in /dev/bus/usb/*/*; do
    if [ -e "$dev" ]; then
        USB_DEVICES="${USB_DEVICES} --device=$dev"
    fi
done

# Check for RealSense devices specifically
REALSENSE_DEVICES=""
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        REALSENSE_DEVICES="${REALSENSE_DEVICES} --device=$dev"
    fi
done


echo "=============================================="
echo "Starting YAM Teleop Docker Container"
echo "=============================================="
echo "Image:        ${IMAGE_NAME}"
echo "Container:    ${CONTAINER_NAME}"
echo "User ID:      ${USER_ID}"
echo "Group ID:     ${GROUP_ID}"
echo "Plugdev GID:  ${PLUGDEV_GID:-not found}"
echo "EgoVerse:     ${EGOVERSE_PATH}"
echo "Demos:        ${DEMOS_PATH}"
echo "=============================================="

# Build group-add arguments
GROUP_ADD_ARGS=""
if [ -n "${PLUGDEV_GID}" ]; then
    GROUP_ADD_ARGS="--group-add ${PLUGDEV_GID}"
fi

# Run the container
# Key options:
#   --network host    : Required for CAN bus access
#   --privileged      : Required for USB device access and CAN
#   -v mounts         : Mount source code and demos for live editing
#   --user            : Run as host user for proper file permissions
docker run -it --rm \
    --name ${CONTAINER_NAME} \
    --network host \
    --privileged \
    ${GROUP_ADD_ARGS} \
    ${USB_DEVICES} \
    ${REALSENSE_DEVICES} \
    -v "${EGOVERSE_PATH}:/home/robot/robot_ws:rw" \
    -v "${DEMOS_PATH}:/home/robot/demos:rw" \
    -v /dev/bus/usb:/dev/bus/usb:rw \
    -v /run/udev:/run/udev:ro \
    -e DISPLAY=${DISPLAY:-:0} \
    -e HOME=/home/robot \
    -e USER=robot \
    -e PYTHONPATH="/home/robot/i2rt:/home/robot/robot_ws" \
    -w /home/robot/robot_ws \
    ${IMAGE_NAME} \
    ${@:-/bin/bash}
