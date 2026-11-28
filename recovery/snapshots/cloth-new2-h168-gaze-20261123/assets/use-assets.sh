# Source this file; it only exports paths, never starts hardware.
CLOTH_ASSETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GAZE_ENS=1 GAZE_ENS_NAME=cloth_new2
export GAZE_ENS_DIR="$CLOTH_ASSETS_DIR/gaze"
export VV_MODEL="$CLOTH_ASSETS_DIR/virtual_cam_cloth_new2.npz"
export RESET_CACHE="$CLOTH_ASSETS_DIR/reset_pose_cloth_new2.npz"
