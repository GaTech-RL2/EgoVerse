#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
image="${EVA_ROLLOUT_IMAGE:-egomimic-eva:py311}"
container_name="${EVA_ROLLOUT_CONTAINER:-egomimic-eva-live}"
shm_size="${EVA_ROLLOUT_SHM_SIZE:-512m}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
checkpoint_dir="${EVA_CHECKPOINT_DIR:-${repo_root}/external_ckpts}"

if [ -n "${EVA_ARIA_AUTH_DIR:-}" ]; then
  aria_auth_dir="${EVA_ARIA_AUTH_DIR}"
elif [ -n "${XDG_CONFIG_HOME:-}" ]; then
  aria_auth_dir="${XDG_CONFIG_HOME}/egomimic/aria"
elif [ -n "${HOME:-}" ]; then
  aria_auth_dir="${HOME}/.config/egomimic/aria"
else
  echo "Error: set EVA_ARIA_AUTH_DIR when HOME and XDG_CONFIG_HOME are unset."
  exit 1
fi

if [[ "${mode}" != "left" && "${mode}" != "right" && "${mode}" != "both" ]]; then
  echo "Usage: $0 {left|right|both}"
  exit 1
fi

echo "Mode: ${mode}"

# Select CAN devices
CAN_DEVICES=()
case "${mode}" in
  left)
    CAN_DEVICES+=(--device /dev/eva_left_can)
    ;;
  right)
    CAN_DEVICES+=(--device /dev/eva_right_can)
    ;;
  both)
    CAN_DEVICES+=(--device /dev/eva_left_can --device /dev/eva_right_can)
    ;;
esac
echo "Using CAN devices: ${CAN_DEVICES[*]}"
for ((i = 1; i < ${#CAN_DEVICES[@]}; i += 2)); do
  if [ ! -e "${CAN_DEVICES[$i]}" ]; then
    echo "Error: required CAN device ${CAN_DEVICES[$i]} is missing."
    exit 1
  fi
done

# Collect all /dev/video* devices
VIDEO_NODES=()
for v in /dev/video*; do
  [ -e "${v}" ] || continue
  VIDEO_NODES+=("${v}")
done

if [ "${#VIDEO_NODES[@]}" -eq 0 ]; then
  echo "Warning: no /dev/video* devices found."
fi

echo "Using video devices: ${VIDEO_NODES[*]}"

VIDEO_DEVICES=()
for v in "${VIDEO_NODES[@]}"; do
  VIDEO_DEVICES+=(--device "${v}")
done

# Find ALL Intel RealSense devices (8086:0b5b)
realsense_lines=$(lsusb | grep '8086:0b5b' || true)
if [ -z "${realsense_lines}" ]; then
  echo "Error: no Intel RealSense (8086:0b5b) devices found."
  exit 1
fi

RS_DEVICES=()
while IFS= read -r line; do
  [ -z "${line}" ] && continue
  bus=$(awk '{print $2}' <<< "${line}")
  dev=$(awk '{print $4}' <<< "${line}" | sed 's/://')
  path="/dev/bus/usb/${bus}/${dev}"
  if [ -e "${path}" ]; then
    RS_DEVICES+=("${path}")
  else
    echo "Warning: ${path} does not exist, skipping."
  fi
done <<< "${realsense_lines}"

if [ "${#RS_DEVICES[@]}" -eq 0 ]; then
  echo "Error: could not resolve any RealSense /dev/bus/usb paths."
  exit 1
fi

echo "Using RealSense devices:"
for p in "${RS_DEVICES[@]}"; do
  echo "  ${p}"
done

# Mount the USB bus rather than individual device nodes. Both RealSense and
# Aria devices can disconnect/re-enumerate while the container is running, so
# binding only their current bus/device paths makes the container stale.
USB_DEVICE_ARGS=(
  -v /dev/bus/usb:/dev/bus/usb
  --device-cgroup-rule="c 189:* rmw"
)

# Keep pairing and streaming certificates across disposable containers. The
# enclosing mode 0700 protects the private keys created by the Aria SDK.
install -d -m 0700 "${aria_auth_dir}"
aria_auth_dir="$(cd -- "${aria_auth_dir}" && pwd -P)"
ARIA_AUTH_ARGS=(
  --mount "type=bind,source=${aria_auth_dir},target=/root/.aria"
)

# Checkpoints are too large for the image and must survive container
# recreation. Keep the host directory gitignored and expose it read-only to
# the robot runtime.
install -d -m 0755 "${checkpoint_dir}"
checkpoint_dir="$(cd -- "${checkpoint_dir}" && pwd -P)"
CHECKPOINT_ARGS=(
  --mount "type=bind,source=${checkpoint_dir},target=/home/robot/robot_ws/external_ckpts,readonly"
)

echo
echo "Running docker with:"
echo "  ${CAN_DEVICES[*]}"
echo "  ${VIDEO_DEVICES[*]}"
echo "  ${USB_DEVICE_ARGS[*]}"
echo "  Aria auth: ${aria_auth_dir}"
echo "  checkpoints: ${checkpoint_dir} (read-only)"
echo "  shared memory: ${shm_size}"
echo "  container: ${container_name}"
echo "  image: ${image}"
echo

docker run --rm -it --name "${container_name}" --network host \
  --shm-size "${shm_size}" \
  --gpus all \
  "${CAN_DEVICES[@]}" \
  "${VIDEO_DEVICES[@]}" \
  "${USB_DEVICE_ARGS[@]}" \
  "${ARIA_AUTH_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${image}"
