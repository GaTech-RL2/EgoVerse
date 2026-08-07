#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"

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

RS_DEVICE_ARGS=()
for p in "${RS_DEVICES[@]}"; do
  RS_DEVICE_ARGS+=(--device "${p}:${p}")
done

if [ ! -e /dev/aria_usb ]; then
  echo "Warning: /dev/aria_usb not found; Aria passthrough may fail."
fi

echo
echo "Running docker with:"
echo "  ${CAN_DEVICES[*]}"
echo "  ${VIDEO_DEVICES[*]}"
echo "  ${RS_DEVICE_ARGS[*]}"
echo "  -v /dev/aria_usb:/dev/aria_usb"
echo

docker run -it --network host \
    --gpus all \
  "${CAN_DEVICES[@]}" \
  "${VIDEO_DEVICES[@]}" \
  "${RS_DEVICE_ARGS[@]}" \
  -v /dev/aria_usb:/dev/aria_usb \
  robot-env:latest
