Eva Clean - ROS2 Data Collection

Overview

This folder contains a clean, self-contained data collection utility that subscribes to your existing ROS 2 topics without modifying any of your original repositories. It records synchronized samples at a fixed FPS into an HDF5 file, capturing VR, IK, and camera feeds (Aria and RealSense).

Features

- Subscribes to VR topics produced by `vr18.py` under a configurable prefix (default `/vr`).
- Subscribes to IK joint state topics produced by `eva_replay_ros17.py` (default `/eva_ik`).
- Subscribes to camera topics:
  - Aria stream (default `/cam_high`)
  - RealSense left/right RGB images (defaults `/cam_left/camera/color/image_raw`, `/cam_right/camera/color/image_raw`)
- Fixed-rate sampling (default 30 FPS) with "latest sample wins" buffering for light-weight synchronization.
- Saves a single HDF5 dataset per run with arrays for images and signals; includes `num_steps` metadata.
- Optional start gating using `/vr/save_demo` rising edge.

Quick Start

1) Ensure your sensors and control nodes are running (examples):
   - `vr18.py` (publishes `/vr/...`)
   - `eva_replay_ros17.py` (publishes `/eva_ik/...`)
   - `aria_realsense.launch.py` (starts Aria + RealSense)

2) Run the recorder:

```bash
python3 /home/rl2-bonjour/Eva_Clean/record_episodes.py \
  --dataset-dir /home/rl2-bonjour/datasets/eva_runs \
  --dataset-name episode_0 \
  --max-timesteps 3000 \
  --fps 30 \
  --wait-for-save-demo
```

3) Press the B button on the right VR controller to start (mapped to `/vr/save_demo`). The recorder will capture up to `max_timesteps` frames and then stop automatically. You can stop earlier with Ctrl+C.

Topic Configuration

Defaults are provided in `config/topics.yaml`. You can override any CLI flag to point to different topic names or prefixes.

Output Structure (HDF5)

- `/observations/images/cam_high`      (T, 480, 640, 3) uint8
- `/observations/images/cam_left`      (T, 480, 640, 3) uint8
- `/observations/images/cam_right`     (T, 480, 640, 3) uint8
- `/observations/ik/joint_state`       (T, 7) float64  (6 joints + gripper if present; padded)
- `/observations/ik/engaged`           (T,) bool
- `/observations/vr/l/pose`            (T, 7) float64  (x,y,z, qx,qy,qz,qw)
- `/observations/vr/r/pose`            (T, 7) float64
- `/observations/vr/l/delta`           (T, 7) float64
- `/observations/vr/r/delta`           (T, 7) float64
- `/observations/vr/l/rpy`             (T, 3) float64
- `/observations/vr/r/rpy`             (T, 3) float64
- `/observations/vr/l/engaged`         (T,) bool
- `/observations/vr/r/engaged`         (T,) bool
- `/observations/vr/l/gripper_act`     (T,) int8
- `/observations/vr/r/gripper_act`     (T,) int8
- `/observations/vr/l/gripper_pos`     (T,) float32
- `/observations/vr/r/gripper_pos`     (T,) float32
- `/observations/vr/l/side_trigger`    (T,) int8
- `/observations/vr/r/side_trigger`    (T,) int8
- `/timestamps/now`                    (T,) float64 (wall-clock seconds)
- File attributes: `num_steps`, `fps`, and topic metadata

Notes

- Image sizes are assumed to be 480x640x3 (as per your current pipelines); if your RealSense setup uses different dimensions, override via CLI or adjust preprocessing.
- This tool does not change or import code from your original repos; it only subscribes to topics.

License

Internal use for Eva data collection.


