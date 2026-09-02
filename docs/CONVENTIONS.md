# Coordinate conventions

## Transform names

A homogeneous transform named `A_T_B` maps coordinates from frame `B` to frame
`A`.

```text
p_A = A_T_B @ p_B
A_T_C = A_T_B @ B_T_C
```

`A` is the destination frame. `B` is the source frame. For example,
`base_T_cam` is the camera pose in the robot base frame. Its inverse maps
coordinates from the robot base frame to the camera frame.

Use `A_R_B` for a rotation that maps coordinates from frame `B` to frame `A`.

## Episode transforms

EVA episode extrinsics store one `base_T_cam` matrix for each arm. Human
episodes store the head pose as `world_T_head`. The head frame and the camera
frame are the same frame for egocentric human data.

## Pose arrays

Pose translations use metres. Quaternion poses use `[x, y, z, qw, qx, qy,
qz]`. A key or function must identify an Euler pose explicitly. Euler poses
use ZYX yaw, pitch, and roll.
