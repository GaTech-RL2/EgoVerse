# Coordinate conventions

A homogeneous transform named `A_T_B` maps points expressed in frame `B` into
frame `A`:

```text
p_A = A_T_B @ p_B
A_T_C = A_T_B @ B_T_C
```

The name therefore describes the destination frame first and the source frame
second. For example, `base_T_cam` is the camera pose in the robot base frame;
`inv(base_T_cam)` maps a base-frame point into the camera frame. Use the same
ordering for rotations (`A_R_B`).

Episode `extrinsics` values follow this convention. Existing EVA episodes store
one `base_T_cam` per arm. Human head poses are `world_T_head`; for egocentric
data the head frame is the camera frame.

Pose arrays use metres and `[x, y, z, qw, qx, qy, qz]` unless a key or function
explicitly says it contains ZYX yaw, pitch, and roll.
