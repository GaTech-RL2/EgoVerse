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

### The calibration block

An episode written after the `calibration` attribute existed names one
reference frame and expresses every pose in it.

```text
calibration.reference_frame        robot_base | slam_world | camera:<name>
calibration.cameras[c].ref_T_cam   camera c's pose in the reference frame
calibration.arm_bases[side]        ref_T_armbase, the arm base pose in it
```

The camera that defines the reference frame needs no `ref_T_cam`. It is the
identity by definition.

`Calibration.base_T_cam(side)` composes the two and returns what the EVA
transform pipeline consumes. An episode that predates the block reaches the
same value through the shim in `egomimic/rldb/zarr/calibration.py`: its
reference frame is `camera:front_1`, so `arm_bases[side]` is the inverse of
the stored `extrinsics[side]`.

### Existing episodes need no migration

The `ref_T_cam` naming records the direction the code already used. It changes
no stored value.

Two writers have ever set `extrinsics`: `eva_to_zarr.py` and `hdf5_to_zarr.py`.
Both pass `Eva.EXTRINSICS` to the writer unmodified, and no write path inverts
it. Every stored EVA episode therefore holds the camera pose in the arm-base
frame, which is what the loader expects. `ActionChunkCoordinateFrameTransform`
inverts the matrix at load time to reach the camera frame. Human writers store
no extrinsics, so only EVA episodes are affected.

To spot-check one episode, read `extrinsics[arm]` from `zarr.attrs` and look at
the translation column. It is the camera origin in the arm-base frame, so it
must place the camera above and behind the arm base. A `cam_T_base` matrix
places the arm base in front of the camera instead. The rotation column for the
camera optical axis gives the same answer: in `base_T_cam` it points forward
and downward in the base frame.

## Pose arrays

Pose translations use metres. Quaternion poses use `[x, y, z, qw, qx, qy,
qz]`. A key or function must identify an Euler pose explicitly. Euler poses
use ZYX yaw, pitch, and roll.
