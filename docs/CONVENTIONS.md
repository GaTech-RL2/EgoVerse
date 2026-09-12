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

Robot poses and keypoints use the episode's declared reference frame. For Human
recordings this is usually the SLAM world frame. `obs_head_pose` stores the
optical ego camera's pose in that reference frame at each RGB frame, including
any calibrated offset from the tracked device to the camera. Optical axes are
+X right, +Y down, and +Z forward.

### The calibration block

New exports declare their reference frame and camera calibration in
`calibration`:

```text
calibration.reference_frame        robot_base | slam_world | camera:<name>
calibration.cameras[c].ref_T_cam   camera c's pose in the reference frame
calibration.arm_bases[side]        ref_T_armbase, the arm base pose in it
```

The camera that defines the reference frame needs no `ref_T_cam`; that transform
is identity. A fixed camera can declare `ref_T_cam`. A moving ego camera uses
`obs_head_pose`; one static transform cannot describe its trajectory.

Each camera also declares its projection model and distortion coefficients.

```text
calibration.cameras[c].model        PINHOLE | OPENCV | KANNALA_BRANDT
calibration.cameras[c].distortion   coefficients in that model's order
calibration.cameras[c].rectified    whether the stored frames are rectified
calibration.cameras[c].K            3×4 intrinsics, [K_3x3 | 0]
calibration.cameras[c].resolution   [width, height] of the stored images
```

A camera that declares no model defaults to `PINHOLE` with no coefficients.
Current overlays require pinhole or rectified images and matching intrinsics;
they do not apply lens-distortion correction. Preserve the measured model and
distortion declaration, and update `K` and `resolution` after resizing or cropping.

For an arm-base pose and camera pose in the same reference frame:

```text
base_T_cam = inverse(ref_T_armbase) @ ref_T_cam
cam_T_base = inverse(base_T_cam)
```

Trajectory overlays and dexterous action chunks transform future observations
using the camera pose at the current frame. The displayed image stream and the
coordinate reference frame are separate choices.

### Existing episodes need no migration

Legacy EVA `extrinsics[side]` is read as `base_T_cam`. The calibration reader
uses `camera:front_1` as its reference and derives `ref_T_armbase` by inverting
that stored matrix. Existing supported Human/EVA episodes do not require new
calibration or morphology attributes to retain their established loading paths.

Verify a transform against the measured rig geometry and image overlay. The
translation of `base_T_cam` is the camera origin in the arm-base frame; no
particular camera mounting position is implied by this convention.

## Pose arrays

Pose translations use metres. Quaternion poses use `[x, y, z, qw, qx, qy,
qz]`. A key or function must identify an Euler pose explicitly. Euler poses
use ZYX yaw, pitch, and roll.
