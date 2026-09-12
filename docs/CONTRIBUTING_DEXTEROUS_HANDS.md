# Contributing robot data with dexterous hands

Follow the [data contribution guide](../CONTRIBUTING_DATA.md) for episode naming,
Zarr storage, annotations, and delivery. The requirements below cover dexterous
robot hands and their camera overlays.

## Register the robot and hands

Use an `embodiment` supported by the [platform registry](../egomimic/rldb/embodiment/registry/platforms.yaml)
and a registered [end effector](../egomimic/rldb/embodiment/registry/end_effectors.yaml)
for each active side. Register new hardware before submitting episodes. Each hand
entry must specify its DOF, ordered `joint_names`, joint limits, `mano21` keypoint
topology, and valid slots. Identify the wrist/palm frame used by `obs_ee_pose`.
When supplying a URDF, also register its pose link, keypoint links, and FK tolerance.

Bundle model assets with the registry under `egomimic/rldb/embodiment/registry/urdf/<vendor-or-platform>/`,
including the source/version and applicable redistribution licenses and notices.
Registry model paths are relative to `registry/urdf/`; record each model's SHA-256.
These files ship with the Python package. Mesh files are unnecessary for FK checks.

Declare the platform and hands in the episode's `morphology` attribute. For example:

```yaml
embodiment: dexmate_bimanual
morphology:
  platform: dexmate_nth_poc1
  end_effector:
    left: sharpa_wave_v1_left
    right: sharpa_wave_v1_right
```

The embodiment, platform, active sides, and installed hands must agree.

## Supply joints and keypoints

**Vendors supply both joint values and robot-hand keypoints.** Compute keypoints
during conversion if necessary, using the observed robot configuration. Export
observations and controller commands in their respective arrays.

Map your source joint order to the registry before export. In `features`, add a
`joint_names` list to each observed/commanded hand array (and auxiliary array
when the registry names that chain), exactly matching registry order. This is
your declaration of column meaning; matching shapes cannot prove physical
order. `ZarrWriter` accepts these descriptions through `metadata_override`.
Its `features` override replaces the whole mapping, so include every exported
array's `dtype` and `shape` alongside the joint-name declarations, following
the writer's feature format in the general guide.
For example, a hand description has the form:

```python
features["left.obs_hand_joints"] = {
    "dtype": str(left_observed_joints.dtype),
    "shape": [left_observed_joints.shape[1]],
    "joint_names": list(resolved.end_effectors["left"].joint_names),
}
```

Here `resolved = Embodiment.from_attrs(episode_attributes)` uses the registered
hardware. Populate this declaration only after mapping the source columns to
that order; copying the names does not reorder the numeric data.

For each active `<side>` (`left` or `right`), provide floating-point arrays:

| Array | Shape | Contents |
|---|---|---|
| `<side>.obs_hand_joints` | `(T, J)` | Observed hand joints in registry `joint_names` order |
| `<side>.cmd_hand_joints` | `(T, J)` | Commanded hand joints in the same order |
| `<side>.obs_hand_keypoints` | `(T, 63)` | Observed 21-point robot-hand geometry, flattened as `[x0, y0, z0, …, x20, y20, z20]` |
| `<side>.obs_ee_pose` | `(T, 7)` | Observed pose of the registered wrist/palm frame |

`J` is the registered hand DOF. Use radians for revolute joints and metres for
prismatic joints; match the registered signs, zero positions, and limits. For
every active side on a platform with arms, also supply `<side>.obs_joints` /
`<side>.cmd_joints` with shape `(T, arm_dof)`. Platforms with an auxiliary chain
also require `obs_aux_joints` / `cmd_aux_joints` in `aux.joint_names` order.

Store poses and keypoints in the declared episode reference frame, typically
`robot_base`. Pose layout is `[x, y, z, qw, qx, qy, qz]`, with metre translations
and unit quaternions. Keypoints must describe the physical robot hand and agree
with its observed joints and palm pose, within the registered FK tolerance when
a model is provided.

## Use the MANO-style landmark order

Map robot landmarks to the nearest anatomical equivalents in this 21-slot order:

| Slots | Landmarks, from hand base toward fingertip |
|---|---|
| `0` | Registered wrist/palm origin |
| `1–4` | Thumb: CMC, MCP, IP, tip |
| `5–8` | Index: MCP, PIP, DIP, tip |
| `9–12` | Middle: MCP, PIP, DIP, tip |
| `13–16` | Ring: MCP, PIP, DIP, tip |
| `17–20` | Little finger: MCP, PIP, DIP, tip |

Use the same semantic order on both hands. For hands with fewer fingers, retain
all 63 columns and declare the supported slots in the registry's `keypoints.valid`;
the visualizer ignores the other slots. Document the robot-specific landmark
mapping in the registry.

## Provide camera calibration and synchronized frames

- Supply `images.front_1` as JPEG-encoded RGB frames. Align all observations to
  those frames and provide strictly increasing `int64` UTC capture timestamps in
  `obs_rgb_timestamps_ns`. Set `total_frames` to the retained frame count and `fps`
  to the actual frame rate.
- Include `calibration.reference_frame` and a `calibration.cameras.front_1` entry
  with `K` (`3×4`, `[K_3x3 | 0]`), `resolution: [width, height]`, `model`, distortion
  coefficients where applicable, and `rectified`. Supply rectified images and
  matching intrinsics for the current overlay tool; update `K` after resizing or
  cropping.
- For moving ego cameras, supply `obs_head_pose` with shape `(T, 7)`: the **optical ego camera's pose in
  the episode reference frame at each RGB frame**. Include the calibrated offset
  from the tracked head/device to the optical camera. Optical axes are +X right,
  +Y down, +Z forward. A constant trajectory is appropriate only when the camera
  is fixed in that reference frame. An explicitly fixed camera may instead supply
  a valid `calibration.cameras.front_1.ref_T_cam`.
- Calibrate additional camera streams separately. A static camera uses
  `calibration.cameras.<name>.ref_T_cam` (`4×4`, camera-to-reference transform).
  The current visualizer reads per-frame poses for `front_1` only; moving
  wrist-camera trajectories are not yet supported.

See [coordinate conventions](CONVENTIONS.md). Complete submissions must include
the calibration needed for the ego overlay; estimated or incomplete samples
should be marked `data_status: structural_sample`.

## Verify before delivery

Write episodes with [ZarrWriter](../egomimic/rldb/zarr/zarr_writer.py), supplying
`calibration=` and `data_status=`, and use `metadata_override` for
`schema_version: v3.1` and `morphology`. Include a `features` entry for every array
and mark finished deliveries `data_status: complete`.

From the repository root, install the locked environment and validate every episode:

```bash
uv sync --locked
uv run python -m egomimic.rldb.zarr.validate /path/to/episode.zarr \
  --schema-version --data-status --calibration-block --calibration-present --rgb-timestamps --ego-overlay
uv run python -m egomimic.rldb.zarr.render_check /path/to/episode.zarr \
  --camera front_1 --out outputs/episode-keypoints.mp4
```

`--ego-overlay` checks the canonical `front_1` stream, decoding every retained
frame and checking matching K/resolution, supplied keypoints, and current optical
camera poses (or a valid fixed transform). It does not require wrist calibration.
The report separates structural validity, status eligibility, and the requested
capability. Exit status 1 means validation errors; 3 means a valid but ineligible
status when `--data-status` is required. A structural sample cannot pass the
complete-delivery command just by declaring its status.

Resolve validation errors and review warnings. Watch the preview across the
episode: the left/right hands, fingers, fingertips, and motion should align with
the RGB frames. Review the adjacent JSON report for unavailable overlays or
calibration gaps, and include both artifacts with the delivery. Estimates and
analysis copies must remain labeled `structural_sample`; obtain measured
calibration and synchronized recordings for a complete delivery.

The visualizer projects the **supplied keypoints** using the supplied camera
calibration and registry slot mask. It does not reconstruct keypoints from joints,
fit camera parameters, smooth trajectories, or retarget the hand. Correct export
errors in the conversion pipeline and rerun verification. A successful render
does not replace validation; see [preview options](DELIVERY_PREVIEW.md) for clips
and trajectory overlays.
