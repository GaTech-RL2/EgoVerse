# Converting Sharpa LeRobot datasets to EgoVerse

`egomimic.scripts.sharpa_process.sharpa_to_zarr` converts Sharpa's published
LeRobot seasons (v2.1 or v3.0) into EgoVerse Zarr v3.1 episodes. The output is
for the `dexmate_bimanual` embodiment (NTH POC1.0 platform with Sharpa Wave
hands). This guide covers running the converter, what it writes, and what
Sharpa still needs to supply before an episode can be delivered as
`data_status: complete`.

## Install

From the repository root:

```bash
uv sync --locked
```

Run every command below from the repository root with `uv run`.

## Run

The converter expects a dataset folder of `season_POC<rig>_<YYYY_MM_DD_HH_MM_SS>_train/`
directories, each holding `lerobotv2.1/` and/or `lerobot3.0/`. A single season
directory, or a bare LeRobot root, also works as `--dataset-root`.

```bash
# Smoke run: the first 300 frames of one episode, validated.
uv run python -m egomimic.scripts.sharpa_process.sharpa_to_zarr \
  --dataset-root /data/RoCo_TaskBoardAssembly \
  --seasons season_POC22002_2026_08_15_14_07_26_train --episodes 0 \
  --task-name task_board_assembly --output-dir converted/ \
  --max-frames 300 --validate

# One full episode, with a front-camera preview video.
uv run python -m egomimic.scripts.sharpa_process.sharpa_to_zarr \
  --dataset-root /data/RoCo_TaskBoardAssembly \
  --seasons season_POC22002_2026_08_15_14_07_26_train --episodes 0 \
  --task-name task_board_assembly --output-dir converted/ --validate --save-mp4

# Every episode of every POC22002 season, read from the v3.0 layout.
uv run python -m egomimic.scripts.sharpa_process.sharpa_to_zarr \
  --dataset-root /data/RoCo_TaskBoardAssembly --seasons 'season_POC22002_*' \
  --lerobot-version v3.0 --task-name task_board_assembly --output-dir converted/

# A complete delivery: measured calibration and a real capture clock.
uv run python -m egomimic.scripts.sharpa_process.sharpa_to_zarr \
  --dataset-root /data/RoCo_TaskBoardAssembly --task-name task_board_assembly \
  --output-dir converted/ --calibration rig_calibration.yaml \
  --utc-timestamp-column observation.utc_timestamp_ns --validate
```

| Option | Effect |
|---|---|
| `--seasons GLOB`, `--episodes 0,3,5-7` | Select seasons by directory name and episodes by index (default: all). |
| `--lerobot-version auto\|v2.1\|v3.0` | `auto` prefers a root at the season directory, then `lerobotv2.1/`, then `lerobot3.0/`. |
| `--task-name`, `--task-description` | Required because the source task string is the placeholder `north ces task`. |
| `--calibration FILE` | Rig calibration YAML; see [Calibration](#fill-in-the-calibration). |
| `--utc-timestamp-column NAME` | A parquet column of int64 UTC capture nanoseconds. |
| `--source-utc-offset +08:00` | Time zone of the season-folder times, used only when the clock is synthesized. |
| `--include-head-right` | Also store `head_right` as `images.head_right`. |
| `--keep-vendor-arrays` | Also store raw `sharpa.*` arrays (see [Not stored](#not-stored)). |
| `--clip-joint-limits` | Clip joints to their limits before FK. For local experiments only. |
| `--max-frames N` | Keep the first `N` frames. A truncated episode is never `complete`. |
| `--jpeg-quality 90`, `--chunk-timesteps 100` | Image and storage settings. |
| `--validate` | Validate each episode and write `<hash>.validation.json`. |
| `--save-mp4` | Write a `<hash>.mp4` preview of the front camera. |
| `--overwrite` | Replace an existing output of the same recording. |

For each episode the converter writes:

- `<output-dir>/<episode_hash>.zarr`. The hash is the UTC time of the first frame, formatted `YYYY-MM-DD-HH-MM-SS-ffffff` at millisecond resolution.
- `<episode_hash>.conversion.json`, holding the episode attributes, the conversion report and a validation summary.
- `<episode_hash>.validation.json` (with `--validate`) and `<episode_hash>.mp4` (with `--save-mp4`).

The converter names every requested episode before decoding anything. A run
fails up front when two episodes share a name, or when an output already holds
a different recording. It never replaces another recording's output, even with
`--overwrite`.

Converting a 9,769-frame episode with three cameras took about 90 s on 8 CPU
cores, with a peak resident memory of 2.7 GB.

Failed episodes are listed and the run continues. The exit status is 0 when
everything converted, 1 when any episode failed, and 2 when everything converted
but `--validate` found errors.

## What is stored

Sharpa's 65-column `observation.state` and `action` vectors are split into the
canonical arrays. The values are unchanged, in radians and float32. Every array
is in `robot_base`, the root link of `NTH_POC1_0_URDF_250905v4.urdf`.

| LeRobot source | EgoVerse array | Contents |
|---|---|---|
| `observation.state[0:7]`, `[29:36]` | `left.obs_joints`, `right.obs_joints` `(T, 7)` | Arm joints `<side>_arm_joint_1..7` |
| `observation.state[7:29]`, `[36:58]` | `left.obs_hand_joints`, `right.obs_hand_joints` `(T, 22)` | Hand joints in registry `sharpa_wave_v1_<side>` order |
| `observation.state[58:65]` | `obs_aux_joints` `(T, 7)` | `lower_body_joint_1..5`, `neck_joint_1..2` |
| `action` (same columns) | `<side>.cmd_joints`, `<side>.cmd_hand_joints`, `cmd_aux_joints` | Absolute joint position setpoints |
| FK of `observation.state` | `<side>.obs_ee_pose` `(T, 7)` | Palm pose (`<side>_hand_C_MC`), `[x, y, z, qw, qx, qy, qz]`, metres |
| FK of `action` | `<side>.cmd_ee_pose` `(T, 7)` | Commanded palm pose |
| FK of `observation.state` | `<side>.obs_hand_keypoints` `(T, 63)` | MANO-21 hand keypoints, metres |
| Calibration and FK | `obs_head_pose` `(T, 7)` | Optical pose of the front camera (calibration only) |
| Vendor column or synthesis | `obs_rgb_timestamps_ns` `(T,)` int64 | UTC capture time of each frame |
| `observation.images.head_left` | `images.front_1` | JPEG frames, 480×480 |
| `observation.images.wrist_left` / `wrist_right` | `images.left_wrist` / `images.right_wrist` | JPEG frames |

`features[...].joint_names` declares the column order of every joint array.
Episode attributes include:

- `embodiment: dexmate_bimanual`, plus `morphology` naming `dexmate_nth_poc1` and `sharpa_wave_v1_{left,right}`
- `schema_version: v3.1`
- `vendor_recording_id` (`<dataset>/<season>/<episode_index>`) and the `source_*` provenance fields
- `rig_name`, `license` (read from the dataset README) and `timestamp_source`
- `conversion_report`

A season that lacks a camera is converted without it, and the report records the
gap. For example, the v2.1 export of the four POC22061 Origami seasons from
2026-05-23 to 05-25 has no `head_right` (their v3.0 export does).

### Derived by forward kinematics

Palm poses, keypoints and the camera trajectory are computed from the joint
values with the registry URDFs. The platform's `<side>_hand_base_link` is
composed directly with the hand URDF's `<side>_hand_flange` root.

The validator's `fk_residual` check recomputes keypoints from the same joints
and URDFs, so it can only confirm internal consistency (joint order, units and
frames). It cannot confirm that the URDF matches the physical robot.
Independently measured keypoints or palm poses would make that check
meaningful.

The one independent signal in the published data does not agree. Where
`observation.state.tcp` is populated, its rotation follows the FK of the same
side's distal arm link, but its position does not match any FK link under any
rigid, offset or scaled fit (best 57 mm RMS), and it moves 3–10× more than the
FK palm. Either `tcp` is not the hand position, or the arm chain or joint
convention used by Sharpa's controller differs from
`NTH_POC1_0_URDF_250905v4.urdf`. Until Sharpa confirms the arm joint order,
signs and zeros, or supplies reference pairs of joint vectors and controller
TCP poses, treat the FK-derived poses and keypoints as unverified.

### Not stored

| Source | Reason |
|---|---|
| `observation.tactile` (60 values per frame) | The Sharpa hand registry entries declare no tactile taxel count, channels or physical unit. EgoVerse cannot interpret the values without them. |
| `observation.state.joint_torque`, `observation.state.tcp` | No contract array exists. |
| `observation.images.tactile_*` videos | Never stored. |
| `timestamp` | It is `frame_index / fps` within the episode, not a capture clock. |

`--keep-vendor-arrays` stores these columns unchanged:

- `sharpa.observation_state` and `sharpa.action`
- `sharpa.joint_torque` and `sharpa.tcp`
- `sharpa.observation_tactile`, when present

These arrays are a vendor extension outside the EgoVerse contract. The
validator ignores them apart from requiring their `features` entries, which the
converter writes.

## Capture clock

When `--utc-timestamp-column` names an integer column that is present, the
converter stores it as `obs_rgb_timestamps_ns`. The values must strictly
increase. The column should hold the capture time of the `head_left` frame in
each row, in UTC nanoseconds.

Without such a column the clock is synthesized in integer arithmetic:

```
season_start_utc_ns + (frames_before_episode * 10**9) // fps + (i * 10**9) // fps
```

The season start is parsed from the folder name and converted with
`--source-utc-offset`. `frames_before_episode` is the sum of the lengths of the
season's earlier episodes, and it must match the parquet `index` column. The
episode records `timestamp_source: synthesized_from_season_name_and_frame_index`.

A synthesized clock gives an episode unique, increasing times and a stable name,
but it is not a measured capture time. It keeps the episode a
`structural_sample`.

## What each missing input blocks

| Missing input | Consequence |
|---|---|
| Measured front-camera calibration | No `calibration` block, `obs_head_pose` or intrinsics. Keypoint overlays and `--ego-overlay` are unavailable, and the episode cannot be `complete`. |
| Wrist-camera calibration | Wrist streams lack a camera matrix (a `camera_coverage` warning). |
| Real capture clock | The clock is synthesized, and the episode cannot be `complete`. |
| Joint values within the URDF/registry limits | The validator reports an integrity error and the episode cannot be `complete` (see below). |
| Tactile units in the hand registry | Tactile values are not stored under the contract keys. |

### Joint-limit conflict

In the published data, hand joints leave the limits in the delivered hand URDF
and the EgoVerse registry:

- RoCo `season_POC22002_2026_08_15_14_07_26_train`, episode 0: the observed
  `right_middle_DIP` exceeds its 1.3963 rad upper limit on 2,562 of 9,769
  frames, by up to 0.152 rad. The commanded middle, ring and pinky DIP joints of
  both hands exceed it on 6,499–9,117 frames, by up to 0.1745 rad.
- Origami `season_POC22032_2026_05_14_19_21_01_train`, episode 0: `pinky_CMC`
  falls below its lower limit, and `pinky_MCP_FE` and `pinky_DIP` exceed their
  upper limits.

Across both public datasets, 4,932 of 5,243 episodes (94%) have a hand joint
outside the limits. Commanded DIP joints sit at exactly π/2 against a limit of
1.3963 rad, observed DIP joints reach 1.617 rad, and observed pinky CMC reaches
0.49 rad against 0.2618.

The validator treats any hand joint outside the registry limits as an integrity
error, so such episodes fail validation. The converter writes the values
unchanged and lists every affected joint in
`conversion_report.joint_limits.source_excursions` (frame counts, largest
excess, and which limit). It also prints a warning.

Please check whether the data, the URDF limits, or the joint zero offsets are
wrong. Then send corrected data or corrected URDF limits. `--clip-joint-limits`
hides the conflict for local experiments only. Clipped episodes stay
`structural_sample`.

## Fill in the calibration

Copy `egomimic/scripts/sharpa_process/calibration_template.yaml` and replace
every `FILL_*` placeholder with measured values. The converter rejects any file
that still contains one. Structure:

- **Top-level keys:** a rig name such as `POC22002`, or `default`.
- **Camera keys:** `head_left`, `wrist_left`, `wrist_right`, and `head_right`.
- **Fields for each camera:**
  - `K`: the 3×3 intrinsic matrix in pixels, for the stored 480×480 frames.
  - `model` and `distortion`.
  - `resolution: [width, height]`.
  - `rectified`.
  - `parent_link`: a platform or hand URDF link the camera is rigidly mounted on.
  - `parent_link_T_cam`: the camera pose in that link, as a 4×4 transform. Use the optical frame: +X right, +Y down, +Z forward, metres.

The current overlay tools project only rectified pinhole images. Deliver
rectified frames with `model: PINHOLE` and `distortion: []`, or describe the
distortion so the frames can be rectified.

The converter stores the intrinsics in `calibration.cameras`. It computes
`obs_head_pose[t] = FK(parent_link)[t] @ parent_link_T_cam` for `head_left` and
records every mount in `camera_mounts`. The contract has no per-frame slot yet
for wrist camera poses.

A camera whose parent link is fixed to `robot_base` would instead get a static
`ref_T_cam`. On NTH POC1.0 only `root_link` and `lower_body_base_link` are
fixed.

## Reach `data_status: complete`

An episode is `complete` only when all of these hold:

1. `--calibration` gives measured `head_left` intrinsics and mount for the rig.
2. `--utc-timestamp-column` names a real capture clock.
3. No joint value is outside its limits, and `--clip-joint-limits` is not used.
4. The episode is not truncated with `--max-frames`.

The conversion report lists the reasons for any `structural_sample`.

## Verify

```bash
uv run python -m egomimic.rldb.zarr.validate converted/<hash>.zarr \
  --schema-version --data-status --calibration-block --calibration-present \
  --rgb-timestamps --ego-overlay
uv run python -m egomimic.rldb.zarr.render_check converted/<hash>.zarr \
  --mode keypoint --max-frames 300 --out converted/<hash>_keypoints.mp4
```

The validator exits with 1 for errors and 3 for a valid episode whose status is
not `complete`. Without calibration, `render_check` writes the RGB video with
"Overlay unavailable" and explains why in its JSON report.

Watch the overlay across the episode: the projected hands should follow the
robot hands in the image. See [VALIDATION.md](VALIDATION.md) and
[CONTRIBUTING_DEXTEROUS_HANDS.md](CONTRIBUTING_DEXTEROUS_HANDS.md).
