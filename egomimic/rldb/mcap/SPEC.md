# EgoVerse raw-data spec 0.1 (MCAP)

The reference for **raw data**: what a raw recording should contain so we can rebuild any
training format from it later (Lance tables, training videos, new formats) without asking
the vendor anything. Vendors deliver **Zarr first**, which is what we train on
([CONTRIBUTING_DATA.md §6](../../../CONTRIBUTING_DATA.md)), and the raw recordings
alongside it (§6.6 there). Raw data may come in any format; this document, written as
an MCAP layout, lists the information and fields it should carry. A vendor who does
deliver MCAP follows it exactly: one raw, self-contained MCAP recording per episode.

- **Raw, not processed.** Original sensor data at its own rates: no resampling, no
  aligning streams to one clock rate, no undistorting, cropping or downscaling video. We
  convert on our side.
- **Self-contained.** Calibration, camera positions and episode info live inside the
  file. No side JSONs, no constants in our converters.
- **Checked on arrival.** Every MCAP episode is run through the validator
  (`python -m egomimic.rldb.mcap.validate`), which lists what is missing or wrong.

The machine-readable version of this document is [`spec.py`](spec.py); the message
definitions are [`proto/egoverse.proto`](proto/egoverse.proto) (generated from
[`schemas.py`](schemas.py)). The three change together. Proto field numbers are never
changed or reused; a new field gets a new number.

Spec 0.1 covers **human (egocentric) recordings** and **Eva robot recordings with
grippers**. Dexterous hands, tactile, humanoid / auxiliary joints (e.g. Sharpa) are not
in 0.1.

## 1. What arrives per episode

```
episode_<id>/
├── episode.mcap      # everything: video, sensors, poses, calibration, episode info
└── annotation.mcap   # optional: /task spans and /subtask-annotation labels
```

- **An episode is one whole recording session, never a pre-cut clip.** It becomes one
  row of `recordings`; the tasks performed in it become `tasks` rows.
- The label topics may instead be inline in `episode.mcap` (as ABC-130k does), but not in
  both files. `annotation.mcap` holds nothing but label topics: no metadata records
  (the episode info lives in `episode.mcap` only).
- MCAP with protobuf encoding; any chunk compression (zstd, lz4) is fine.

## 2. Clock

- **Every message's MCAP log time is its capture time in absolute Unix nanoseconds**, in
  both files. Devices that stamp on their own clock (e.g. Aria device time) convert
  with their device-to-UTC offset before writing.
- Within a topic, log times never decrease; each camera's strictly increase (every
  frame has its own time).
- `start_time_unix` / `end_time_unix` in the episode info are **milliseconds** and must
  match the recorded data within 1 s.
- **Steps clock.** The main camera (`/main-camera`, human) or top camera (`/top-camera`,
  robot) defines the Lance `steps` rows: one row per decoded frame. Every other stream is
  aligned to those frame times during conversion; wrist and SLAM cameras stay separate
  videos and add no rows.

## 3. Who, what, where: episode info

One MCAP metadata record named **`session-metadata`**. All values are strings.

The first five keys are the required metadata of every EgoVerse episode, with the same
names and rules as a zarr episode ([CONTRIBUTING_DATA.md §6.3](../../../CONTRIBUTING_DATA.md)):
real values only (empty strings and placeholders such as `unknown`, `n/a`, `tbd`, `test`
are rejected). Keys are snake_case; a key not in this table fails validation, and
placeholders are rejected in every text key.

| Key | Example | Meaning |
|---|---|---|
| `task_name` | `fold_clothes` | Task name; reuse an existing one if it fits. |
| `operator` | `op_07` | Stable ID of who performed it. |
| `lab` | `rl2` | Who the data came from: short, stable, lowercase (`a-z`, `0-9`, `_`, `-`), e.g. `rl2`, `mecka`. Never changes once chosen. |
| `scene` | `kitchen_a` | Where it was recorded. |
| `objects` | `baby_shirt_2t,basket` | Objects involved, comma-separated; every entry a real name. |
| `missing_metadata` | `scene` | *Optional.* Comma-separated required keys the source data truly does not have; leave those keys out. Only `scene` and `objects` can be listed: `task_name`, `operator` and `lab` are always required. **Only if absolutely necessary**: such episodes are much harder to filter and train on, and fail validation without `--allow-missing`. |
| `task_description` | `fold the 2T shirt on the blue table` | *Optional.* Full task text. |
| `vendor_recording_id` | `acme-rec-0001` | Your own ID for this recording. **Stable across every re-export or correction** (new labels, fixed calibration, re-run poses). Zarr episodes trimmed from it name it in their `vendor_recording_id` attr. |
| `embodiment` | `human_bimanual` | One of `human_{bimanual,left_arm,right_arm}`, `eva_{bimanual,left_arm,right_arm}`. Must match the topics in the file. |
| `rig_name` | `aria_gen1` | Capture setup. |
| `device_serial` | `1WM103600M1292` | Physical unit; ties the episode to its calibration. |
| `start_time_unix`, `end_time_unix` | `1736651249664` | Recording bounds, Unix **ms**. |
| `spec_version` | `0.1` | Version of this spec the file follows. |
| `license` | `CC-BY-4.0` | License the data is under. |
| `camera_timestamp_ref` | `mid` | *Human only.* Where in the exposure each frame is stamped: `start`, `mid` or `end`. |
| `shutter` | `rolling:5ms` | *Human only.* `global`, or `rolling:<readout>ms`. |
| `hand_joints` | see §6 | *Human only.* Names of the hand joints, in the order `HandKeypoints.positions` lists them. |

**IDs.** Our recording ID is
`egomimic.rldb.lance.schema.recording_id(lab, vendor_recording_id)` (UUIDv5 of
`<lab>/<vendor_recording_id>`); the validator prints it. Task and annotation IDs are
computed downstream from their content, so vendors send none.

**Tasks.** Without `/task` topics, `task_name` / `task_description` / `objects` describe one task
spanning the whole recording: [first steps frame, last steps frame + one frame), as
every span is half-open. With `/task`, the spans are authoritative and these keys
only summarize the session (e.g. `task_name=laundry` over iron / fold / pack spans); no extra
task is made from them.

## 4. Topics

Message types are Foxglove's (`foxglove-schemas-protobuf`) where one exists, otherwise
`egoverse.*` from `proto/egoverse.proto`. Topic names, message types and required fields
are checked exactly.

### Human / egocentric

| Topic | Type | | Rate | Notes |
|---|---|---|---|---|
| `/main-camera` | `foxglove.CompressedVideo` | MUST | ≥ 30 fps | Head-mounted RGB; the steps clock. |
| `/main-camera-info` | `foxglove.CameraCalibration` | MUST | once | See §7. |
| `/slam-left-camera`, `/slam-right-camera` | `foxglove.CompressedVideo` | optional | ≥ 30 fps | Each with its `-info`. |
| `/left-wrist-camera`, `/right-wrist-camera` | `foxglove.CompressedVideo` | optional | ≥ 30 fps | Each with its `-info`. |
| `/imu` | `egoverse.Imu` | MUST | ≥ 200 Hz | Raw: gravity included, no fusion. |
| `/imu-info` | `egoverse.ImuInfo` | MUST | once | Noise model; datasheet values are fine. |
| `/head-pose` | `foxglove.PoseInFrame` | MUST | ≥ 30 Hz | `frame_id = "world"`, pose = T_world_head. |
| `/head-pose-quality` | `egoverse.TrackingQuality` | MUST | one per head pose | `OK`, `LOST`, `RELOCALIZED` (first pose after a SLAM reset) or `UNKNOWN`. |
| `/{left,right}-hand-keypoints` | `egoverse.HandKeypoints` | MUST per arm in use | tracker rate | See §6. |
| `/eye-gaze` | `egoverse.EyeGaze` | optional | tracker rate | Yaw / pitch / depth in a frame between the eyes (+x left, +y up, +z forward). |
| `/tf_static` | `foxglove.FrameTransforms` | MUST | once | See §5. |

No arm, gripper or action topics: actions are derived from future hand and head poses.

### Robot (Eva with grippers)

`{side}` is `left` and/or `right`, as the embodiment uses.

| Topic | Type | | Notes |
|---|---|---|---|
| `/top-camera` | `foxglove.CompressedVideo` | MUST | ≥ 30 fps; the steps clock. With `-info`. |
| `/{side}-wrist-camera` | `foxglove.CompressedVideo` | MUST | ≥ 30 fps. With `-info`. |
| `/{side}-arm-state` | `egoverse.ArmState` | MUST | Measured joints (names, position, velocity, torque) and `ee_pose`. |
| `/{side}-arm-action` | `egoverse.ArmAction` | MUST | The command sent at that time: target joints and `ee_pose`. |
| `/{side}-ee-state` | `egoverse.GripperState` | MUST | Measured opening, 0 = closed, 1 = fully open. |
| `/{side}-ee-action` | `egoverse.GripperState` | MUST | Commanded opening, same range. |
| `/tf_static` | `foxglove.FrameTransforms` | MUST | See §5. |

`ArmState` / `ArmAction`: `frame_id` is the arm-base frame (`left_arm_base`,
`right_arm_base`); every joint array has one entry per `joint_names` entry, in that
order; joints in rad (m for prismatic), velocities rad/s, torques N·m; `ee_pose` is
T_base_ee as 16 numbers, a **row-major** 4×4. ABC-130k's own `RobotState` /
`GripperState` are converted to these types before validation.

### Both profiles: tasks and labels (optional)

| Topic | Type | Notes |
|---|---|---|
| `/task` | `egoverse.Task` | One message per task performed: `name` (slug), `description`, `objects`; starts at its log time, ends at `end_time_unix_ns` (after the start). |
| `/subtask-annotation` | `egoverse.Annotation` | A language label `data` over [log time, `end_time_unix_ns`); `end_time_unix_ns = 0` is a point event (one frame wide). `annotation_type` is `language` (the default when empty); `annotation_source` defaults to the episode's `lab`. |

Every label lies **inside one task**; without `/task`, inside the whole-recording task
made from the episode info. All spans lie inside the recorded data. Labels added or corrected after delivery go through
the Lance annotation review (`egomimic/rldb/lance/annotations.py`), not a re-export.

## 5. Frames and conventions

- Right-handed frames; metres, radians, seconds.
- Quaternions are Foxglove's named `x, y, z, w` fields and must be unit length.
- A transform or pose is **T_parent_child**: the child frame's pose in the parent
  (Foxglove's `FrameTransform` / `PoseInFrame` convention).
- Every camera frame is a Foxglove optical frame: +x right, +y down, +z forward, in the
  raw sensor orientation (see `upright_rotation_cw_deg`, §7). A camera's video,
  `-info` and `-info-extra` all name that one frame.
- **Every frame any topic names must reach `world`** through `/tf_static` plus the
  moving links the profile defines:
  - **Human:** `world → head` moves via `/head-pose`. `/tf_static` links `head` to each
    camera, the IMU, the gaze frame, and any frame hands are given in. The main camera
    must be rigidly mounted on the head: a `/tf_static` path from `head` to its frame.
  - **Robot:** `world → {side}_arm_base` and the top camera in `/tf_static`;
    `{side}_arm_base → {side}_ee` moves via `/{side}-arm-state`; the wrist camera hangs
    off `{side}_ee` in `/tf_static`.

## 6. Hands

`HandKeypoints`, one message per hand per tracker frame:

| Field | |
|---|---|
| `frame_id` | Frame `positions` and `wrist_pose` are in (linked to `world`). |
| `positions` | `(x, y, z)` in m per `hand_joints` entry, flattened; or empty for devices without keypoints. |
| `confidence` | Whole-hand tracking confidence in [0, 1]. |
| `joint_confidence` | Optional: one value in [0, 1] per joint. |
| `wrist_pose` | Optional: T_frame_wrist. Required when `positions` is empty. |
| `valid` | `false` when the hand is not tracked. **Never write zeros instead.** |

**`hand_joints`** names the joints the device tracks, in its own order, from this
vocabulary: the 21 MANO joints `wrist`, `{thumb}_{cmc,mcp,ip,tip}`,
`{index,middle,ring,pinky}_{mcp,pip,dip,tip}`, plus `palm_center`. Each name at most
once; `wrist` is required. Joints with MANO names are what our MANO fit matches; extras
are kept but not fitted. A device that already outputs MANO sends the canonical order:

```
wrist,thumb_cmc,thumb_mcp,thumb_ip,thumb_tip,index_mcp,index_pip,index_dip,index_tip,
middle_mcp,middle_pip,middle_dip,middle_tip,ring_mcp,ring_pip,ring_dip,ring_tip,
pinky_mcp,pinky_pip,pinky_dip,pinky_tip
```

and a device with its own layout lists it as is, e.g. Aria MPS's 21 landmarks (no thumb
CMC, a palm center instead):

```
thumb_tip,index_tip,middle_tip,ring_tip,pinky_tip,wrist,thumb_mcp,thumb_ip,
index_mcp,index_pip,index_dip,middle_mcp,middle_pip,middle_dip,ring_mcp,ring_pip,
ring_dip,pinky_mcp,pinky_pip,pinky_dip,palm_center
```

## 7. Cameras

- **Video.** H.264 or H.265, Annex B byte stream, `CompressedVideo.format` `h264` /
  `h265`. **Each message is exactly one frame**, stamped with that frame's capture time;
  the first message is a keyframe and carries SPS/PPS (repeat them at every keyframe).
  **No B-frames**, so messages decode one-to-one into frames in timestamp order (as in
  `foxglove.CompressedVideo`); with x264 pass `-bf 0`, with x265 `bframes=0`.
  Original resolution, frame rate and orientation: no resizing, cropping, undistorting
  or rotating. Send the video as the device stores it, or lightly compressed (H.264 /
  H.265 at a high-quality setting), encoded at most once.
- **Calibration** `<camera>-info`: exactly one `foxglove.CameraCalibration` message,
  finite values, principal point inside the image:
  `width` / `height` equal the decoded frames; `K` has fx, fy > 0. Name the lens model in
  `distortion_model` and give exactly its number of coefficients in `D` (an empty model
  with no coefficients is a pinhole camera):

  | `distortion_model` | `K` | `D` |
  |---|---|---|
  | `plumb_bob` | fx, fy, cx, cy | k1, k2, p1, p2, k3 |
  | `rational_polynomial` | fx, fy, cx, cy | k1, k2, p1, p2, k3, k4, k5, k6 |
  | `kannala_brandt` | fx, fy, cx, cy | k1, k2, k3, k4 |
  | `fisheye624` (Aria) | f, f, cx, cy | the 12 coefficients after f, cx, cy, in projectaria order (k0–k5, p0, p1, s0–s3) |

- **Optional** `<camera>-info-extra`, `egoverse.CameraInfoExtra`, published once:
  `valid_radius` (px from the principal point where the lens model holds; 0 = whole
  image) and `upright_rotation_cw_deg` (0, 90, 180 or 270: the clockwise turn that makes
  a frame upright, e.g. 90 for Aria's sideways RGB sensor).

## 8. Please don't

- Resize, crop, rotate or undistort video, or re-encode it more than once. One light
  compression pass is fine. *(Devices that store frames as images, e.g. Aria VRS,
  encode them once to H.264 / H.265 at full resolution.)*
- Pack several frames into one video message.
- Resample or align streams to one rate; we match by timestamp.
- Trim streams to the camera's span; deliver every sample.
- Write zeros when tracking is lost; set `valid = false`.
- Remove gravity from, filter or fuse the IMU data.
- Ship calibration as a separate file.

## 9. How a delivery is checked

`python -m egomimic.rldb.mcap.validate episode_<id>/ [...]` prints `PASS` or every
problem per episode (with the episode's `recording_id`), and exits 1 if any fails. Every
problem fails the episode.

| Group | Checks |
|---|---|
| everything there | Chunk CRCs valid; required topics present with the exact type and encoding, one channel each; once-topics published once; required fields non-empty on every message; exactly one `session-metadata` record; every episode-info key known and present, values not placeholders, the required metadata real values (waivers in `missing_metadata` fail unless `--allow-missing`; only `scene` and `objects` can be waived), integers / enums valid; `embodiment` matches the topics; `annotation.mcap` holds only label topics. |
| video usable | Codec exactly `h264` / `h265`, the same on every message; every message holds a coded picture; first message a keyframe; no B-frames; one frame per message; frames decode without bitstream errors; size matches the calibration; ≥ 30 fps. |
| clocks | Unix ns (after 2015); never decreasing; no gap over 0.5 s; streams cover the recording to within 1 s; start / end times match the data to within 1 s; head pose and quality ≥ 30 Hz. |
| calibration | Every camera has a valid `-info` with a known lens model and its coefficient count; `-info-extra` values valid; a camera's video and calibration name the same frame; every named frame reaches `world`; a head-worn rig's main camera has a `/tf_static` path to `head`; `/tf_static` rotations unit and translations finite. |
| imu | ≥ 200 Hz; mean \|accel\| in the stillest 1 s within 0.5 m/s² of 9.81 (gravity not removed); noise values > 0. |
| poses | No NaN / Inf, unit rotations (head, wrist, `/tf_static`); hand positions match `hand_joints`; confidences in [0, 1]; no all-zero hands marked valid; gaze finite when valid; `ee_pose` a rigid 4×4; joint arrays match `joint_names`; grippers in [0, 1]. |
| tasks & labels | Spans end after they start (labels may be points); inside the recorded data; each label inside one task; `annotation_type` known. |

The 30 fps, 200 Hz, 30 Hz and 9.81 m/s² thresholds are fixed. The others
(5 % rate tolerance, 0.5 s gap, 1 s span tolerance, 0.5 m/s² gravity tolerance, 1e-3
unit tolerance) are **provisional, to be tuned on real deliveries**; all live in
`spec.py`. Structural video checks (non-empty, frame_id, format, holds a picture) run on
every message; full decoding runs on the first 60 messages of each camera. The converter
then decodes all of it and rejects an episode where any message is not exactly one frame.

## 10. Where each Lance field comes from

| Lance | MCAP |
|---|---|
| `recordings.recording_id` | `recording_id(lab, vendor_recording_id)` |
| `recordings.vendor_recording_id` | `vendor_recording_id` |
| `recordings.lab`, `operator`, `embodiment`, `rig_name`, `scene`, `license` | episode info |
| `recordings.captured_at` | `start_time_unix` |
| `recordings.num_frames`, `fps` | Decoded steps-clock frames and their timestamps |
| `recordings.video_path` | Assigned when each camera topic is extracted |
| `recordings.intrinsics` / `extrinsics` | Each camera's `-info` (+ `-info-extra`) / `/tf_static` and the moving links |
| `tasks.*` | `/task` messages, or the episode-info task keys over the whole recording |
| `annotations.*` | `/subtask-annotation` (type, text, source, span) |
| `steps.timestamp_ns` | Steps-clock frame times |
| `steps.head_pose`, `head_pose_valid` | `/head-pose` (× the camera's link), `/head-pose-quality`, timing gaps |
| `steps.{side}.keypoints`, `ee_pose` | `HandKeypoints.positions`, through the MANO fit when `hand_joints` is not canonical MANO |
| `steps.{side}.wrist_pose`, `confidence`, `{side}_valid` | `HandKeypoints.wrist_pose`, `confidence`, `valid` |
| `steps.eye_gaze` | `/eye-gaze` yaw, pitch, depth |
| Eva steps `obs_*` / `cmd_*` | `/{side}-arm-state` / `-arm-action` (`position`, `ee_pose`), `/{side}-ee-state` / `-ee-action` |

## 11. Not in 0.1

Deliberately left out of this version, to revisit with real deliveries:

- Graded SLAM quality (Aria MPS gives 0 / 0.5 / 1), beyond the `TrackingQuality` states.
- More than one IMU per device.
- Video from devices that store per-frame images: re-encode (as now) or accept them.
- Dexterous hands, tactile and auxiliary joints (Sharpa).
