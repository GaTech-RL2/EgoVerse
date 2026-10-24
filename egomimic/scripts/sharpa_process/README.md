# SharpaDex → Zarr: `sharpa_to_zarr`

Converts the SharpaDex LeRobot release (`Sharpa-Robotics/SharpaDex-v1.0`) into
`dexmate_bimanual` Zarr episodes: a Dexmate NTH POC1.0 robot with a Sharpa Wave
hand on each arm. Each episode has per-side arm and hand joints, 30-D tactile,
tactile images, MANO-21 hand keypoints and palm poses, plus a head-link pose.

- **Licence:** CC-BY-NC-4.0 (`license: cc-by-nc-4.0` in the release README).
  It is stored in every episode's `license` attribute. The leads approved
  ingestion on 2026-10-24.
- **Embodiment:** `dexmate_bimanual` (`EMBODIMENT.DEXMATE_BIMANUAL = 8`). The
  array rules are in `egomimic/rldb/zarr/dexmate_rules.py`.
- **Lab:** `sharpa_dex`. It must equal the `processed_v3/<lab>/` folder the
  episodes are uploaded to.

---

## ⚙️ Before running

Use the locked environment (see `AGENTS.md`). Conversion is CPU-only, so on a
GPU node hide the GPUs:

```bash
uv sync --locked
export CUDA_VISIBLE_DEVICES=
```

You need two inputs:

- **The release**, laid out as `<release_root>/<task>/<season>/lerobot_v2.1/{data,meta,videos}`.
  The `lerobot_v3.0/` export beside it is not read.
- **A manifest**, one JSON line per release episode, after a header line
  (`sharpadex_records.read_manifest`). It decides which episodes are kept and
  gives each kept episode its UTC start and episode hash. The converter never
  re-plans names.

---

## 🚀 Converting

```bash
# Two episodes of one task.
uv run python -m egomimic.scripts.sharpa_process.sharpa_to_zarr \
  --dataset-root /data/SharpaDex-v1.0 --manifest manifest_v1.jsonl \
  --tasks iron_garment --episodes 0-1 --include-head-right \
  --output-dir converted/

# Exactly the listed episodes (source ids, one per line), resuming a killed run.
uv run python -m egomimic.scripts.sharpa_process.sharpa_to_zarr \
  --dataset-root /data/SharpaDex-v1.0 --manifest manifest_v1.jsonl \
  --recording-ids ids.txt --include-head-right --skip-done --overwrite \
  --output-dir converted/
```

| Option | Effect |
|---|---|
| `--tasks GLOB`, `--seasons GLOB`, `--episodes 0,3,5-7` | Select task folders, seasons and episode indices (default: all). Episodes the manifest does not keep are skipped. |
| `--recording-ids FILE` | Convert exactly these source ids. Blank and `#` lines are ignored. Every id must be a kept manifest episode. Only the listed seasons are opened. |
| `--include-head-right` | Also store `head_right` as `images.head_right`. |
| `--skip-done` | Skip an episode whose `<hash>.conversion.json` names it. |
| `--overwrite` | Replace an existing output of the *same* episode. |
| `--license NAME` | Override the licence read from the README. |
| `--jpeg-quality 90`, `--chunk-timesteps 100` | Image and storage settings. |

Exit status: 0 if everything converted and verified, 1 if an episode failed to
convert, 2 if a verification report has errors.

FFmpeg threads are capped at the CPUs the process may use, divided by the
streams decoded at once. Pin a run with `taskset` or Slurm's `--cpus-per-task`.

---

## 📦 Outputs

For each kept episode `--output-dir` gets:

| File | Contents |
|---|---|
| `<episode_hash>.zarr` | The episode. The hash is the manifest's UTC start (`YYYY-MM-DD-HH-MM-SS-ffffff`, millisecond resolution). |
| `<episode_hash>.mp4` | H.264 preview of `images.front_1` (`CONTRIBUTING_DATA.md`, the merge script derives its path). |
| `<episode_hash>.validation.json` | The verification report (below). |
| `<episode_hash>.conversion.json` | The sidecar: ids, frame count, `data_status`, a validation summary and every attribute. It is written last, so its presence marks a finished episode. |

A failed episode leaves only `.<hash>.*.partial` files, which are removed.

### Episode IDs

There are two IDs, and the converter keeps both:

- **Source ID** `<task>/<season>/<episode_index>`. This is the manifest's
  `recording_id`. It is used only to select and locate the source episode, and
  is stored as `source_episode_id`.
- **Canonical ID** `<task>__<season>__<episode_index:06d>`, from
  `sharpadex_records.vendor_recording_id`. It is the episode's
  `vendor_recording_id`: one path component, as `CONTRIBUTING_DATA.md` §6.6
  requires.

A task or season containing `/` or `__`, or ending in `_`, is rejected, so every canonical ID
parses back to exactly one source ID. `read_manifest` checks this over the
whole manifest. `manifest_v1.jsonl` (28,993 records) passes.

The zarr attributes and the sidecar store both IDs. Resume and ownership checks
compare the canonical ID, and they refuse an output whose `source_episode_id`
does not correspond to it. The converter never replaces a store or sidecar that
holds another episode, even with `--overwrite`.

### Required metadata

| Field | Value |
|---|---|
| `task_name` | The task folder, e.g. `iron_garment`. |
| `task_description` | Sharpa's `task_instruction` tag. |
| `operator` | `sharpa`, the `OPERATOR` constant. |
| `lab` | `sharpa_dex`. |
| `scene`, `objects` | `null` (`MissingMetadata()`): SharpaDex has neither field. |
| `intrinsics` | `null` (`MissingMetadata()`): SharpaDex ships no camera calibration. Only `dexmate_bimanual` may store this. |

**`operator` is a surrogate.** SharpaDex does not say who teleoperated an
episode. `sharpa` names the vendor, not a person. When Sharpa supplies operator
IDs, replace it with an SQL update over `lab = 'sharpa_dex'`.

### Arrays

Everything is in `robot_base`, the root of `kinematics/NTH_POC1_0_URDF_250905v4.urdf`.
Joint values are stored as delivered (float32, radians). Poses and keypoints are
float64.

| Source | Array | Contents |
|---|---|---|
| `observation.state[0:7]`, `[29:36]` | `{side}.obs_joints` (T, 7) | `<side>_arm_joint_1..7` |
| `observation.state[7:29]`, `[36:58]` | `{side}.obs_hand_joints` (T, 22) | `dexmate_rules.HAND_JOINT_NAMES[side]` order |
| `observation.state[58:65]` | `obs_aux_joints` (T, 7) | `lower_body_joint_1..5`, `neck_joint_1..2` |
| `action`, same columns | `{side}.cmd_joints`, `{side}.cmd_hand_joints`, `cmd_aux_joints` | Absolute joint position setpoints |
| FK of state / action | `{side}.obs_ee_pose`, `{side}.cmd_ee_pose` (T, 7) | Palm (`<side>_hand_C_MC`) pose, xyz + wxyz |
| FK of state / action | `{side}.obs_hand_keypoints`, `{side}.cmd_hand_keypoints` (T, 63) | MANO-21 keypoints; slot → link map in `kinematics/sharpa_dexmate.yaml` |
| FK of state | `obs_head_link_pose` (T, 7) | `head_base_link`. A link pose, not a camera pose, so it is not `obs_head_pose`. |
| `observation.tactile` | `{side}.obs_tactile` (T, 30) | Left 30, then right 30. Fingers run thumb to little, each `fx fy fz tx ty tz`. Units are N and N·m, inferred from value ranges; Sharpa has not confirmed them. |
| `tactile_deform` | `{side}.obs_tactile_image` | One 240×1200 RGB JPEG strip per hand: row 0 of the grayscale mosaic is left, row 1 is right, and each has 5 tiles, thumb to little. `tactile_raw` is not stored. |
| `head_left`, `wrist_left`, `wrist_right`, (`head_right`) | `images.front_1`, `images.left_wrist`, `images.right_wrist`, (`images.head_right`) | JPEG, 480×480 |
| Manifest start + frame index | `obs_rgb_timestamps_ns` (T,) int64 | UTC ns: the start plus `i * 10**9 // fps`. `timestamp_source` says whether the start came from the episode's recording name (one-second resolution) or was synthesized from the season name. |
| Vendor columns | `sharpa.joint_torque`, `sharpa.tcp`, `sharpa.obs_tcp_pose`, `sharpa.cmd_tcp_pose`, `sharpa.subtask_index` | Unchanged, outside the contract |
| `meta/annotations.jsonl` | `annotations`, `annotations_skill` | Subtask spans, made half-open (`end_idx = end_step + 1`). The ` \| Skill: …` suffix becomes a skill span. Per-hand text (`Left hand: … \| Right hand: …`, in 8 tasks) is joined with `; ` so neither hand is lost. |

Two converter-owned attributes describe the columns. The writer's `features`
names them `dim_0`.

- `sharpa_array_names` holds the joint names of every joint array, the
  taxel-channel names of `{side}.obs_tactile` and the finger of each tile.
- `sharpa_tactile_units` holds the unit of each tactile channel.

The provenance attributes are `source_*`, `rig_name`, `timestamp_source`,
`start_source`, `hash_offset_ms`, `quality_flags` and `conversion_report`.

---

## ✅ Verification

`<hash>.validation.json` (`format: sharpadex_validation_v1`) merges three
checks into one list of errors and warnings:

1. Main's validator, run with `--allow-missing`. It includes the
   `dexmate_rules` array checks (shapes, finite values, unit head-link
   quaternions, hand joint limits, one-frame spikes). Only `scene`, `objects`
   and `intrinsics` may be waived.
2. **FK residual.** On up to 64 frames, the stored observed and commanded
   keypoints must match FK of the stored hand joints within 5 mm, in the palm
   frame. The keypoints are FK outputs, so this check confirms joint order,
   units and frames. It cannot confirm that the URDF matches the robot.
3. **TCP side check.** Sharpa's `observation.state.tcp_pose` must match FK of
   each arm's `Link_7` under the fitted tool transform in
   `sharpa_tracks.TCP_MODEL`: within 2° median rotation and 5 cm p95
   position. A pairing with the other arm must fit worse, otherwise the arm
   columns look swapped.

The report also checks that `vendor_recording_id` and `source_episode_id`
correspond, and that the conversion is complete: `data_status` is `complete`
and the source, retained and stored frame counts are equal. Episodes are
always converted whole. `ok` is `not errors`. `sharpa_to_zarr.read_report` is the one
reader. It counts a missing, unreadable or malformed report as a failure and
recomputes `ok` from the errors.

You can also run main's validator by hand:

```bash
uv run python -m egomimic.rldb.zarr.validate converted/*.zarr --allow-missing
```

### `data_status`

`conversion_report.data_status` is `complete`, or `structural_sample` when the
stored data has a defect:

- no `head_left` video;
- an observed hand joint more than 0.26 rad past its URDF limit;
- a failed TCP check.

Gaps in what the delivery provides go to `data_limitations` instead: no
calibration, no head camera pose, a synthesized clock, or joint excursions
inside the validator's margin. A `structural_sample` fails its report, so only
`complete` episodes are ingested.

Smaller hand-joint excursions (past 0.035 rad) are validator warnings. Sharpa's
firmware stops sit inside the URDF limits, and teleoperation commands overshoot
them. `conversion_report.joint_limits.source_excursions` lists every
excursion, with the verdict main's validator gives it.

---

## 🦾 Kinematic models

`kinematics/` ships as package data:

- the platform URDF, and the left and right hand URDFs (Wave V3.0.4 from
  `sharpa-urdf-usd-xml`), byte for byte as delivered;
- the hand package's `LICENSE.txt` and `NOTICE.txt`;
- `sharpa_dexmate.yaml`, which records each URDF's sha256, the head link, the
  hand mount, the aux joint order, all 21 slot → link maps and the tactile
  layout.

`load_chain` refuses a URDF whose bytes differ from the recorded sha256. Only
the kinematic tree is read, so no meshes are needed. Training reads only the
stored arrays and never loads these models.
