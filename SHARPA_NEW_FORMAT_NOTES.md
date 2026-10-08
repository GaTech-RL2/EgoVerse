# Sharpa public data and the proposed EgoVerse format

*Status: input for the format discussion. Date: 2026-10-08.*

This note extends `SHARPA_FORMAT_PROPOSAL_ALIGNMENT.md` (30 Sept), a design note kept outside
this branch, which numbers the open points B1–B5. That comparison relied on a single 370-frame
structural sample. Since then Sharpa has released two public LeRobot datasets, summarised in
[`SHARPA_PUBLIC_DATA_AUDIT.md`](SHARPA_PUBLIC_DATA_AUDIT.md):

| Dataset | Seasons | Episodes | Hours | Downloadable (v2.1) |
|---|---|---|---|---|
| RoCo Task Board Assembly | 162 | 3,252 | 107.1 | all |
| Robotic Origami Challenge | 143 | 1,991 | 109.1 | 1,827 episodes / 100.4 h |

The counts come from the v2.1 metadata. Fourteen Origami seasons have metadata but incomplete
data on the Hub (audit T3).

## 1. The new format has not been merged, so the branch stays on Zarr v3.1

The proposal in `egoverse_dataset_proposal.md` (Lance tables plus MP4 GOP2; also kept outside this
branch) does not exist in code:

- `origin/main` (`42817f5dd`, 2026-09-29) does not contain `lance`, `pylance`, `recording_id`,
  `task_instance_id`, or `uuid5`.
- Neither do the 40 most recently updated remote branches, checked after refreshing them on 2026-10-08.

This branch therefore makes no Lance migration. The Sharpa converter writes Zarr v3.1, but it is
built to make a later migration cheap:

- `egomimic/scripts/sharpa_process/lerobot_reader.py` reads LeRobot v2.1 and v3.0 and does not
  depend on Zarr.
- `sharpa_tracks.py` derives the canonical arrays and runs batched forward kinematics (FK).
- Only `sharpa_to_zarr.py` writes Zarr.

A Lance writer can replace that last step. It would use the same `vendor_recording_id`
(`<dataset>/<season>/<episode_index>`) and the same `rig_name` (POC id) that the Zarr
attributes record.

## 2. Points to re-raise, with the new evidence

### B1 — Separate robot and hand identity

- **Evidence.**
  - Every season reports one string, `robot_type: north_ces`.
  - The 65 joints use generic names (`left_hand_j0..21`, `motor_j0..6`).
  - No field identifies the hand model, its revision, or the platform revision.
  - Data comes from five rigs: POC22002, POC22018, POC22032, POC22048 and POC22061.
  - In 4,932 of 5,243 episodes (94%), hand joints exceed the published Sharpa Wave URDF limits,
    so the validator rejects them. Commanded DIP joints sit at exactly π/2 against a limit of
    1.3963 rad; observed DIP reaches 1.617 rad and pinky CMC 0.49 rad (limit 0.2618). The
    failure rate varies by rig, from 91–94% on POC22061 to 100% on the other four. This suggests
    the installed hand, or its encoder zeros or firmware limits, differs from the published model.
  - The vendor TCP pose (populated on some rigs) cannot be reconciled with FK through our
    platform URDF: no rigid, offset or scaled fit beats 57 mm RMS. So the arm chain or joint
    convention used by Sharpa's controller may also differ from the URDF we registered.
- **Why it matters.**
  - The registry splits hardware into `dexmate_nth_poc1` (platform) and `sharpa_wave_v1_*`
    (hands). The proposal's `(lab, embodiment)` Steps key cannot record that the hand changed
    on the same platform.
  - With a limit conflict like this one, you need to know which recordings used which hand
    revision, and which URDF revision the FK-derived columns came from.
- **Re-raise.**
  - Add `platform`, `end_effector_left`, `end_effector_right` and `rig_name` to Recordings,
    all as registry IDs.
  - Generate the Steps schema from the registry.
  - Record the registry commit in each release tag.
  - Record whether each pose or keypoint column was measured by the vendor or derived by FK,
    and from which URDF (path and SHA-256). For Sharpa today every pose is FK-derived and
    unverified.
- **Naming.**
  - `north_ces` (NTH POC1.0) still conflicts with the `dexmate_*` registry keys.
  - Settle the name before IDs and table names lock it in.

### B2 — Per-step camera extrinsics for every moving camera

- **Evidence.** All four RGB cameras move relative to `root_link`:
  - The **stereo head pair** (`head_left` and `head_right`, 480×480) sits on a 2-DOF neck above a
    5-DOF lower body.
  - The **wrist cameras** ride on 7-DOF arms.
  - The neck moves in every episode (head pitch 4–60°), and lower-body joints 3–5 move, so
    even the arm bases are not static in `root_link`.
  - The URDF has a `head_base_link` but no camera links.
  - No season includes intrinsics, distortion or mount transforms. The wrist cameras are
    visibly fisheye, and the head pair shows a consistent 1–2.5 px vertical offset, so it is
    not exactly rectified.
- **Consequences.**
  - The proposal's static 4×4 `extrinsics` per camera cannot describe any of these cameras.
  - Today's Zarr contract can describe only one moving camera, through `front_1` and
    `obs_head_pose`.
  - Neither format can describe the second head camera or the wrist cameras.
- **How the converter handles it.**
  - The calibration input stores a static mount per camera, `{parent_link, parent_link_T_cam}`.
  - Each frame's optical pose is computed as `FK(parent_link)[t] @ parent_link_T_cam`.
  - Only `front_1` has an array to store that pose (`obs_head_pose`). The other mounts are kept as
    an attribute.
- **Re-raise.**
  - Store each camera in Recordings as `{model, K, distortion, resolution, rectified}` plus either
    `mount: {parent_link, parent_link_T_cam}` or, for fixed cameras, `ref_T_cam`.
  - Store a per-step optical pose column for **every** moving camera (`<camera>_pose`, float32[7],
    with `<camera>_pose_valid`) in Steps for any embodiment, not only in the human `head_pose`.
  - Treat the stereo pair as two cameras; their baseline is implied by the two mounts.
  - Keep the definition "optical pose, +X right, +Y down, +Z forward, in `reference_frame`".

### B3 — The recording clock

- **Evidence.**
  - In all 5,243 episodes, `timestamp` equals `frame_index / 30` as episode-relative float32.
  - There is no absolute clock.
  - The only wall-clock information is the season folder name, at 1 s resolution. Its start
    hours fall between 10:00 and 21:00, which looks like local time; an earlier sample matched
    UTC+8. Per-episode start offsets are unknown.
- **Consequences.**
  - The proposal puts task and annotation bounds on a nanosecond "recording clock". With synthetic
    timestamps those bounds are only frame indices × 33.3 ms. The converter does generate an
    integer clock, but it records `timestamp_source: synthesized_from_season_name_and_frame_index`
    and marks the episode `structural_sample`.
- **Re-raise.**
  - Define `timestamp_ns` as int64 UTC from the capture clock.
  - Add `recordings.clock_source` (`device | synthesized`) so synthetic clocks cannot pass as real.
  - Require per-camera frame times, or proof of hardware sync.

### B4 — What a robot recording is

- **Evidence.**
  - Median episode length is 1.75 min for RoCo (range 0.15–12.4) and 3.3 min for Origami
    (range 2.1–7.0), against EgoVerse's usual 1–2 min. 67% of RoCo episodes are at most
    2 min; no Origami episode is.
  - Each season groups the episodes recorded on one rig in one session. LeRobot's `index`
    runs on across episodes, but that is an export property, not evidence of continuous
    recording.
  - **Episodes look like trimmed trials, not one continuous stream.** At both boundaries
    examined, commanded hand joints jump 12–30× a normal per-frame step (1.2–1.75 rad), while
    the arms restart near a common parked pose. Released episodes fill a median of 29% of the
    wall-clock time between same-day season starts (maximum 86%), so most session time was
    reset, idle, or discarded. This rests on two boundaries and needs Sharpa's confirmation.
- **Re-raise.**
  - Map `episode → recording`, with the season kept as a session or grouping attribute.
    `season → recording` would imply a continuous clock across gaps that were cut.
  - Require a real per-episode UTC start (B3) so that order and gaps within a session can be
    reconstructed.
  - Ask Sharpa to confirm what was cut, before more data is released.

### B5 — Per-column validity, not sentinels

- **Evidence.**
  - TCP pose and wrench (24-D) are **zero-filled** in all of Origami and in 43% (1,411 of
    3,252) of RoCo episodes, instead of being marked missing. TCP is populated only on rig
    POC22018 and on POC22061 from 2026-07-30; no season mixes the two.
  - All seven aux joint torques are exactly zero in every episode, again with no flag.
  - The v2.1 export of four POC22061 seasons (2026-05-23 to 05-25) has no `head_right`, no
    tactile video and no `observation.tactile`, although their v3.0 export has them.
    `modality.json` omits `head_right` and both tactile streams in every season.
  - Hand joint values outside the model limits appear in 94% of episodes.
- **Re-raise.**
  - Use a validity flag per column group (`<group>_valid`) and store NaN in invalid rows.
  - Make "column absent from the source" distinguishable from "value invalid in this frame".
  - Decide whether out-of-limit joints invalidate a frame, a column, or nothing. Today the
    validator rejects the whole episode.

### Also carry forward

- **Tactile.**
  - `observation.tactile` is 10 fingertips × (fx, fy, fz, tx, ty, tz).
  - Two tactile video streams exist (480×1200 deformation and 480×1600 raw).
  - The units are still undeclared.
  - Steps needs an optional tactile column group, declared from the registry (taxels, channels,
    units). The F/T layout answers the old question of how taxels map to fingertips.
- **Video.**
  - Most source MP4s are H.264. AV1 appears in 9 Origami seasons in v2.1 and 26 in v3.0; all
    RoCo v3.0 video is H.264, although the README says "primarily AV1". GOP varies (250 frames
    in the RoCo sample, 2 in the Origami sample).
  - The Zarr path re-encodes them to JPEG, which loses quality a second time.
  - In the new format they could be remuxed or transcoded straight to GOP2. Check that K matches
    the encoded resolution.
- **Raw retention.**
  - The public HF repos *are* the raw vendor source, under CC-BY-4.0. That licence is compatible
    with inclusion in a CC BY-SA 4.0 release, with attribution.
  - `vendor_recording_id` plus the HF revision SHA makes a reproducible pointer back to it.
- **Training.**
  - The Dexmate keymap (`egomimic/rldb/embodiment/dexmate.py`) requires `obs_head_pose` as
    proprio.
  - Uncalibrated conversions omit it rather than inventing a camera pose, so they cannot be loaded
    for training even with the status gate bypassed.
  - The fix is camera mounts from Sharpa, not a placeholder.
