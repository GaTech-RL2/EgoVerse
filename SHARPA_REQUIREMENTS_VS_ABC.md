# Sharpa requirements: what the ABC-130K ingest shows we can relax

*Status: input for a decision. Date: 2026-10-09.*

Sharpa's public LeRobot data cannot reach `data_status: complete` today (see
[`SHARPA_PUBLIC_DATA_AUDIT.md`](SHARPA_PUBLIC_DATA_AUDIT.md)). This note compares it with the
other public robot dataset EgoVerse ingested without vendor preprocessing, ABC-130K. The aim is
to decide which requirements can be relaxed for Sharpa and which must stay.

## Short answer

- **Intrinsics: yes, ABC shipped them.** Each camera has a `foxglove.CameraCalibration` message
  with resolution, `K` and a distortion model, wrist cameras included, "when intrinsics are
  available".
  - The EgoVerse converter kept only the front-camera `K` and dropped distortion.
  - On a large minority of episodes the calibration contradicts itself: `width` says 640 while
    `K` belongs to the 848-wide capture. EgoVerse corrects this heuristically. The error was
    found by projecting ABC's own end-effector poses into the image.
- **Extrinsics: no, neither static nor per-frame.**
  - ABC's top camera is fixed to the station, and its wrist cameras have no extrinsics at all.
  - EgoVerse derived one static top-camera transform from the published i2rt station model,
    plus a 4.3 cm correction fitted by hand to two marked gripper positions (residual
    ~18 px → ~4 px). It covers RealSense D405 stations only; ZED-X episodes have none.
  - ABC training used `eef_frame` cartesian and joint-space modes. Neither reads extrinsics.
- **ABC never met the v3.1 contract.** It trains because an episode with no `data_status`
  defaults to `complete` (`egomimic/rldb/zarr/episode_attrs.py`). Its first ingest wrote no
  camera metadata at all.

**Sources.** The ingest code is on unmerged branches, not on main:
- `origin/aniketh/abc`: `egomimic/scripts/abc_process/abc_to_zarr.py`,
  `backfill_abc_metadata.py`, `egomimic/rldb/embodiment/yam.py`.
- `origin/aidan/abc-sim-pipeline`, `origin/aidan/abc-stationery-pi`.
- ABC's own `docs/YAM_DATA_FORMAT.md` and README (`XDOF/ABC-130k` on Hugging Face).

Whether the metadata backfill was ever applied to the R2 copies is **unknown**. Aidan's
2026-10 data config says those zarrs still read `eva_bimanual`, although SQL was relabelled to
`yam_bimanual` on 2026-09-01.

## How ABC was ingested

- **Source.** MCAP per episode: 130,703 episodes, 3,590.7 h, Apache-2.0. Two station types:
  RealSense (mono top camera, 640×480) and ZED-X (stereo top, 1920×1200).
- **Clock.** Every message carries absolute Unix nanoseconds, and each stream has its own clock.
  The converter uses the front-camera frame times as the episode clock and matches joints and
  grippers to them by nearest neighbour, as ABC's doc prescribes.
- **End-effector pose.** Vendor-supplied: the controller's 4×4 pose in a station-fixed world
  frame. ABC also publishes a MuJoCo model for FK. Some episodes ship an empty pose; the
  converter skips them rather than computing FK.
- **Metadata.** Task from `/instruction`, `operator-id`, and episode start and end times.
  Subtask annotations cover 42,980 episodes (33%).
- **Identity.** The first ingest labelled the station `eva_bimanual`. A later `yam_bimanual`
  embodiment and registry entry (`yam_x6`, "no model or calibration constants are inferred
  from another platform") followed. Aidan's branch then maps YAM back to EVA for training.
- **Scale in EgoVerse.** About 18.5k episodes were in R2 when the backfill ran, per its code
  comments.

## Comparison

"Consumed by" says what breaks without the item:
- **training**: the dataloader or the labels need it;
- **QA**: overlays and visual checks;
- **bookkeeping**: naming, SQL, provenance.

| Requirement | Consumed by | ABC-130K | Sharpa public data | Recommendation |
|---|---|---|---|---|
| Front-camera intrinsics (K, distortion) | QA; camera-frame training modes | Vendor, per episode. Inconsistent on a minority, corrected by us | None | **Ask Sharpa; don't block on it alone.** Needed to project anything, so needed for the independent hand-pose check below |
| Static camera extrinsics | camera-frame training; QA | None. Derived from the published station model, D405 only | Not applicable: no Sharpa camera is static | — |
| Per-frame camera pose (`obs_head_pose`) | **training** for Dexmate (head-frame keypoints, proprio); QA | Not needed: fixed camera, `eef_frame` modes | FK gives `head_base_link`; the camera mount is unknown | **Keep, but accept a model- or CAD-derived mount with provenance** (ABC precedent). Not a mount fitted to our FK (see Cautions) |
| Wrist-camera calibration | QA only | Intrinsics yes, extrinsics no | None | **Relax.** The contract already makes it optional |
| Capture clock (int64 UTC) | bookkeeping: episode ID, annotation alignment, cross-stream sync | Absolute ns per stream | Synthetic: `frame_index/30` | **Relax with provenance** (`timestamp_source`, already written by the converter). Keep asking for the real clock |
| End-effector / palm pose | **training** input | Vendor-measured; some episodes empty | Our FK; Sharpa's TCP disagrees (no fit better than 57 mm) | **Keep. Must be resolved with Sharpa** |
| Hand joints within registry limits | integrity; drives FK labels | Not applicable (parallel jaw) | 94% of episodes fail | **Keep. Must be resolved with Sharpa** |
| Hand keypoints (MANO-21) | **training** labels (Dexmate predicts future keypoints) | Not applicable | Our FK | **Keep.** Only as good as the two rows above |
| Annotations | language conditioning; QA | Subtasks on 33% | None | **Relax** (already agreed) |
| Task name | training filters and prompts | Per episode, from `/instruction` | Placeholder `north ces task` | **Relax** to a dataset-level name given at conversion |
| Operator, scene | SQL bookkeeping | `operator-id` yes, scene no | None | **Relax** |
| Tactile | optional array | None | Present, units undeclared | **Relax:** omit until Sharpa declares units |
| Embodiment and platform identity | resolver; training | Mislabelled at ingest, relabelled later | `north_ces` vs. our `dexmate_*` keys | **Keep.** Confirm with Sharpa |
| `data_status`, `schema_version` | resolver gate | Absent, so it defaults to `complete` | Converter writes `structural_sample` | **Decision needed** (next section) |

## The asymmetry to decide

The two datasets are treated differently:
- **ABC:** an unlabelled legacy ingest, with less camera metadata than the contract asks for,
  trains.
- **Sharpa:** an honest conversion is labelled `structural_sample`, and the resolver and staging
  refuse it.

`egomimic/rldb/zarr/test_legacy_compatibility.py` shows the split. Legacy episodes validate with
a warning, and fail once `calibration_present` is required.

Options, none applied:

1. **Provenance plus a policy.** Record `timestamp_source` and a new `calibration_source`
   (`vendor | model | fitted | none`), and let a lab policy decide which sources may still be
   `complete`. This keeps labels honest and makes relaxations explicit and queryable.
   **Recommended.**
2. **Per-lab waiver** in the validator's status-eligibility check (for example, `lab=sharpa`
   waives the clock and wrist calibration). Simpler, but the reasons stay outside the episode.
3. **Opt-in training flag** that admits `structural_sample` (for example, on the resolver). Good
   for experiments now, but it bypasses the gate rather than refining it.

Relaxing these requirements alone recovers little:
- **Joint limits still block 94% of episodes.** The limit check is an integrity error whatever
  the status, so at most the 311 in-limit episodes pass validation until the conflict is
  resolved.
- **Dexmate training still needs `obs_head_pose`.** `Dexmate` supports only head-frame keypoint
  modes, so it cannot load episodes without a per-frame camera pose.
- **A sidestep exists.** A joint-space Dexmate mode, like ABC's joints mode, would need neither
  camera geometry nor FK labels. It does not exist yet.

## Cautions

- **A fitted camera cannot check FK.**
  - `sharpa_preview.fit_camera` fits a head mount and focal lengths to image picks of our
    FK keypoints, so it absorbs any FK error.
  - ABC's correction was fitted against the vendor's own end-effector poses, so it was not
    circular.
  - For Sharpa, an overlay is the cheapest independent test of the hand-pose problem. But it
    only tests anything if the camera parameters come from Sharpa or from CAD, not from a fit.
- **`head_base_link` is not the optical camera.**
  - Its FK pose works as a head frame for Sharpa-only training.
  - It is offset from the optical frame used by human data, which matters when cotraining.
  - If used, it needs its own key or an explicit attribute, never `obs_head_pose`.

## For Sharpa regardless: joint limits and hand pose

Both point to a model mismatch: the URDFs we registered may not be the ones the recording rigs
ran. What we registered:
- **Hands:** `{left,right}_sharpa_wave_with_flange.urdf`, Sharpa Wave **V3.0.4** from
  `sharpa-urdf-usd-xml`.
- **Platform:** `NTH_POC1_0_URDF_250905v4.urdf`, supplied separately.
- SHA-256 values are in the registry.

The data spans 2026-05 to 2026-09 on five rigs (POC22002, POC22018, POC22032, POC22048,
POC22061).

**Evidence.**
- **Joint limits.** 94% of episodes leave the Wave V3.0.4 limits.
  - Commands clamp DIP joints at exactly π/2, against a URDF limit of 1.3963 rad.
  - Observed DIP reaches 1.617 rad, above even π/2. So a revised limit alone cannot explain the
    data; zero offsets or a different hand model must also be involved.
  - Pinky CMC reaches 0.49 rad (limit 0.2618) and thumb MCP_AA ±0.415 (limit ±0.349).
- **Hand pose.** `observation.state.tcp` rotation follows our FK of the same arm's distal link
  (correlation 0.95), but its position matches no FK link under any rigid, offset or scaled
  fit (best 57 mm RMS), and it moves 3–10× more than our palm.

**Ask Sharpa:**
1. Which hand and platform URDF (name and revision) the controller loaded on each rig, by date.
   Send those files if they differ from ours.
2. Encoder zero and sign conventions for the hand and arm joints, and the arm joint order in
   `observation.state[0:7]` / `[29:36]`.
3. Whether firmware or teleop clamps DIP commands at π/2, and the true hardware limits.
4. What link and frame `tcp` reports, with a few reference pairs: a joint vector and the
   controller's TCP pose for it.
5. Head-camera intrinsics and the head-camera mount relative to `head_base_link`, from
   calibration or CAD. These let an overlay check the hand poses independently.
