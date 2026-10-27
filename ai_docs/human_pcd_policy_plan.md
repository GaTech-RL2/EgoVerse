# Human-data 3D policies (DP3 / Adapt3R) — feasibility + build plan

**Goal (user request, 2026-08-04):** train DP3 on HUMAN data with the point cloud
transformed to the ROBOT's view, using SEW-Geometric-Teleop's robot-placement design and
the functional-retargeting action labels. Secondary: assess Adapt3R on human data.

**Verdict: feasible — all ingredients exist on disk. Nothing is blocked on new capture.**

---

## Inventory (verified 2026-08-04)

| ingredient | where | status |
|---|---|---|
| Retargeted 49-D actions (`actions/joint`, functional retargeting) | `full_pick_and_place_hdf5/glove_nav_pick_and_place_trimmed.hdf5` (30 demos) + siblings | ✅ |
| **SEW robot placement per frame** | `obs/robot_base_pose` (x, y, yaw) in the same HDF5 | ✅ the key to "where the robot is" |
| Retargeted robot joints (torso+head for FK) | `obs/robot0_joint_pos` (26) | ✅ |
| Human depth (Fast-FS) | `aria_fs_out/fastfs_fpp_depth_npy` — 146 recs, 59 glove | ⚠ **~10 s window per rec @7.5 Hz** (not full episodes) |
| Human VRS (calib + VIO + SLAM streams) | `aria_fs_out/fpp_all_vrs` | ✅ one device, same as the 176 known VRS |
| Their exact rect convention | `aria_gen2_scripts/src/utils/aria2_utils.py::create_scanline_rectified_cameras` + `teleop_common.rect_params` | ✅ reproducible, no guessing this time |
| SEW placement code | `SEW-Geometric-Teleop` + `aria_gen2_scripts/src/place_teleop_robot.py` | ✅ |

**Depth-window content check:** windows trend median 2.1 m → 1.6 m with ~48% of points
inside 1.5 m by the end → the window covers **approach + at-table manipulation**, which is
the phase that matters (teleop policies are manip-only anyway).

## The two technical problems

### 1. Clock anchoring (depth ↔ HDF5 rows)
Depth npy names are DEVICE time (ns since boot); HDF5 `aria_ts` is host wall clock. No
stored mapping. **Chosen solution: trajectory correlation** — `obs/robot_base_pose` xy is
the (scaled) human pelvis path; the VRS VIO xy is the head path over the same walk. 1-D
time-correlation of speed profiles gives the clock offset per rec, with a sharp peak =
built-in validation. (Extraction job `hvio` produces per-rec VIO paths for this.)

### 2. Human→robot view transform (the core of the request)
Exact version (target): per frame,
`T_robotglass_humanglass = inv(FK_head(robot_base_pose, robot0_joint_pos)) @ T_world_humanhead`
with `T_world_humanhead` from VIO (extracted), FK via the SEW RBY1 model (`yourdfpy` in
their env). This is exactly "put the camera where SEW put the robot's head".
v0 fallback if FK integration drags: SKR-style scale-about-pelvis approximation
(`P' = pelvis + (P − pelvis)/s`), with `s` FIT from data by matching the camera→table
distance distribution to the TELEOP clouds (which are the ground-truth robot view), and
pelvis at an anthropometric constant offset in the glass frame. Validation gate either
way: transformed human table-plane distance distribution must overlap the teleop one.

## Dataset yield estimate
30 glove demos × ~10 s × 10 Hz (nearest-depth ≤66 ms stale) ≈ **3,000 frames**; extending
to all 146 fpp recs (other HDF5s: `fullpp_train_v1.hdf5` etc.) ≈ up to ~11k depth frames.
Teleop reference: 7,003.

## Adapt3R on human-only: possible, one extra step
Adapt3R needs `image_rect` pixel-aligned to depth. The human npy stores don't ship it, BUT
the rect convention is now reproducible from their own code + the VRS SLAM stream, so I
can generate `image_rect` myself with zero convention risk. Same clock anchoring applies.
→ Do DP3 first (no image needed); Adapt3R-human is a follow-on using the same aligned rows.

## Execution order
1. ✅ `hvio` job (running): per-depth-frame VIO pose + RGB device-ts + 10 Hz VIO path, all glove recs.
2. Clock anchor per rec (correlation) + QC report (peak sharpness threshold).
3. Transform: FK-based if straightforward, else validated-v0; **gate = teleop-distribution match**.
4. Build LeRobot dataset (SG actions, deterministic sampling, range crop 0.25–1.5 m — all
   the audited fixes) → verify suite → train DP3-human.
5. Adapt3R-human: generate image_rect for the same rows → slam-rect-style dataset → train.

## REAL Adapt3R: TCP (hand-centered) frame — design (added 2026-08-04)

User direction: implement Adapt3R's actual hand-centered frame at the **EEF TCP — the
alignment point of SEW functional retargeting**. This is the piece our port never had,
and it is what makes human and robot clouds share ONE representation: functional
retargeting matches human hand TCP to robot TCP by construction, so a TCP-centered cloud
is embodiment-invariant *by the same construction*.

### Verified ingredients
| piece | source | status |
|---|---|---|
| Human wrist pose **in device frame** | VRS handtracking stream `371-1`: `transform_device_wrist` (SE3), confidence 0.9997, palm + landmarks | ✅ **native** — no solve needed |
| Human rect→device | their `create_scanline_rectified_cameras` (pure rotation) | ✅ reproducible |
| Teleop TCP in **base** frame | `obs/eef_right_proprio` = 4×4 (verified orthonormal, z≈1.2 m) | ✅ |
| Teleop base→head-link FK | `robot0_joint_pos` (torso 6 + head 2) + RBY1 MJCF `SEW-.../rby1_v1.3_xhand.xml`, mujoco 3.10 in emimic | ✅ |
| **Teleop head-link→glass (Aria mount)** | UNKNOWN (~20° per depth-agent note) | ❌ the one solve needed |

### Mount-extrinsic solve (teleop, one constant)
`T_glass_tcp(t) = inv(T_mount) @ inv(T_base_headlink(t)) @ T_base_tcp(t)`; everything known
except constant `T_mount`. Solve: detect the right-gripper cluster in our own clouds on
high-confidence frames (nearest large cluster, RANSAC over frames), Kabsch between the
FK-predicted TCP trajectory and the observed cluster trajectory. Torso motion is handled
by the FK term, so the solve is exact, not subset-approximate. Validation: after the
solve, the observed gripper cluster must sit at the origin of the TCP frame across ALL
frames (median offset < ~5 cm).
Note: the robot-side VRS capture (delegated to the deployment agent) would give T_mount
directly and supersede this solve — the data-driven estimate is the bridge until then.

### Encoder plumbing
`lift_point_cloud_batch` already takes per-sample `E`; the fixed-buffer limitation is at
the encoder wrapper. Plan: store `T_rect_tcp` per frame in the dataset (16-D metadata
key, like `aria_depth`), route via a `pose_key_map` mirroring `depth_key_map`, encoder
uses it as per-sample extrinsics → cloud lifted directly into TCP frame. Optional
Adapt3R-style local crop: keep points within R of the TCP (config), replacing the fixed
range crop for this variant.

### Bimanual note
v1 uses the RIGHT TCP frame (task is right-hand dominant; left arm near-static in the
demos). Two-token (per-hand) variant is the follow-up if v1 works.

## Open items
- SKR height-scale constant: not found by grep in `convert_human_csv_to_robot_hdf5_skr.py`;
  need to read the solver (`rby1_with_xhand_sew_solver.py`) — only needed for the v0 path.
- Non-glove fpp recs live in other HDF5s — confirm which have `robot_base_pose` + trimmed QC.
- Depth is a 10 s window: fine for manip-phase policies; if we ever want full-episode
  (nav+manip) human 3D, ask the depth agent to densify (their pipeline, ~GPU-hours).

## Colour image_rect: black-hole QC (2026-08-04, jobs 3652578/3651665)

The colour rect images carry black regions. Measured on the built dataset (810 frames):
median/mean 4.1% of pixels, p90 4.2%, max 7.5%. Location: rows ≥207/224 (bottom arc)
plus sporadic far-corner patches; rows 0–168 average 0.0% — hands/table/objects never
affected. Mechanism (verified by warping a white frame): out-of-bounds projection 0.00%;
**beyond the RGB fisheye image circle 4.34%** — the SLAM camera physically sees further
down than the RGB camera. Constant across all frames (same mount), identical in both A/B
arms, so the comparison is unaffected. Secondary coupling: colour is transported via
depth, so invalid-depth pixels (rare, far dark corners) are also colour-black.
Possible fix if ever needed: fill black with the grey-rect luminance (same rays, data
already in the depth store). Not applied — not worth a retrain at <5% marginal pixels.
