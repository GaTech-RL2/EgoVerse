# real_mug rig: conventions + calibration (2026-10-07)

Raw demos: `/cluster/scratch/agavryushin/ben_icra/demos/*.h5` (Euler), copied to
`/iopsstor/scratch/cscs/jiaqchen/data/real_mug_30hz/raw`. Single Franka (FR3/Panda) + ORCA v1
right hand on Gavin's 90° mount, 30 Hz, fixed overhead front cam + palm wrist cam (432x240),
aligned depth. Images in the raw files are stored **BGR**.

## Verified conventions
- `fk_check.py`, `rot_check.py`, `fk_final.py` — Panda modified-DH FK vs `arm/ee_target`:
  controller TCP = flange + (0.13, 0, 0.07) m, identity rotation (matches the Franka GUI
  "Flange to End-Effector" entry; data fit gave (0.1325, -0.0049, 0.0735)). Quaternion order
  **xyzw**. Residual 7.8 mm / 2.5 deg = controller lag (~3 frames).
- `seam_check.py` — recorded tool frame puts the converter's ZYX yaw on the +-180 seam for 15% of
  frames; post-multiplying by Rx(180 deg) gives 0%. Applied in `raw_to_processed.py` to proprio
  and actions (`tool_frame_flip_xyzw` attr). Deployment: post-multiply predicted orientation by
  Rx(180 deg) to recover the recorded tool frame.
- Hand joints: 17-D in orca_core v1 `joint_ids` order (`thumb_mcp, thumb_abd, thumb_pip,
  thumb_dip, index_abd/mcp/pip, middle_*, ring_*, pinky_*, wrist`), radians; per-joint recorded
  ranges match the config ROMs.

## Calibration registry (`egomimic/utils/egomimicUtils.py`, `egomimic/utils/orca_hand_fk.py`)
- `INTRINSICS["mug_front"]` — from `camera/intrinsics/front` (identical across the 148 files
  that carry it; 2 files lack the key).
- `EXTRINSICS["mug_front"]` — the rig's own T_base_cam (cvg-dex-teleop calibration:
  translation + quat xyzw). Projects the TCP onto the finger pads in held-out episodes.
- `HAND_MOUNTS["mug_tcp"]` — exact T_tcp_hand from cvg-dex-teleop's combined URDF
  (`egomimic/assets/orcahand/fr3_mount_orcahand.urdf`, see `MOUNT_ASSEMBLY_NOTES.md`):
  `inv(T_flange_tcp) @ mount_joint @ mount_to_hand @ inv(T_world_tower)`. Wireframe verified on
  held-out episodes; data-driven fits (RGB-D hand masks) were 50 deg off and are not kept.

## Scripts
- `raw_to_processed.py` / `validate_processed.py` — raw recorder H5 -> processed FAIVE layout for
  `faive2lerobot/convert_faive_to_lerobot.py` (BGR->RGB, FK proprio, tool flip, cam renames).
  Copies live in the iopsstor `code/faive2lerobot` dir.
- `render_hand_overlay.py` — GT wireframe + TCP on held-out frames / one episode video
  (sanity check for the mount + extrinsics). Run on Euler (`~/claude_fk/` has fk_check.py,
  orca_hand_fk.py, orcahand_right.urdf).

## v2 recordings (bimanual, 2026-10-10 sanity_check, 20 episodes)
Raw layout changed: `arm/*/{left,right}`, `hand/*/{left,right}`, `camera/image/{top,left,right}`,
`camera/depth/top`. Both arms move (cup-on-saucer handover on a black mat). Still 30 Hz, BGR,
xyzw quaternions; FK+TCP vs `ee_target` 6-9 mm (left) / 11-18 mm (right).
- `raw_to_processed_v2.py` — bimanual adapter (`--arms left,right`): each arm in its own base
  frame (as recorded), top -> `aria_rgb_cam`, left/right wrist -> `oakd_{left,right}_wrist_view`.
- Extrinsics: new `T_base_cam` from Alexey = `EXTRINSICS["mug2_front"]["right_cam"]` (0.5 deg /
  6 mm from the v1 value); `left_cam = Trans(0, -0.7, 0) @ right_cam` because the left base is
  the right base translated +0.7 m along +Y (verified: both FK TCPs land on the hands).
- Left hand mount `HAND_MOUNTS["mug_tcp_left"]` = mirrored Gavin mount (MOUNT_ASSEMBLY_NOTES
  left section) + `orcahand_left.urdf`; it is the y-mirror of the right mount.
- `render_hand_overlay_v2.py` — both-hand wireframe check on the v2 demos.
- Training: `data/mug2_eve_local{,_full}`, `visualization/eve_cartesian_mug2`,
  `train_zarr_eve_mug2_bc_full` (bag BC recipe: eve_bimanual + wrist cams);
  scripts in `eth_clariden/mug2/`.
- `render_fr3_urdf_overlay.py` / `orca_hand_fk.FR3OrcaFK` — FK through the combined
  `fr3_mount_orcahand{,_left}.urdf` driven by the recorded 7 arm + 17 hand joints. Cross-check
  (2026-10-16): hand keypoints agree with the DH+TCP+HAND_MOUNTS path to 0.001 mm, link8+TCP vs
  `ee_target` 9.6 mm (same as DH). Keypoint 0 is `{side}_tower` (the stand-alone URDF's `world`
  root is 6 cm behind it). The viz still uses the DH/mount path because zarr only carries
  Cartesian EE poses, not arm joints.
- `render_mesh_overlay.py` — sim-asset overlay: FR3 + ORCA visual meshes from the combined
  URDFs, posed by the raw joints, rasterized through the top camera (per-arm extrinsics) and
  blended as a green silhouette. Meshes are NOT in the cvg-dex-teleop git tree: FR3 visuals from
  github.com/frankaemika/franka_description `meshes/robots/fr3/visual/visual_link{i}.dae`
  (converted to STL with trimesh, no axis remap -> `meshes/franka/link{i}_vis.stl`), ORCA
  visuals from orcahand_description `v1/assets/urdf/{right,left}/visual/*.stl` -> `meshes/orca/`.
  The Gavin mount STL (`Gavin90DegMount_edited_P1S.stl`) is missing -> mount link skipped.
  CPU fillPoly over ~350k triangles/arm: ~2-3 s per frame.
- Left-hand joint sign (2026-10-17): the recorded left `wrist` angle is opposite to
  `orcahand_left.urdf`'s wrist axis (left hand config: `wrist: -17`, right: `17`); all other
  joints match. Applied in `orca_hand_fk.JOINT_SIGNS["left"]` (FK/viz only - the dataset keeps
  the recorded convention, so the policy predicts in it and deployment needs no change).
  Found with `left_hand_candidates.py` (zoomed mesh overlays under candidate sign flips).
