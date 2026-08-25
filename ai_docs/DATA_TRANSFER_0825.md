# Data transfer manifest — everything in current use (2026-08-25)

For the labmate receiving the full 3D data stack over SCP. This file travels inside
`handoff_calib/` together with the transform matrices and the two deep-dive docs
(`rgbd_data_handoff.md` = formats/frames/semantics; `DATA_INVENTORY.md` = provenance).
Host: `sky2.cc.gatech.edu` (any Skynet login node). All paths group-readable
(`coc-skynet-access`).

---

## A. The one-command downloads (destination `.`)

**TIER 1 — 3D essentials (~30 G): depth + point clouds + transforms**
(you already have RGB; this is everything you said you're missing)

```bash
scp -r \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/handoff_calib \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_fullpp_rgbd \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_dp3_robotglass \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_dp3_dual \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_dp3c_dual \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_dp3_transplant \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_colour_rgbd \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_pcd1024_glass \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/scratch/czhang883/aria_fs_out/depth_store \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/scratch/czhang883/aria_fs_out/fastfs_fpp_depth_npy \
  .
```

**TIER 2 — the rest (adds ~35 G: every remaining dataset + raw VRS)**

```bash
scp -r \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_fullpp_rgbd_eef \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_dp3_eefball \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_dp3_eefframe \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/human_dp3_full_eefframe \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/aria_fullpp \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_slamrect_rgbd \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_pcd2048_glass \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_pcd1024_tight \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_pcd2048_tight \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_pp_rect_sg \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_pickplace_val_rgb \
  czhang883@sky2.cc.gatech.edu:/coc/flash7/czhang883/Documents/mobile_generalist/data/pick_place_aria_vrs \
  .
```
(For 30–65 G, `rsync -avP` with the same source list is resumable and safer than
scp over a flaky connection — same syntax, sources then `.`.)

---

## B. What each item is (table of everything above)

### Transform / calibration bundle — `handoff_calib/` (<1 MB, get it first)

| file | what it is | use it for |
|---|---|---|
| `aria_transforms.json` | **every matrix + convention in one file**: rect intrinsics (both devices), stereo baselines, `T_device_rect` (glass↔rect), head-mount `M` (0726), eef-chain definitions, 49-D action layout | the single source of truth for frames |
| `robot_rect224_lut.npz` | precomputed remap: robot raw 640² BGR → 224² colour-rect (`map_x`,`map_y`) | colourize robot depth; deploy-side images |
| `robot_rgb_solved.npz` | solved robot RGB camera model (rect→640 RGB, pinhole+k1k2; 0.43 px) | project anything into the robot RGB image |
| `head_mount.json` | solved Aria-on-robot mount per capture day (`M` = T_head2_device) | robot FK → glass frame |
| `human_vio/*.npz` (59) | per-depth-frame VIO poses `T_world_dev` for glove recordings | human camera trajectories without VRS parsing |
| `tag_manifest.tsv` | demo-tag ↔ VRS filename ↔ collection (146 rows) | trace any demo to its raw recording |
| `data_inventory.json` | machine-readable inventory of every raw file | scripted access |
| `rgbd_data_handoff.md` / `DATA_INVENTORY.md` | the deep-dive docs: formats, frames, alignment residuals, 60 Hz caveat, gotchas | READ FIRST |

### Tier-1 datasets (LeRobot: parquet rows; images = PNG bytes; clouds/depth = flat float32 lists)

| dataset | size | the 3D payload per row | frame / transformation applied |
|---|---|---|---|
| `human_fullpp_rgbd` | 11G | `obs.aria_depth` (50176→224², metres) + RGB | human rect frame; colour↔depth aligned 1.26 px |
| `human_dp3_robotglass` | 838M | `obs.aria_pcd` (3072→1024×3) | human depth → world (VIO Kabsch fit) → **estimated robot glass frame** (retarget FK + mount); crop = robot-rect Z 0.25–2.0 m |
| `human_dp3_dual` | 1.6G | `obs.aria_pcd` (global, glass) + `obs.aria_pcd_local` (1024×3, **right-eef coords**, 1.5 m ball) + `obs.eef_pose_glass` (9) + `obs.eef_pos_glass` (3) | as above + per-row `T_glass_eefR` re-expression |
| `human_dp3c_dual` | 2.2G | `obs.aria_pcdc` / `obs.aria_pcdc_local` (6144→1024×**6 xyzrgb**, RGB∈[0,1]) + eef pose | same frames + per-point colour; rgb=0.5 grey within 0.30 m of either eef (embodiment neutralization) |
| `human_dp3_transplant` | 1.6G | same keys as dual | same frames + human arm pts excised (11 cm of retargeted robot arm surfaces), robot arm mesh pts inserted |
| `rby1_teleop_colour_rgbd` | 1G | robot-corpus RGB-D (colour-rect + depth) | robot rect frame; colour via solved model 0.43 px |
| `rby1_teleop_pcd1024_glass` | 138M | teleop 1024-pt clouds | robot glass frame; the hardware-validated DP3 recipe |
| `depth_store/` | 2.7G | **raw 512² teleop depth (uint16 mm)** + grey rect imgs + per-file calib attrs (`K`,`T_device_rect`,`baseline_m`) | rect frame — 4× the resolution of dataset depth |
| `fastfs_fpp_depth_npy/` | 11G | **raw 512² human depth (float32 m)**, filename = device-ts ns | human rect frame; ≤66 ms staleness vs rows |

### Tier-2 (completeness): `human_fullpp_rgbd_eef` (+`obs.eef_T` 16-d), the 3 remaining
single-stream eef variants, raw-fisheye `aria_fullpp`, grey teleop RGBD, remaining
pcd budgets/crops, v4 RGB teleop set, channel-fixed val set, and the 146 raw VRS
(all streams: SLAM stereo, VIO, handtracking, calib — see `DATA_INVENTORY.md` §2).

## C. Reading recipes (30 seconds each)

```python
import pyarrow.parquet as pq, numpy as np, json
t  = pq.read_table("human_dp3_dual/data/chunk-000/episode_000000.parquet")
pc = np.array(t.column("obs.aria_pcd")[0].as_py(), np.float32).reshape(1024,3)      # metres, glass
T  = json.load(open("handoff_calib/aria_transforms.json"))                          # all matrices
# depth npy (human, 512²): np.load(".../<ts>.npy")  -> float32 metres, rect frame
# depth store (teleop):    h5py: f["depth"][i]/1000. -> metres; f.attrs["K"], f.attrs["T_device_rect"]
# lift: X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy  with the rect intrinsics for that device
```
⚠ Traps that cost us debugging cycles (details in `rgbd_data_handoff.md`): use
`actions.joint_base_torso_head_arm_hand`, never `actions.joint`; human rows are
60 Hz labeled 10; depth 0 = invalid, NEAREST resize only; robot raw images are BGR.
