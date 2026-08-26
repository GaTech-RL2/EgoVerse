# Complete data inventory — absolute paths + metadata (Skynet)

Every recording we have, reconciled against the collection table. All counts measured
from disk 2026-08-06 (job 3681990), not recalled. Machine: `sky2.cc.gatech.edu`
(any Skynet login node — these are shared filesystems). Group `coc-skynet-access`.

Machine-readable version of everything below:
**`/coc/flash7/czhang883/tmp/data_inventory.json`**

---

## 1. Reconciliation with the collection table — it matches exactly

### Human (Aria Gen 2 glasses)

| collection | your table | VRS files on disk | ok | note |
|---|---|---|---|---|
| [Glove] Nav + Pick-and-Place | 30 | 30 | 30 | |
| [Glove] Pick-and-Place Only | 30 | 30 | 30 | |
| [Bare Hands] Nav + Pick-and-Place | 56 | **57** | **56** | 1 marked `dropped` (`nvpp26`) |
| [Bare Hands] Pick-and-Place Only | 29 | 29 | 29 | |
| **Total** | **145** | 146 | **145** | ✅ |

Downstream: 145 ok → **135 demos** in the training file `fullpp_train_v1.hdf5`
(10 further dropped during trimming/QC).

### Teleop (RBY1 robot)

| collection | your table | where it lives | note |
|---|---|---|---|
| [Teleop] Nav + Pick-and-Place | 5 | `0717_unified/fix_head_base_0717_merged.hdf5` | 5 demos / 2,243 frames — also serves as the **validation set** |
| [Teleop] Pick-and-Place | 72 | `0724_*` + `0726_*` (8 session files, 82 raw demos) | 72 survive QC → the training set |
| **Total** | **77** | | ✅ |

---

## 2. HUMAN — raw VRS (the richest source: all streams + calibration)

**Root:** `/coc/flash7/czhang883/Documents/mobile_generalist/data/pick_place_aria_vrs/`

| subfolder | files | size | absolute path |
|---|---|---|---|
| `glove_nav_pick_and_place/` | 30 | 3.0 GB | `…/pick_place_aria_vrs/glove_nav_pick_and_place/` |
| `glove_pick_and_place_only/` | 30 | 2.6 GB | `…/pick_place_aria_vrs/glove_pick_and_place_only/` |
| `nav_pick_and_place/` | 57 | 6.8 GB | `…/pick_place_aria_vrs/nav_pick_and_place/` |
| `pick_and_place_only/` | 29 | 2.5 GB | `…/pick_place_aria_vrs/pick_and_place_only/` |
| **total** | **146** | **~15.4 GB** | |

Flat symlink view (all in one dir, convenient for globbing):
`/coc/flash7/scratch/czhang883/aria_fs_out/fpp_all_vrs/`

**Streams in every VRS** (verified by opening them):

| stream id | label | detail |
|---|---|---|
| 214-1 | **camera-rgb** | 2560×1920, H.265, ~10 Hz |
| 1201-1 / 1201-2 | **slam-front-left / -right** | greyscale stereo pair — the depth source |
| 1201-3 / 1201-4 | slam-side-left / -right | |
| 211-1 / 211-2 | camera-et-left / -right | eye tracking, 200×200 |
| **371-1** | **handtracking** | on-device hand poses |
| **371-2 / 371-3** | **vio / vio_high_frequency** | **6-DoF device trajectory, on-device — no MPS needed** |
| 373-1 | eyegaze | |
| 1202-1/2, 1203-1 | imu-left/right, mag0 | |
| 231-1, 246-1, 247-1, 248-1 | mic, temperature, baro0, ppg | |
| 281-1, 282-1, 283-1, 285-1 | gps-app, wps, bluetooth, utc | |
| — | **factory calibration** | all camera intrinsics + extrinsics, embedded |

**Episode ↔ VRS mapping (essential):**
`/coc/flash7/czhang883/Documents/mobile_generalist/data/full_pick_and_place_hdf5/tag_manifest.tsv`
— TSV, 146 rows: `tag · collection · vrs_filename · status`. Tags look like `gonly1`,
`gnvpp7`, `nvpp26`, `ppo3`. Every processed demo carries its `source_tag` attribute, so
any demo traces back to its VRS through this file.

---

## 3. HUMAN — processed HDF5 (224², ready to load)

**Root:** `/coc/flash7/czhang883/Documents/mobile_generalist/data/full_pick_and_place_hdf5/`

| file | demos | frames | size | what it is |
|---|---|---|---|---|
| **`fullpp_train_v1.hdf5`** | **135** | **53,664** | 2.14 GB | ★ **the training set** (all 4 collections merged, QC'd) |
| `nav_pick_and_place.hdf5` | 56 | 33,767 | 1.31 GB | bare-hands nav+pp, untrimmed |
| `nav_pick_and_place_trimmed.hdf5` | 56 | 27,866 | 1.09 GB | …raise-cut applied |
| `glove_nav_pick_and_place.hdf5` | 30 | 15,207 | 0.60 GB | glove nav+pp |
| `glove_nav_pick_and_place_trimmed.hdf5` | 30 | 12,602 | 0.50 GB | |
| `glove_pick_and_place_only.hdf5` | 30 | 10,747 | 0.44 GB | glove pp-only |
| `glove_pick_and_place_only_trimmed.hdf5` | 30 | 8,397 | 0.35 GB | |
| `pick_and_place_only.hdf5` | 29 | 10,397 | 0.42 GB | bare-hands pp-only |
| `pick_and_place_only_trimmed.hdf5` | 29 | 7,752 | 0.32 GB | |
| `exp1_bare_train.hdf5` | 56 | 19,110 | 0.77 GB | experiment split |
| `exp1_glove_train.hdf5` | 56 | 20,075 | 0.81 GB | experiment split |
| `exp1_bare_navonly.hdf5` | 29 | 11,749 | 0.46 GB | experiment split |

Structure: `data/demo_<i>/obs/{aria_image (T,224,224,3) uint8 RGB, aria_ts, …}`,
`data/demo_<i>/actions/joint (T,49)`, attrs incl. `source_tag`.
**Rows are 60 Hz** (measured Δt = 16.67 ms) even though downstream fps is labeled 10.

---

## 4. TELEOP — raw HDF5 (per-session, 640² BGR)

**Roots:** `…/data/0724_teleop_pick_and_place/0724_teleop_pick_and_place/`,
`…/data/0726_teleop_pick_and_place/`, `…/data/0717_unified/`

| session file | demos | frames | size | depth? |
|---|---|---|---|---|
| `0717_unified/fix_head_base_0717_merged.hdf5` | **5** | 2,243 | 1.29 GB | — (the **nav+pp** set / val) |
| `0724…/fix_head_base_20260724_174423.hdf5` | 17 | 1,685 | 2.01 GB | ✅ |
| `0724…/fix_head_base_20260724_180309.hdf5` | 16 | 1,270 | 1.22 GB | ✅ |
| `0726…/fix_head_base_20260726_131558.hdf5` | 14 | 1,647 | 1.68 GB | ✅ |
| `0726…/fix_head_base_20260726_133802.hdf5` | 2 | 350 | 0.41 GB | ✅ |
| `0726…/fix_head_base_20260726_134055.hdf5` | 7 | 891 | 0.93 GB | ✗ |
| `0726…/fix_head_base_20260726_134934.hdf5` | 3 | 494 | 0.39 GB | ✗ |
| `0726…/fix_head_base_20260726_153841.hdf5` | 6 | 753 | 0.90 GB | ✅ |
| `0726…/fix_head_base_20260726_154438.hdf5` | 17 | 2,108 | 2.52 GB | ✅ |
| | **82 raw pp** | | | |

Derived merges (already 224², convenient):
`0724…/0724_teleop_pp_merged_224_rgb.hdf5` (28 demos) and `…_sg.hdf5` (same, SG-smoothed
actions); `0717_unified/fix_head_base_0717_merged_224_rgb.hdf5` (5 demos, 224²).

Structure: `data/demo_<i>/obs/{aria_image (T,640,640,3) uint8 **BGR**, slam, vio_pose,
robot joint states}`, `actions/…`. **Rows are 10 Hz.**
⚠ Robot-recorded images are **cv2-BGR**; the human pipeline is RGB.

---

## 5. DEPTH (Fast-FoundationStereo, 512²)

| what | path | contents |
|---|---|---|
| **Teleop depth store** | `/coc/flash7/scratch/czhang883/aria_fs_out/depth_store/{0724_teleop,0726_teleop}/*.h5` | 6 files, **7,428 frames**; `depth` uint16 **mm** 512², `image_rect` uint8, `demo_id`, `frame_idx`, `ts_ns`, `depth_valid`; attrs `K`, `T_device_rect`, `focal_px`, `baseline_m` |
| **Human depth** | `/coc/flash7/scratch/czhang883/aria_fs_out/fastfs_fpp_depth_npy/<rec>/<ts_ns>.npy` | **146 dirs, 10,841 frames**; float32 **metres** 512², filename = device timestamp |
| Human VIO poses (pre-extracted) | `/coc/flash7/czhang883/tmp/human_vio/*.npz` | 59 glove recs: `T_world_dev` 4×4 per depth frame |

Rect-frame intrinsics: teleop `fx=fy=307.336684`, human `fx=fy=308.052437`,
`cx=cy=256.0` @512. Zero distortion (pinhole).

---

## 6. TRAINING-READY LeRobot datasets

**Root:** `/coc/flash7/czhang883/Documents/EgoVerse/datasets/`

### Human-corpus

| dataset | eps | frames | obs |
|---|---|---|---|
| **`human_fullpp_rgbd`** | 135 | 53,664 | colour-rect image + **depth** |
| `aria_fullpp` | 135 | 53,664 | raw fisheye image (row-parallel to above) |
| `aria_fullpp_wam3` | 135 | 53,664 | image (world-model variant) |
| `exp1_glove` / `exp1_bare` | 56 / 56 | 20,075 / 19,110 | image |
| `exp1_navonly` | 29 | 11,749 | image |

### Teleop-corpus

| dataset | eps | frames | obs |
|---|---|---|---|
| **`rby1_teleop_pp_rect_sg`** | 72 | 8,240 | rectified image (the 72-demo set) |
| `rby1_teleop_pp_0724_0726_sg` | 72 | 8,240 | image |
| **`rby1_teleop_colour_rgbd`** | 63 | 7,003 | colour-rect + **depth** |
| `rby1_teleop_slamrect_rgbd` | 63 | 7,003 | grey-rect + **depth** |
| `rby1_teleop_pcd{1024,2048}_glass` | 63 | 7,003 | **point cloud** (glass frame) |
| `rby1_teleop_pcd{1024,2048}_tight` | 63 | 7,003 | point cloud (tight crop) |
| `rby1_teleop_pp_0724{,_sg,_wam3}` | 28 | 2,570 | image (v2 lineage) |
| **`rby1_teleop_pickplace_val_rgb`** | 5 | 2,243 | image — ★ the **nav+pp val set**, channel-fixed |
| `rby1_teleop_pickplace_val` | 5 | 2,243 | image (⚠ BGR — superseded by `_rgb`) |
| `rby1_teleop_val_v2` | 5 | 1,956 | image |

The 63-episode RGBD/PCD sets are the subset of the 72 that have depth (the 6 sessions
marked ✅ in §4, minus 4 QC-dropped demos from session 131558).

---

## 7. Loading recipes (for whoever visualizes)

```python
# LeRobot dataset (training-ready)
import pandas as pd, numpy as np, cv2
pi = pd.read_parquet("datasets/human_fullpp_rgbd/data/chunk-000/episode_000000.parquet")
v = pi.iloc[0]["obs.aria_image"]                      # PNG bytes in a dict
img = cv2.imdecode(np.frombuffer(v["bytes"], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
depth = np.asarray(pi.iloc[0]["obs.aria_depth"], np.float32).reshape(224, 224)   # metres
act = np.asarray(pi.iloc[0]["actions.joint_base_torso_head_arm_hand"], np.float32)  # (49,)
```
⚠ **Use `actions.joint_base_torso_head_arm_hand`, not `actions.joint`** — both are 49-D
and they are *different vectors*.

```python
# Raw HDF5
import h5py
f = h5py.File(".../fullpp_train_v1.hdf5", "r")
img = f["data/demo_0/obs/aria_image"][t]              # (224,224,3) RGB (human) / BGR (teleop)
act = f["data/demo_0/actions/joint"][t]               # (49,)
```

```python
# VRS (needs projectaria_tools)
from projectaria_tools.core import data_provider
dp = data_provider.create_vrs_data_provider(".../glove_pick_place_only_1_20260718_214945.vrs")
[dp.get_label_from_stream_id(s) for s in dp.get_all_streams()]
```

Repo visualizer:
`python egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py <ds>/LeRobot -k actions.joint_arm --dims 0:14 -e 0`

---

## 8. Gotchas worth passing on

1. **Channel order.** Human = RGB. Teleop raw = **BGR**. Datasets: all human + `*_rgbd` +
   `*_val_rgb` are RGB; `rby1_teleop_pp_rect_sg`, `pp_0724*`, `pickplace_val` are BGR.
2. **Row rates differ.** Human HDF5/parquet rows are **60 Hz** (labeled fps=10); teleop is
   genuinely 10 Hz. A 32-row chunk = 0.53 s of human motion vs 3.2 s of robot motion.
3. **Human proprio is retargeted**, not measured — it is ≈ the action label. Teleop proprio
   is real encoders.
4. **Effective visual dataset size**: human's 53,664 rows contain only ~9,062 unique RGB
   frames (rows share observations), roughly the same as teleop's 7,003.
5. Depth `0` = invalid. Resize with **NEAREST** only.

Full technical companion (frames, calibration, action layout, alignment residuals):
`ai_docs/rgbd_data_handoff.md`.
