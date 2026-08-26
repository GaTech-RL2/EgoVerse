# RGB-D data handoff — teleop + human corpora (RBY1 pick-and-place)

**Audience:** a collaborator training their own encoder on this data, possibly using
depth. Everything here is measured/verified as of 2026-08-05, not recalled from memory;
numbers with a job id in parentheses were produced by that SLURM job and are
reproducible from its script in `/coc/flash7/czhang883/tmp/`.

All paths below are on the Skynet cluster, group-readable (`coc-skynet-access`).
Repo root: `/coc/flash7/czhang883/Documents/EgoVerse/` (paths like `datasets/...` are
relative to it).

---

## 0. One-paragraph orientation

Two corpora of the same tabletop task (pick penguin, place in blue basket; some
episodes include walking to the table), same 49-D whole-body action space, same RBY1
robot embodiment:

| | **teleop** | **human** |
|---|---|---|
| collected by | operator teleoperating the RBY1 | a person wearing an Aria Gen 2, gloves |
| observations | robot's head-mounted Aria | human's head-mounted Aria |
| actions | recorded robot joint commands | SEW functional retargeting of human motion |
| proprio | **real measured** joint encoders | **retargeted** (≈ the action — see §7!) |
| episodes / rows | 63 / 7,003 | 135 / 53,664 |
| wall-clock row rate | **10 Hz** | **60 Hz** (⚠ labeled 10 — see §6) |
| depth source | Fast-FoundationStereo on the robot Aria's SLAM stereo pair | Fast-FoundationStereo on the human Aria's SLAM stereo pair |
| unique obs frames | every row unique (all streams row-synced) | 9,062 unique RGB / 6,811 unique depth over 53,664 rows |

The flagship RGB-D datasets (one per corpus) are **row-for-row parallel** to the
established RGB datasets of their corpus, so any result is directly comparable to the
existing baselines (§10).

---

## 1. The policy frame ("slam-rect"): where image and depth live

Both RGB-D datasets put image and depth in the **same pinhole frame**, called
*slam-rect*: the scanline-rectified LEFT SLAM camera. This is the frame the stereo
depth is *native* to — depth[i,j] and image[i,j] are the same ray **by construction**
(the rectified left SLAM image is literally the image the stereo matcher consumed).
There is no cross-camera warp between image and depth in the grey datasets, and a
*validated* warp brings colour into this frame in the colour datasets (§4).

Axes: optical convention — **X right, Y down, Z forward** (out of the camera).
Units: metres everywhere.

### Intrinsics (pinhole, zero distortion, exact constants)

| corpus | K at 512×512 | K at 224×224 (×224/512) | stereo baseline |
|---|---|---|---|
| teleop (robot Aria) | fx=fy=**307.336684**, cx=cy=256.0 | fx=fy=134.4598, cx=cy=98.0 | 0.134383 m |
| human (human Aria) | fx=fy=**308.052437**, cx=cy=256.0 | fx=fy=134.7729, cx=cy=98.0 | 0.1420 m |

Each corpus is ONE physical device with one factory calibration — the human-side
constant was checked across 12 recordings: spread exactly 0. FOV ≈ 79.6° HFOV.
Construction: `fisheye_to_linear_calib(slam-front-left, focal_scale=1.25, 512)` +
`create_scanline_rectified_cameras` from
`Documents/aria_gen2_scripts/src/utils/stereo_utils.py` (their own convention,
reproducible).

### Rect → device rotation (for lifting to a head-fixed frame / point clouds)

`P_device = R @ P_rect` with `R = T_device_rect[:3,:3]` (pure rotation, zero
translation). For the teleop store it is saved in each store file's attrs; value for
the robot device:

```
T_device_rect = [[ 0.934902, -0.141367, 0.325537, 0],
                 [ 0.149511,  0.988760, 0.0,      0],
                 [-0.321878,  0.048671, 0.945529, 0],
                 [ 0, 0, 0, 1]]
```

"device" = the Aria device frame (anchored at slam-front-left, X right / Y down /
Z fwd). The DP3 point-cloud datasets (§9) are in this device ("glass") frame.

---

## 2. Dataset inventory (LeRobot format, ready to train)

All under `datasets/`. LeRobot v2-style: parquet per episode + PNG images
(`dtype=image`, NOT video — no codec loss), `meta/info.json` carries the feature spec,
`meta/episodes.jsonl` the per-episode lengths.

### Flagship RGB-D (what you probably want)

| dataset | corpus | eps / rows | image | depth | size |
|---|---|---|---|---|---|
| **`rby1_teleop_colour_rgbd`** | teleop | 63 / 7,003 | COLOUR rect 224², **RGB** | ✅ | ~1.0 G |
| **`human_fullpp_rgbd`** | human | 135 / 53,664 | COLOUR rect 224², **RGB** | ✅ | ~11 G |
| `rby1_teleop_slamrect_rgbd` | teleop | 63 / 7,003 | GREY rect ×3ch | ✅ | 846 M |

⚠ `rby1_teleop_colour_rgbd` was **rebuilt on 2026-08-05** (job 3654159): the first
build stored BGR, which silently degrades any frozen RGB-pretrained backbone (§8).
If your copy predates 08-05, re-sync. Quick self-check: frame (demo 0, row 10) must
show a **blue** basket and **orange** penguin feet when interpreted as RGB.

### RGB-only baseline datasets (for encoder A/Bs without depth)

| dataset | corpus | eps / rows | image |
|---|---|---|---|
| `aria_fullpp` | human | 135 / 53,664 | raw fisheye 224², RGB — row-parallel to `human_fullpp_rgbd` |
| `rby1_teleop_pp_rect_sg` | teleop | 72 / 8,240 | RGB-camera pinhole rect (f=373@640→224), **BGR**, SG actions — the v4 dataset. ⚠ different frame AND different episode set than the slam-rect datasets; not row-parallel |
| `rby1_teleop_pp_0724*` | teleop | 28–72 | raw fisheye 224², **BGR** (v2/v3 lineage) |

### Point-cloud datasets (geometry-only, teleop)

| dataset | cloud | crop | frames |
|---|---|---|---|
| `rby1_teleop_pcd1024_glass` / `pcd2048_glass` | 1024 / 2048 × xyz | Z-slab 0.25–2.0 m | 63 / 7,003 |
| `rby1_teleop_pcd1024_tight` / `pcd2048_tight` | 1024 / 2048 × xyz | **range** 0.25–1.5 m | 63 / 7,003 |

Glass-frame builds beat tight offline at both budgets (see
`ai_docs/pcd_policy_deployment_guide.md`). Construction recipes in §9.

### Row-parallelism guarantees (verified by assert at build time)

- `human_fullpp_rgbd` ≡ `aria_fullpp`: same 135 episodes in the same order, same
  53,664 rows, `actions/joint` **bit-identical**. Only the image differs (+ depth added).
- `rby1_teleop_colour_rgbd` ≡ `rby1_teleop_slamrect_rgbd` ≡ both `pcd*` datasets:
  same 63 episodes, same 7,003 rows, same SG-smoothed actions. Only the observation
  differs.

So "my encoder on dataset X vs baseline B" is a controlled comparison within a corpus.

---

## 3. Feature schema (identical key set in both flagship RGB-D datasets)

| lerobot key | shape | dtype | units / semantics |
|---|---|---|---|
| `obs.aria_image` | (224,224,3) | uint8 image | slam-rect frame, **RGB** (see §8 for the per-dataset channel-order table) |
| `obs.aria_depth` | **(50176,)** | float32 | **flattened 224×224, reshape yourself**; metres; **0 = invalid** |
| `obs.robot0_joint_pos` | (26,) | float32 | raw joint vector (incl. wheels) — most training uses the 22-D instead |
| `obs.robot0_joint_pos_no_wheel` | (22,) | float32 | **order: [torso 6, r_arm 7, l_arm 7, head 2]** — the proprio the baselines consume |
| `obs.hand_left_qpos` / `obs.hand_right_qpos` | (12,) each | float32 | XHand joint positions |
| `actions.joint` | (49,) | float32 | THE action (layout §5) |
| `actions.joint_arm`, `actions.joint_arm_hand`, … | various | float32 | sliced variants of the 49-D (arm-only 14, arm+hand 38, …) — derived, same source |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | scalars | | LeRobot bookkeeping; `timestamp = frame_index / 10` (declared fps) |
| `metadata.embodiment` | scalar | int | RBY1 = **12** |

Depth: resize with **NEAREST only** (bilinear invents flying points at object edges);
never in-paint the zeros — the baselines see them as-is.

---

## 4. Where the pixels come from (provenance + measured alignment)

### 4a. Depth (both corpora)

Fast-FoundationStereo (learned stereo) on the rectified SLAM front pair, computed
offline on SLURM. Dense within the stereo overlap (a human-side sample frame:
min 0.28 m, no holes); the bottom sliver of the frame is outside the overlap.

- **Teleop**: stored per-row in the *depth store*
  (`/coc/flash7/scratch/czhang883/aria_fs_out/depth_store/<set>/<rec>.h5`):
  `depth` uint16 **millimetres** 512², `image_rect` uint8 512², `demo_id`,
  `frame_idx`, `ts_ns`, `depth_valid`; attrs `K`, `T_device_rect`, `focal_px`,
  `baseline_m`. Depth row ↔ HDF5 row pairing is exact (same recorder row).
- **Human**: per-frame float32-metre npys named by device timestamp (ns),
  `/coc/flash7/scratch/czhang883/aria_fs_out/fastfs_fpp_depth_npy/<rec>/<ts>.npy`,
  at **7.5 Hz** over a ~10 s window per recording (the manipulation phase). Rows are
  matched to the *nearest* depth frame: staleness ≤ **66.5 ms** by construction
  (half-period), verified max across all 135 demos. `aria_ts` in the source HDF5s is
  **device time** (same clock as the npy names) — no cross-clock solve needed.

Human depth value distribution over the built dataset: valid 100.0%, median 1.74 m,
p95 5.08 m (includes nav phases — much longer range than the teleop table scene,
which lives at ~0.3–3.3 m).

### 4b. Colour (the part that was historically impossible)

Colour is **transported into the rect frame via the depth**: every rect pixel with
depth is lifted to 3D, projected into the RGB camera, and samples its colour. Two
different RGB models were needed:

**Human corpus** — the VRS ships a factory `camera-rgb` calibration
(FISHEYE624, 2560×1920, f=1118.811). RGB is undistorted to a 1408² pinhole
(`focal_scale 0.35`, f=391.58) and sampled via the factory RGB↔SLAM extrinsic
(‖t‖≈12 mm). Measured residual against the by-construction-aligned grey image
(job 3651665): **median 1.26 px @512** (0.55 px at 224), 0.69 px on static frames,
p90 3.38 px. RGB is 10 Hz → nearest frame per row, |gap| mean 24.9 ms.

**Teleop corpus** — the robot recordings **never stored a camera-rgb calibration**
(publisher published SLAM-only; there is no robot VRS). The RGB model was **solved
from the data** (job 3652632): depth gives 3D at every rect pixel; SIFT
correspondences into the stored 640² fisheye RGB; robust least-squares over
`[rvec, t, f, cx, cy, k1, k2]` (pinhole + 2 radial terms). Solution:

```
rvec = [-0.246718, -0.003774, 0.006088]      # rect -> rgb rotation (Rodrigues)
t    = [-0.009099,  0.000110, -0.004534]     # metres
f=370.397  cx=310.47  cy=319.73  k1=-0.31958  k2=0.11009
# full params: /coc/flash7/czhang883/tmp/robot_rgb_solved.npz
# projection: q=R@P+t; x=qx/qz; y=qy/qz; r2=x²+y²; d=1+k1·r2+k2·r2²
#             u=f·x·d+cx; v=f·y·d+cy   (into the stored 640×640 image)
```

2,716 inliers at 1.22 px median reprojection. Held-out validation with ONE model
across all 6 recording sessions: **median 0.43 px @512, p90 1.55 px** — per-session
medians 0.15–0.96 px, i.e. it is a true device constant. (The stored 640² is a centre
crop of the fisheye — 1920² of 2560×1920 — scaled ⅓; hence f≈373 ≈ 1118.8/3.)

### 4c. The black regions in colour images — expected, quantified, not a bug

Colour is transported via depth from a *different physical camera*, so pixels with no
colour ray are black:

| corpus | black/frame (median) | where | mechanism |
|---|---|---|---|
| human | 4.1% (max 7.5%) | rows ≥207/224 (bottom arc) + rare far-corner specks | SLAM camera sees below the RGB fisheye's image circle (measured: 0.00% out-of-bounds, 4.34% beyond image circle) |
| teleop | 9.8% (max 10.4%) | rows ≥168/224, 0.0% above | same + the stored RGB is a centre crop → narrower FOV |

**Rows 0–168 (all task content: table, objects, hands/grippers) are 0.0% black in
both corpora — verified over the full datasets, worst frames included.** Secondary
coupling: invalid-depth pixels are also colour-black (colour rides on depth). The
grey teleop dataset has NO black regions (native camera image) — if you compare grey
vs colour encoders, the colour arm sees ~90–96% of the pixels.

---

## 5. Action space (49-D, identical both corpora)

`actions.joint`, concatenation (indices inclusive-exclusive):

| block | idx | dim | meaning |
|---|---|---|---|
| base | [0:3] | 3 | **per-step deltas** (Δx, Δy, Δyaw) in the frame-0 heading — integrate by plain cumsum; NOT absolute pose |
| torso | [3:9] | 6 | absolute joint targets (rad) |
| head | [9:11] | 2 | absolute (rad) |
| l_arm | [11:18] | 7 | absolute (rad) |
| r_arm | [18:25] | 7 | absolute (rad) |
| l_hand | [25:37] | 12 | absolute (rad) |
| r_hand | [37:49] | 12 | absolute (rad) |

- **Teleop**: recorded commands, then **Savitzky-Golay smoothed (win 7, poly 3) —
  `actions.joint` only**, sliced variants unsmoothed. (SG at 10 Hz = 0.7 s window;
  validated on hardware in the v2 round.)
- **Human**: SEW functional-retargeting output, **no smoothing** (solver output is
  already smooth). Bit-identical to `aria_fullpp`'s actions.
- Hierarchical decoding (if your head wants it): `block_dims [3,6,2,14,24]`, DAG
  parents `[[],[0],[0,1],[0,1],[0,1,3]]`.

### Training target convention (what every baseline predicts)

Action **chunk of 32 consecutive rows**: LeRobot `delta_timestamps` on
`actions.joint_base_torso_head_arm_hand` (the same 49-D under its schematic name) =
`[0.0, 0.1, ..., 3.1]` at the declared fps 10 → rows t..t+31. Match this (32 rows,
stride 1) or your numbers are not comparable to §10.

---

## 6. ⚠ Time semantics — the one thing most likely to bite you

**Both corpora declare fps=10 in LeRobot metadata. Only teleop is actually 10 Hz.**

- Teleop rows: measured Δt ≈ 0.10–0.11 s wall clock. A 32-row chunk = **3.2 real
  seconds**. Robot executes at 10 Hz → 1:1 speed.
- Human rows: measured Δt = **16.67 ms (60.0 Hz)** wall clock (the retargeting/mocap
  rate). A 32-row chunk = **0.53 real seconds** of human motion. Deployed at 10 Hz,
  the robot replays human motion at **1/6 speed**. This is a known, accepted property
  of the whole human-policy line (all hardware results were obtained this way), not a
  bug — but if your encoder uses temporal context or you compute velocities, know
  that "1 frame" means 16.7 ms of motion on human and 100 ms on teleop.
- Consequence of 60 Hz rows vs ~10 Hz sensors (human only): consecutive rows **share**
  observation frames — 53,664 rows contain 9,062 unique RGB images and 6,811 unique
  depth maps (~6 rows per image, ~7.9 per depth). Effective visual dataset size is
  ~9k/~7k frames, roughly EQUAL to teleop (7,003 unique), not 7.6× bigger. Plan
  regularisation accordingly (63-demo teleop overfits hard; 135-ep human is not as
  much bigger as the row count suggests).

---

## 7. Proprio — real vs retargeted (second most likely bite)

- **Teleop `robot0_joint_pos*`**: real measured encoders. Honest input.
- **Human `robot0_joint_pos*`**: produced by the SAME retargeting that produced the
  actions. Proprio(t) ≈ action(t−1) — a near-copy of the label. A capable model will
  happily "solve" human data from proprio alone and learn nothing visual. Every human
  baseline in §10 was trained with **proprio dropout p=0.9 + Gaussian noise
  σ=0.03 (normalized units)** on all three proprio stems for exactly this reason.
  If you train on human data with full proprio and beat the baselines, you have
  probably measured leakage, not your encoder. (This is also why old human-round
  val numbers on proprio-heavy configs were dishonest.)
- Hands (`hand_*_qpos`): measured on teleop; retargeted on human (same caveat).

---

## 8. Channel order per dataset (updated 2026-08-05, learn from our scar tissue)

The robot-side recorder writes **cv2-BGR**; the human Aria pipeline writes **RGB**.
Current state of every image dataset:

| dataset | channel order |
|---|---|
| `human_fullpp_rgbd`, `aria_fullpp`, all human/`exp1_*` datasets | RGB |
| `rby1_teleop_colour_rgbd` (post 08-05 rebuild) | **RGB** (flipped at build) |
| `rby1_teleop_slamrect_rgbd` | grey ×3 (order-invariant) |
| `rby1_teleop_pp_rect_sg` (v4), `rby1_teleop_pp_0724*` (v2/v3), `rby1_teleop_pickplace_val` | **BGR as recorded** |
| `rby1_teleop_pickplace_val_rgb` | RGB (fixed copy — use this one for eval) |

Rule that was violated twice in this project, at measurable cost (+0.021 val MAE
once; one voided training run once): **a frozen RGB-pretrained backbone (DINO, CLIP,
…) must receive RGB.** BGR is only tolerable for from-scratch / fully-fine-tuned
encoders where deployment also sends BGR. Fastest sanity check on ANY of these
datasets: the basket is blue, the penguin's feet are orange.

---

## 9. Point-cloud construction (if you'd rather lift depth yourself)

Exact recipe behind the `pcd*_glass` datasets (reproduce, or just use them):

```
depth (512², metres, rect frame)
  -> mask 0.25 < Z < 2.0                          # glass builds; tight: 0.25 < range < 1.5
  -> X=(u-cx)·Z/fx ; Y=(v-cy)·Z/fy                 # K from §1
  -> subsample to ≤16384                           # glass: random; tight: deterministic stride
  -> farthest-point-sample to N (1024 or 2048)
  -> P_device = P_rect @ R.T                       # R = T_device_rect[:3,:3]
```

Stored flattened float32 (`obs.aria_pcd`, 3072-D for N=1024). Empirical findings you
get for free: 1024 > 2048 on held-out val (background points are memorisable noise);
tight crop < glass at both budgets (far points carry context); deterministic
subsampling matters (random re-draws add 22 mm chamfer flicker between identical
frames). Human-corpus clouds don't exist yet as a dataset, but everything needed is
in §4a + §12.

---

## 10. Baselines to beat (all: same split, same 32-row chunk, MAE in native units)

Metric: `Valid/..._mae_avg` = mean |error| over the 32-step chunk × 49 dims,
**unnormalized** (radians/metres mix). Split: deterministic episode holdout —
`sorted(names)` shuffled by `random.Random(42)`, first `int(N·ratio)` episodes
(min 1) are val (`egomimic/rldb/utils.py:split_dataset_names`). Teleop 3D round:
`valid_ratio=0.03` → **1 val episode, 62 train**. Human round: `valid_ratio=0.2` →
**27 val, 108 train**. Use the same ratios or re-run the baselines on your split.

| corpus / dataset | model | best val (ep) | notes |
|---|---|---|---|
| teleop `pp_rect_sg` | v4: ResNet RGB | **0.0859** (299) | the teleop RGB bar |
| teleop `slamrect_rgbd` | Adapt3R grey+depth | 0.1157 (249) | frozen DINOv2 on grey; degrades past optimum |
| teleop `pcd2048_glass` | DP3 (xyz only) | 0.1112 (299) | |
| teleop `pcd1024_glass` | DP3 (xyz only) | **0.1069** (1299) | best 3D policy; flat curve |
| teleop `pcd*_tight` | DP3 | 0.1107 / 0.1161 | tight-crop negative result |
| teleop `colour_rgbd` | Adapt3R colour+depth | *training (RGB rerun)* | first attempt (BGR) void |
| human `human_fullpp_rgbd` | ResNet colour-rect, no depth | **0.0652** (999, still running) | the human bar |
| human `human_fullpp_rgbd` | Adapt3R colour+depth | 0.0895 (699, running) | depth via frozen-DINO lift *costs* ~0.024 at equal train fit |
| human `aria_fullpp` (fisheye) | ResNet (fpp_hd lineage) | — | old runs used a different (teleop) val set; not comparable to the two rows above |

Context worth knowing before you burn GPUs: every 3D encoder tried so far genuinely
*uses* its 3D input (zeroing depth/cloud degrades MAE 17–33×) yet **loses to its
RGB-only counterpart** on both corpora. Nobody has shown depth *helping* on this task
yet — that is precisely the open question your encoder would be answering.
Recipe shared by all baselines: flow-matching action head on a 16-block transformer
trunk, AdamW lr 1e-4, cosine T_max 1400 (human) / 2000 (teleop-3D round), batch 32,
2000 epochs, image augs = random crop ~20–40 px + rotation ≤5° + colour jitter
(brightness/contrast/sat 0.1, hue 0.05) + ImageNet normalize; proprio dropout §7.

---

## 11. How to consume the data

### Option A — plug your encoder into this repo (least work, exact comparability)

The training stack takes any encoder as a config entry; you write one nn.Module with
`output_dim`, register it under `encoder_specs.front_img_1`, and inherit the entire
verified pipeline (chunking, norm, augs, val protocol). See `WAM_QUICKSTART.md` on the
`rby1_encoder_dev` fork (`ZhangChuye/EgoVerse`) for the end-to-end walkthrough; the
Adapt3R config (`egomimic/hydra_configs/model/experiments/wholebody_image/
rby1_adapt3r_human.yaml`) is a live example of an encoder consuming BOTH
`obs.aria_image` and `obs.aria_depth` (depth routed via `depth_key_map` +
`metadata_keys`, arriving unnormalized in metres).

### Option B — standalone (your own stack)

LeRobot: `LeRobotDataset(repo_id=..., root="datasets/human_fullpp_rgbd",
local_files_only=True, delta_timestamps={"actions.joint": [i/10 for i in range(32)]})`.
Reshape `obs.aria_depth` to (224,224).

Or skip LeRobot entirely — the pre-conversion HDF5s are simpler and identical in
content (`data/demo_<i>/` groups, `obs/aria_image` (T,224,224,3) uint8,
`obs/aria_depth` (T,50176) float32, `actions/joint` (T,49), attrs carry provenance):

```python
import h5py, numpy as np
f = h5py.File("/coc/flash7/czhang883/tmp/human_rgbd.hdf5", "r")     # human, 135 demos
d = f["data/demo_0"]
img   = d["obs/aria_image"][t]                        # (224,224,3) uint8 RGB
depth = d["obs/aria_depth"][t].reshape(224,224)       # float32 m, 0=invalid
act   = d["actions/joint"][t:t+32]                    # (32,49) chunk target
```

### Raw / intermediate artefacts (for going deeper than 224²)

| what | where |
|---|---|
| human merged RGB-D HDF5 (224²) | `/coc/flash7/czhang883/tmp/human_rgbd.hdf5` |
| teleop colour RGB-D HDF5 (224²) | `/coc/flash7/czhang883/tmp/tel_colour_rgbd.hdf5` |
| teleop depth store (**512²** depth mm + grey rect + calib attrs) | `/coc/flash7/scratch/czhang883/aria_fs_out/depth_store/` |
| human depth npys (**512²** float32 m, device-ts names) | `/coc/flash7/scratch/czhang883/aria_fs_out/fastfs_fpp_depth_npy/` |
| human per-frame VIO poses + ts indexes (59 glove recs) | `/coc/flash7/czhang883/tmp/human_vio/*.npz` |
| human VRS (full calib, all streams) | `/coc/flash7/scratch/czhang883/aria_fs_out/fpp_all_vrs/` |
| human source HDF5s (actions, 60 Hz rows, `source_tag` attrs) | `.../mobile_generalist/data/full_pick_and_place_hdf5/` (`fullpp_train_v1.hdf5` = the 135) |
| teleop source HDF5s (10 Hz rows, 640² BGR fisheye, SLAM images) | `.../mobile_generalist/data/072{4,6}_teleop_pick_and_place/fix_head_base_*.hdf5` |
| solved robot RGB model | `/coc/flash7/czhang883/tmp/robot_rgb_solved.npz` (§4b) |
| build/validation scripts (every number above) | `/coc/flash7/czhang883/tmp/{build_human_rgbd,merge_human_rgbd,build_tel_colour,solve_robot_rgb,valid_human_rgbd,hole_analysis}.sbatch` |

⚠ At 512² you get 4× the depth resolution of the datasets — if your encoder can use
it, build from the store/npys directly (the 224² datasets exist because the baseline
stack is 224-native, not because 512 is unavailable).

---

## 12. QC / provenance fine print

- **Teleop episode set**: 63 demos from 6 recording sessions (tags 174423, 180309,
  131558, 133802, 153841, 154438). QC exclusion baked into every teleop dataset:
  session 131558 demos {0,1,12,13} dropped (camera-stream freezes).
- **Human episode set**: `fullpp_train_v1.hdf5` = 135 demos (upstream QC dropped 10
  of the 145 raw); mixes 4 sub-collections (glove/bare × nav/pick-only), per-demo
  `source_tag` attr preserved in the intermediate HDF5s. Trimming: episodes end at
  a "raise-cut" (hand approaching glasses at episode end removed).
- **Depth staleness (human)**: per-demo max stored as attr `depth_staleness_ms_max`
  in `human_rgbd.hdf5` (global max 66.5 ms).
- **Normalization**: nothing pre-normalized. Baselines compute quantile stats over
  the train split at startup; actions/proprio are normalized by those, depth is
  consumed in raw metres, images as uint8→float with ImageNet stats. Do your own.
- **Episode ordering**: `demo_<i>` index order == LeRobot `episode_index` order ==
  source order of `fullpp_train_v1` / sorted store iteration. Nothing is shuffled on
  disk; the val split (§10) is the only randomized thing, and it's seeded (42).
- Deeper background docs in `ai_docs/`: `adapt3r_depth_blocker.md` (3D round
  findings + the colour-unblock story), `human_pcd_policy_plan.md` (human 3D
  inventory + colour-hole QC), `pcd_policy_deployment_guide.md` +
  `pcd_tight_deployment_guide.md` (cloud contracts), `depth_data_spec.md`.

*Maintainer: czhang883. Compiled 2026-08-05; verification jobs cited inline.*
