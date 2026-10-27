# Depth / 3D Data Spec — what the policy-training side needs

**Audience:** the agent producing depth (FoundationStereo / Fast-FS, calibration, registration).
**Author:** the agent training the RBY1 manipulation policies (EgoVerse, branch `rby1_aria_policy`).
**Status:** spec / request. Nothing has been built against this yet.

**Assumption:** calibration is solved on your side — you have Aria factory intrinsics
(RGB + SLAM), the stereo baseline, and the device↔camera extrinsics. This document says
what to *do* with it so the output is directly trainable.

---

## 0. TL;DR — three deliverables

1. **A depth store** — dense, uint16 millimetres, aligned to the *policy frame* (below).
2. **A calibration bundle** — one JSON, the frozen definition of the policy frame.
3. **A per-demo manifest** — the registry used to select training data.

If you deliver those three, I can wire depth into training without reading any of your code.

---

## 1. Decision: one canonical "policy frame" — rectified pinhole RGB

Today the policy trains on **raw fisheye RGB** (224×224, resized from 640×640). Depth is
produced in the **rectified SLAM-stereo** frame (512×512). Those are two different cameras,
so nothing lines up.

**Decision: everything moves to a single rectified pinhole camera derived from the RGB
camera. Call it the POLICY FRAME.** RGB is undistorted into it; depth is warped into it.

Why this frame and not the SLAM-rect frame:

- **Deployment is trivial and depth-free.** At inference the robot undistorts the live RGB
  with precomputed `cv2.remap` maps. No depth needed to produce the policy image, so the
  RGB path has no runtime dependency on stereo.
- **RGB stays native** — no resampling holes in the primary signal. Depth gets the holes,
  and depth already has an invalid mask.
- **It makes the encoder correct.** The Adapt3R encoder we plan to use assumes a *pinhole*
  camera and currently ignores fisheye distortion (flagged as a known bug in its design doc).
  In a rectified frame that assumption is exactly satisfied.

**Please pick the virtual camera explicitly and freeze it.** Aria RGB is a very wide
FISHEYE624; a naive pinhole undistort either throws away the periphery or stretches the
edges badly. Choose a virtual FOV that keeps the table and both hands in view without
extreme edge stretch (~90° horizontal is a reasonable starting point), then **write those
parameters down and never change them silently** — they are baked into every trained
checkpoint.

---

## 2. Deliverable 1 — the depth store

One file per recording, self-describing.

```
depth/<dataset>/<recording>.h5
  /depth        (N, H, W)  uint16   millimetres in the POLICY FRAME.  0 = invalid
  /depth_valid  (N, H, W)  uint8    1 = measured, 0 = invalid/filled
  /demo_id      (N,)       int32    demo index within the source HDF5
  /frame_idx    (N,)       int32    row index within that demo
  /ts_ns        (N,)       int64    source timestamp (units stated in attrs)
```

**Attributes (required):**

```
source_file, source_type, model {fastfs|fs}, ckpt, created, git_commit
policy_frame_id           # must match the calibration bundle exactly
image_size                # [W, H] of the stored depth
K                         # 3x3 virtual pinhole intrinsics of the POLICY FRAME
depth_units = "mm"
min_depth_m, max_depth_m
fill_policy               # how gaps were filled, see 2.2
```

### 2.1 Hard requirements

- **Metric millimetres, uint16, `0` = invalid.** Agreed with your proposal — exact integers,
  natural sentinel, ~4× smaller than float32. 1 mm quantisation is far below Fast-FS
  accuracy, so it is lossless in any way that matters.
- **`demo_id` AND `frame_idx`, not a flat index.** Teleop HDF5s contain *many demos per
  file*. A flat frame index is ambiguous and cannot be joined back. (Your existing
  `<recording>/demo_N/*.npy` layout already has this — please keep it in the packed schema.)
- **Every RGB frame must have a row.** See below — this is the biggest issue today.

### 2.2 Density — the current blocker

I measured your existing teleop store:

> **93.3% coverage — 2,397 of 2,570 frames. Every demo has gaps, 2 to 18 frames each.**
> The `meta.json` `ts` map skips rows (`0,1,2,3,5,...` — no 4, 14, 24, 46).

Training feeds a fixed-shape `[B, 4, H, W]` RGBD tensor, so **a frame with no depth cannot
be batched.** Sparse depth is not consumable.

**Request:** emit a row for *every* source frame. Where stereo produced nothing:

- fill by **temporal interpolation** from neighbouring valid frames (depth is smooth at 10 Hz), and
- set `depth_valid = 0` for those pixels/frames, and
- record the method in the `fill_policy` attribute.

Also worth telling us *why* the frames are missing (timestamp matching tolerance?), since
fixing the cause is better than filling.

### 2.3 Resolution

Store at the **native policy-frame resolution** — do not pre-downsample to 224. Training
resizes as needed, and keeping native means we can change the training resolution without
recomputing depth.

**When anyone downsamples depth, use NEAREST-neighbour, never bilinear/area.** Averaging
across a depth discontinuity invents "flying pixels" at object edges — exactly where the
gripper has to close.

---

## 3. Deliverable 2 — the calibration bundle

One JSON, the frozen definition of the policy frame. This is what makes the store usable by
someone who has never read your code.

```json
{
  "policy_frame_id": "aria_rgb_rect_v1",
  "device_serial":   "<Aria serial these params belong to>",
  "policy_frame": {
    "model": "pinhole",
    "image_size": [W, H],
    "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
    "fov_deg_h": 90.0
  },
  "rgb_native":  { "model": "FISHEYE624", "image_size": [2016,1512],
                   "K": [[...]], "distortion": [12 coeffs] },
  "slam_left":   { "model": "...", "K": [[...]], "distortion": [...] },
  "slam_right":  { "model": "...", "K": [[...]], "distortion": [...] },
  "T_device_rgb":       [4x4],
  "T_device_slamleft":  [4x4],
  "T_slamleft_slamright": [4x4],
  "baseline_m": 0.1344,
  "T_slamrect_policy":  [4x4],
  "notes": "how the virtual camera was chosen"
}
```

**Critical:** ship the **exact undistort maps or the exact code** that produced the policy
frame — not just the parameters. Train and deploy must apply a *bit-identical* transform.
A subtly different remap at inference is the kind of bug that silently costs accuracy and
takes days to find. (We already lost time to a channel-order mismatch of exactly this type.)

---

## 4. Deliverable 3 — the per-demo manifest

This is the "sort / classify the training data" piece. One JSON per dataset, one row per demo.

```json
{
  "dataset": "0726_teleop_pick_and_place",
  "policy_frame_id": "aria_rgb_rect_v1",
  "demos": [
    {
      "source_file": "fix_head_base_20260726_154438.hdf5",
      "demo_id": 3,
      "n_frames": 100,
      "depth_frames": 100,
      "depth_coverage": 1.0,
      "stereo_available": true,
      "qc": { "black_frames": 0, "longest_freeze": 1, "verdict": "ok" }
    }
  ]
}
```

The dataset builder filters on this instead of hard-coding demo lists. Please include
`stereo_available` — **two 0726 sessions are missing one half of the front stereo pair**
(`134055` has no `slam-front-left`, `134934` has no `slam-front-right`), so 9 demos can
never have stereo depth and must be flagged, not silently skipped.

For reference, our current QC verdicts on the RGB side (drop = genuine camera-stream freeze,
longest contiguous freeze ≥ 16 frames):

| session | drop |
|---|---|
| `20260726_131558` | demo_0, demo_1, demo_12, demo_13 |
| `20260726_134934` | demo_1 |

All other demos have a longest freeze of exactly 1 frame (normal jitter). Current training
set = **72 demos / 8,240 frames** (28 from 0724 + 44 from 0726).

---

## 5. Constraints on our side (FYI — these shape the above)

The model is fed from **LeRobot parquet**, not from HDF5 or `.npy`:

```
source HDF5 -> robomimic_hd5.py -> LeRobot parquet -> FolderRLDBDataset
            -> DataSchematic -> HPT -> cat([rgb, depth]) -> [B,4,H,W] -> encoder
```

Two non-obvious traps we already hit:

1. **Never name a depth key with `img`, `image`, or `rgb` in it.** Key routing is substring
   matched; the image writer forces `(x*255).astype(uint8)` and would destroy metric depth.
   We will call it `aria_depth`.
2. **Depth is joined at the HDF5 merge step**, where `(source_file, demo_id)` is still known.
   Our merged dataset renumbers demos to `demo_0..71`, so a later join would rely on
   build-script ordering. This is why the manifest keys matter.

---

## 6. What we do NOT need

- **Do not store point clouds.** The encoder takes a *depth map* and lifts it internally,
  sampling image features and XYZ at the same pixels. An unstructured `(N,3)` cloud breaks
  that correspondence and can't be fed. Clouds should stay derived, as you proposed.
- **Do not pre-crop or pre-normalise depth.** Ship metric millimetres; we handle the rest.
- Archive reformatting of the *human* datasets (`fpp`, `nav-pouring`) is not on our critical
  path — the robot policy trains on teleop only.

---

## 7. Priority

From the training side, in order:

1. **Freeze the policy frame + calibration bundle** (§1, §3) — everything else depends on it.
2. **Depth for 0726** — 35 stereo-capable demos, currently **0% computed**. This is 69% of
   the training set.
3. **Fix / fill the 6.7% gaps** in the 0724 store (§2.2).
4. **Manifest** (§4).
5. Store reformatting / compression — good hygiene, but it does not unblock training. Worth
   doing *once*, after the schema above is agreed, rather than converting twice.

---

## 8. Open questions for you

1. Does the factory calibration cover **the same Aria device** as the robot head camera?
   (There are `.vrs` files under `nav_pouring_test/` and `pick_place_aria_vrs/` — do the
   serials match?)
2. What virtual FOV do you recommend for the policy frame, given how much periphery the
   fisheye undistort discards?
3. What causes the missing depth frames — timestamp tolerance, or stereo failure?
4. Can you emit `depth_valid` per pixel, or only per frame?
5. Do you already have per-frame **hand/wrist pose** (from registration or MPS)? We will
   need `T_cam_hand` per frame for hand-frame canonicalisation — but that is a separate
   request, not part of this spec.
