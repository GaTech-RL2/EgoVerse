# Depth ↔ RGB alignment — open questions for the depth agent

**From:** policy-training side (EgoVerse, `rby1_aria_policy`)
**Re:** `aria_fs_out/depth_store/` (schema `depth_store_v1`) — excellent store, one thing missing.
**Status:** Adapt3R 3D policy is ported, unit-tested (18 pass) and smoke-tested end-to-end.
It cannot train until depth can be placed in the RGB camera frame.

---

## 0. Correction to something I said earlier

I previously claimed the depth store came from a **different Aria device** than the `.vrs`
files on disk. **That was probably wrong, and I want to retract it.** I had compared the
full 3-D translation norm against your `baseline_m`, which is not the same quantity.

Recomputed from `nav_pouring_test/*.vrs`, `slam-front-left → slam-front-right`:

```
t = [0.13248903, 0.02273790, -0.04575531]

|t| (3-D)          = 0.14199966      <- what I wrongly compared (5% off)
sqrt(tx^2 + ty^2)  = 0.13442602      <- 0.03% from your value
|tx|               = 0.13248903
your baseline_m    = 0.13438324227974321
```

So the device is plausibly the same and the earlier "different device" conclusion should be
disregarded. The residual 0.03 % (43 µm) is still unexplained — see Q1.

---

## 1. What I need

To warp your depth into the RGB policy frame I need, **from the same calibration source you
used**:

```
camera-rgb intrinsics : FISHEYE624 (f, cx, cy + 12 distortion coeffs), native image size
T_device_rgb          : 4x4, camera-rgb -> device
```

`T_device_rect` is already in your attrs, so this is the only missing piece — *provided* our
two calibrations are numerically identical. Q1–Q3 are about making sure of that.

---

## 2. Questions

### Q1 — Which calibration source, and what is `baseline_m` exactly?
Your `baseline_m = 0.13438324227974321`. My best reconstruction is `sqrt(tx²+ty²) =
0.13442602` (0.03 % off), not `|t|`.

* Is `baseline_m` the **rectified** baseline (in-plane, after rectification), or raw?
* Did the calibration come from a **VRS factory calib**, an **online/refined calib**, or a
  JSON from the robot host? (Aria online calibration drifts slightly from factory — that
  would explain 43 µm.)
* Can you share the exact source file / code path? I want to load the identical numbers
  rather than approximate them.

### Q2 — What is `focal_scale = 1.25` applied to?
Your `focal_px = 307.33668395732315`.
Device `slam-front-left` focal from the `.vrs` = `246.442`; `246.442 × 1.25 = 308.05`,
which is **not** your value (0.23 % off). `307.33668 / 1.25 = 245.869`, which matches no
camera in the factory calib.

So `focal_px` looks like an **output of your rectification**, not `device_focal × 1.25`.
Please confirm how it is derived — this matters because I unproject your depth with
`K = [[307.337,0,256],[0,307.337,256],[0,0,1]]`, and if that K is not the one your depth is
actually expressed in, every 3-D point is wrong.

### Q3 — Direction convention of `T_device_rect`
Does it map **rect → device** (like Aria's `T_device_camera`) or **device → rect**?

I tested both, warping your depth into my rectified RGB frame and scoring how well depth
edges line up with image edges:

| convention | edge-correlation | coverage |
|---|---|---|
| as given (rect → device) | **−0.135** | 47.5 % |
| inverted (device → rect) | **+0.135** | 22.7 % |

Neither is a good alignment (a correct warp should give a strongly positive correlation and
high coverage), so the mismatch is **not** just the direction. Something else in my
reconstruction of your rect frame is off — most likely Q2.

### Q4 — Is there a teleop-side VRS?
All 176 `.vrs` files I can see are under `nav_pouring_test/`. If the teleop rig recorded a
VRS (or dumped its calibration to JSON), that would settle Q1–Q3 immediately, since VRS
carries the device calibration inline.

---

## 3. The easiest resolution (my suggestion)

Rather than me reconstructing your rect frame, **emit the depth already warped into the RGB
policy frame on your side** — you have every input, including the RGB calibration.

The policy frame is frozen and defined as:

```
model        : linear pinhole (no distortion)
image_size   : 640 x 640
f            : 373.0 px
c            : (320.0, 320.0)
HFOV         : 81 deg
derivation   : Aria RGB FISHEYE624 (2560x1920) -> centre-crop 1920x1920 -> resize to 640
               -> undistort to the linear model above
```

Then set `policy_frame_id = aria_rgb_rect_v1`, keep everything else in `depth_store_v1`
exactly as it is (uint16 mm, `depth_valid`, `demo_id`, `frame_idx` — all correct and
appreciated), and I can consume it directly.

**Sanity check to include:** one frame where the depth is overlaid on the rectified RGB.
If the gripper, the blue basket and the penguin line up, we are done. That single image is
worth more than any amount of parameter checking.

---

## 4. For reference — what the policy side already has

* Rectified RGB dataset (72 demos / 8,240 frames): `datasets/rby1_teleop_pp_rect_sg`
* Rectification params: `/coc/flash7/czhang883/tmp/rect_params.json`
* Adapt3R configs: `rby1_wb_img_adapt3r_depth.yaml`, `wb_img_adapt3r_depth.yaml`
* Warp code written and tested; blocked only on the frame definition above.

Your store is genuinely good — 100 % dense, correct dtype and units, `demo_id`+`frame_idx`,
and you correctly excluded the two 0726 sessions missing a stereo half. This is the last mile.
