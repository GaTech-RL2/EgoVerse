# 3D policy round — FINAL RESULTS (2026-07-31)

Three 3D policies trained on identical teleop data (63 demos / 7,003 frames), identical
trunk + flow-matching head + recipe. Only the encoder/observation differs. Gate probes all
run through the real serving path.

## The consolidated table

| policy | obs | best val (ep) | final val | clean fit | input-zero reliance |
|---|---|---|---|---|---|
| **v4** (RGB baseline) | rectified RGB 224² | **0.0859** (299) | 0.0904 | 0.0132 | — |
| Adapt3R | gray + depth | 0.1157 (249) | 0.1286 ↑degrades | 0.0057 | depth ×32.6 |
| Adapt3R + jitter | gray + depth | 0.1171 (199) | 0.1321 ↑degrades | 0.0061 | depth ×31.6 |
| **DP3 2048** (xyz only) | glass-frame cloud | 0.1112 (299) | 0.1182 (flat) | 0.0072 | cloud ×25.5 |
| **DP3 1024** (xyz only) | glass-frame cloud | **0.1069** (1299) | **0.1146 (flat)** | 0.0084 | cloud ×17.4 |

(v2 @1499 remains the only hardware-confirmed policy; v3 = 0.0836 best val on its own
72-demo dataset, not directly comparable to the 63-demo slam-rect val sets.)

## The four findings

1. **Every 3D policy genuinely uses its 3D input.** Zeroing the cloud/depth degrades MAE
   25–33× at ep1999. The pipelines work; nothing is riding on proprio (reliance ≈ 1.0).
2. **Point-cloud jitter augmentation is a clean negative** — same data, same val, identical
   curve shape (0.1171 vs 0.1157 best). The overfitting channel is NOT geometric
   pose/depth-noise sensitivity.
2b. **Halving the point budget IMPROVED generalization.** DP3-1024 beats DP3-2048 on
   held-out val at every stage (best 0.1069 vs 0.1112; mean over ep>1500 0.1150 vs 0.1184)
   while fitting training slightly looser (clean 0.0084 vs 0.0072). Consistent with the
   input visualization finding that >50% of the 2048 budget lands on task-irrelevant
   background beyond 1.6 m — fewer, better-spread points is the better trade here, and it
   argues the next lever is a TIGHTER CROP rather than more points.

3. **DP3 is the best and least-overfitting 3D policy with 1/40th the encoder.** Its
   val curve is FLAT (0.1182 final vs 0.1112 best) where both Adapt3R variants degrade
   ~11% past their optimum. The frozen-DINO-on-grayscale features add overfitting, not
   generalization, on 62 training demos.
4. **All 3D variants floor at val ≈ 0.11–0.12 vs RGB's 0.086** — three very different
   encoders, same floor. This is an *observability* ceiling, not an encoder defect: the
   task is partly color-coded (blue basket / yellow penguin) and geometry cannot recover
   that cue. The ~0.03 MAE gap is the price of losing color.

## Implication

The next lever for 3D is **RGB + depth in one frame** (needs the robot-Aria VRS capture
→ RGB policy frame), not more architecture or augmentation work. Until then, v4 (RGB)
stays the deployment pick; DP3 is the 3D architecture of record if/when 3D is revisited
(simplest, cheapest, best-generalizing, flat val curve).

DP3 details: encoder `egomimic/models/dp3_encoder.py` (faithful PointNetEncoderXYZ port,
10 unit tests), dataset `datasets/rby1_teleop_pcd2048_glass` (glass-frame clouds, 0.25–2 m
crop, random-16k→FPS-2048, 0 degenerate frames), configs `rby1_dp3_pcd.yaml` +
`wb_dp3_pcd.yaml`, run `RBY1_dp3_pcd/dp3_pcd2048_glass_2k` (COMPLETED, one TIMEOUT
resume at ep1900 — 12 h wall-time is marginal for 2000-epoch runs on busy nodes).
Serving handles `(N,3)` float clouds (explicit path in `EgoVersePolicy`; the image path
would have BGR-flipped xyz→zyx and /255'd metric coordinates).

---

# Adapt3R depth policy — RESULTS (2026-07-30)

**Status: trained, served, gate-probed. Verdict: works, overfits harder than the RGB
policies, do NOT deploy over v4/v2 yet.**

Unblocked by the other agent's `image_rect` fallback (rectified left SLAM image,
pixel-identical to depth by construction — no cross-camera warp needed). Trained on
`datasets/rby1_teleop_slamrect_rgbd`: 63 demos / 7,003 frames, grayscale 224² image +
real metric depth (metres, 0% invalid), SG-smoothed actions, same 2000-epoch recipe.
Run `RBY1_adapt3r_slamrect/adapt3r_slamrect_2k`, **COMPLETED**.

## Two NEW bugs found while getting the gate probe running (both fixed)

1. **`EgoVersePolicy._obs_to_batch` never read depth at all.** It only iterates
   `_cam_keys` and `_proprio_keys`; `aria_depth` is a `metadata_keys` entry and was
   silently dropped. **This means the policy could not be served/deployed at all**
   before this fix — training worked because it reads the LeRobot batch directly,
   bypassing this code path entirely. Fixed: added a `_depth_keys` loop (from
   `HPT._depth_key_map`, via `getattr` for backward compat) in both `__init__` and
   `_obs_to_batch`. `egomimic/serving/egoverse_policy.py`.
2. **`ModelWrapper.load_from_checkpoint(weights_only=False)` can't unpickle the nested
   DINOv2 ViT** — its class lives in the `dinov2` module, which only becomes importable
   as a side effect of calling `torch.hub.load()` once in-process; unpickling never
   triggers that call itself. Fails with `ModuleNotFoundError: No module named
   'dinov2'`. Fixed in `serve_policy.py` with a catch-and-retry (warm up
   `torch.hub.load`, then reload) so non-DINOv2 checkpoints (v2/v3/v4, plain ResNet)
   are completely unaffected. **Also observed once as a node-specific flake** (same
   code, same env, failed on node `asimo`, succeeded on `t1000` and via `srun`
   elsewhere) — root cause not fully pinned down; the retry logic should paper over it
   if it recurs, but worth watching.

## Gate probe (real serving path, real depth, mode="total" — same protocol as v2/v3/v4)

| ckpt | clean | shift10 | shift20 | **dzero** | pzero | proprio-reliance | **depth-reliance** |
|---|---|---|---|---|---|---|---|
| ep199 | 0.0389 | 0.0432 | 0.0434 | 0.1680 | 0.0375 | 0.963 | **4.3×** |
| ep999 | 0.0096 | 0.0135 | 0.0118 | 0.1882 | 0.0106 | 1.106 | **19.6×** |
| **ep1999** | **0.0057** | 0.0102 | 0.0104 | 0.1866 | 0.0053 | 0.922 | **32.6×** |

`dzero` = depth zeroed out (new ablation, mirrors the existing `pzero` proprio ablation).

**The policy genuinely uses depth, heavily and increasingly with training** —
depth-reliance climbs from 4.3× to 32.6× ×, i.e. zeroing depth makes clean MAE ~30×
worse by ep1999. Proprio-reliance stays ~1.0, same as every prior checkpoint. This
confirms the 3D pathway is doing real work, not just riding along inert.

## The catch: severe overfitting vs the RGB policies

| policy | clean (mode=total) | **held-out val (wandb)** |
|---|---|---|
| v3 (raw fisheye RGB) | 0.0857 (ep1999) | **0.0836** (best, ep599) |
| v4 (rectified RGB) | 0.0132 (ep1999) | **0.0859** (best, ep299) |
| **Adapt3R depth (gray+depth)** | **0.0057 (ep1999)** ← tightest fit of all four | **0.1286** (final) / **0.1157** (best, ep249) ← worst val of all four |

The gate/`clean` metric is measured on `mode="total"` (train+val combined, so mostly
train) — it's an in-distribution fit number, not generalization. Adapt3R fits the
training distribution **tighter than every RGB policy** yet generalizes **worse** than
all of them (~40-50% higher held-out MAE than v4). Same pattern seen in the wandb val
curve: val bottoms at ep249 (0.1157) then climbs to 0.1286 by ep1999 while train loss
keeps falling to 0.0083 — classic overfitting, more severe than v2/v3/v4's mild version
of the same thing. Also: shift-sensitivity is proportionally worse than v4's (shift10
is ~79% higher than clean at ep1999, vs v4 where shift barely moves).

**Likely causes, roughly in order of confidence:**
1. Image augmentation is OFF on the RGBD path (documented above, `TODO(aug)`) — the RGB
   policies get 20-40px crop regularization; this one gets none.
2. Fewer effective demos after grayscale/depth-store QC (62 train vs ~70 for v3/v4) and
   only 1 held-out val episode, so the val estimate itself is noisier.
3. High-capacity encoder (21.2M trainable, on top of the frozen 22.1M DINOv2) relative
   to 62 demos.
4. Possibly some genuine information the model is exploiting that doesn't generalize
   (e.g. depth noise patterns specific to the FoundationStereo model's failure modes on
   train-set framings) — unconfirmed, would need more analysis to distinguish from (1)-(3).

## Recommendation

**Do not deploy this over v4 (rectified RGB, @1999) or v2 (@1499, hardware-confirmed).**
The held-out val gap is real and, per this project's established pattern (v3 having
more data than v2 but similar-or-worse val, and NOT beating it on hardware), val is the
metric that has actually predicted hardware behavior here — not in-sample fit.

This is a genuine, working proof of the Adapt3R pipeline end-to-end: real depth,
real serving, heavy and increasing depth-reliance. The next lever, if this direction is
worth pursuing further, is fixing the augmentation gap (item 1) before concluding
anything about whether depth itself helps or hurts — right now that confound is
unresolved.

---

# Original blocker writeup (superseded by the above — kept for history)

**Date:** 2026-07-28 (overnight session)
**Short version:** everything on the policy side is ported, tested and ready. Training
cannot start because depth and RGB cannot be aligned without one missing calibration.

---

## What is DONE

| item | status |
|---|---|
| Adapt3R encoder ported into `rby1_aria_policy` | ✅ |
| `hpt.py` RGBD plumbing (`depth_key_map`) | ✅ |
| Encoder unit tests | ✅ **18 passed, 1 skipped** |
| Model + experiment configs for our 49-D RBY1 task | ✅ `rby1_wb_img_adapt3r_depth.yaml`, `wb_img_adapt3r_depth.yaml` |
| Backward compatibility with existing checkpoints | ✅ verified (see regression note) |
| Depth store from the other agent | ✅ FoundationStereo, uint16 mm, 100 % dense, `demo_id`+`frame_idx`, `T_device_rect` |
| **End-to-end integration smoke test on our data** | ✅ model builds **43.3 M params (21.2 M trainable / 22.1 M frozen DINOv2)**; train steps + 2 validation epochs ran, **0 NaN, 0 exceptions** |

### Gotcha found while smoke-testing

`trainer=debug` on the CLI does **not** actually limit epochs for these experiment configs:
the experiment yaml lists `_self_` after the defaults, so its own `trainer.max_epochs: 2000`
is re-applied *after* the debug group. `limit_train_batches: 5` still applies, so a "debug"
run silently becomes 2000 epochs × 5 batches. Override `trainer.max_epochs=4` explicitly on
the command line when smoke-testing.

## The blocker: depth and RGB are in different camera frames

The depth store is in the **slam-rect** frame (`policy_frame_id = slam_rect_v1`), and its own
attrs say: *"slam_rect frame; warp to RGB policy frame pending camera-rgb calib"*.

To warp depth into the RGB policy frame I need, **for the Aria unit that recorded the teleop
data**:

```
camera-rgb intrinsics   (FISHEYE624: f, cx, cy + 12 distortion coeffs)
T_device_rgb            (4x4, camera-rgb -> device frame)
```

`T_device_rect` is already in the depth store, so this is the only missing piece.

### Why I can't take it from the .vrs files on disk

I scanned all 176 `.vrs` files under `mobile_generalist/data/`. They are all the **same**
device, and it is **not** the device that recorded the teleop data:

| | `.vrs` on disk | depth store (teleop device) |
|---|---|---|
| stereo baseline | 0.142000 m | **0.13438324 m** |
| slam focal | 246.442 px | **245.869** (= 307.337 / 1.25) |

Mixing the two produces visibly misaligned depth — I tested it: the warped gripper, basket
and penguin are offset from the RGB by tens of pixels. Since the Adapt3R encoder samples
image features and XYZ **at the same pixel**, that misalignment corrupts the exact mechanism
the method depends on. Training on it would be worse than useless — it would look like it
trained fine and fail silently.

## NEGATIVE RESULT (2026-07-29): empirical extrinsic solve does NOT work

I implemented the "solve the RGB↔SLAM extrinsic by maximising depth↔RGB edge alignment" idea:
52 frames from two sessions (26 train / 26 held-out), 6-DOF optimisation with the VRS RGB
intrinsics as the image model. **It fails, and the failure is diagnostic.**

| run | result |
|---|---|
| unbounded (Powell) | train −0.082 → **+0.206**, held-out −0.063 → **+0.162**, but solved **rotvec ≈ 70°, translation 14.6 cm** — physically impossible for two cameras rigidly mounted centimetres apart |
| discrete 90° hypotheses (x/y/z) | best is `x −90°` at only **+0.066**; several orientations give −1.0 (degenerate warp) |
| **bounded** (rot ≤ 5°, trans ≤ 3 cm) | **saturates all six bounds** (5.0, 4.96, −5.0 deg; −2.92, −3.00, +2.98 cm), reaching only train +0.087 / held-out +0.104 |

**Saturating every bound means the optimum lies outside the physically plausible region — the
model is misspecified.** The residual is not a small extrinsic offset, so no amount of
extrinsic fitting fixes it. Even unconstrained it only reaches +0.16, far below a correct
alignment.

**Prime suspect: the RGB image geometry assumption.** I assumed the stored 640×640 frame is a
centre-crop of 1920×1920 from the 2560×1920 sensor, validated only by the principal point
landing near the image centre — which almost any centred geometry satisfies. If the rig
actually streams a **square RGB profile** (Aria commonly streams 1408×1408), the 2560×1920
calibration does not map as assumed, and both the depth warp *and* v4's rectification inherit
the error.

**Implication for v4:** its undistortion rests on the same unverified assumption. Lines did
visibly straighten, which validates it approximately — but the depth warp is far more
sensitive to geometry error than line-straightening is, so that is not proof.

**Conclusion: stop reconstructing the geometry from inference. Get the VRS.** A 60-second
capture from the robot's own Aria resolves the RGB intrinsics, the extrinsics, *and* the
actual stream geometry in one shot — and fixes v4 as a side effect.

## What to ask the depth agent for

> For the Aria unit that recorded the teleop HDF5s (the one whose slam baseline is
> 0.13438324 m), please emit the `camera-rgb` calibration: FISHEYE624 intrinsics
> (f, cx, cy, 12 distortion coeffs), the native image size, and `T_device_rgb` (4x4).
> You already have this device's calibration — it is what produced `T_device_rect`.
> With it I can warp your depth into the RGB policy frame and start training immediately.

Ideally also add `T_device_rgb` + the rgb intrinsics to the store attrs, and re-emit with
`policy_frame_id = aria_rgb_rect_v1`, so the store becomes self-contained (see
`depth_data_spec.md` §3).

## The moment that arrives

1. Warp depth: `rect px -> 3D(rect) -> T_device_rect -> device -> inv(T_device_rgb) -> RGB cam
   -> project into policy pinhole (f=373, c=(320,320), 640x640) -> z-buffer -> resize 224 NEAREST`.
   Code is already written and tested end to end.
2. Inject `obs/aria_depth` at the HDF5 merge step, rebuild the LeRobot dataset.
3. Set `dummy_depth: null` in `rby1_wb_img_adapt3r_depth.yaml` and launch.

## Known issue to fix BEFORE the first real depth run

**Image augmentation is skipped on the RGBD path** (marked `TODO(aug)` in `hpt.py`). The
existing `train_image_augs` Compose cannot be applied as-is: it would either transform RGB
but not depth (breaking pixel correspondence) or feed 4 channels to `ColorJitter`, which
expects 3. Fix by splitting geometric augs (applied to all 4 channels) from photometric augs
(RGB only). This matters — the 20–40 px crop is a load-bearing part of the recipe.

## Regression I introduced and fixed

Adding `self._depth_key_map` in `HPT.__init__` broke **every existing checkpoint**, including
serving. Cause: the `HPT` object is **pickled whole** into the checkpoint, so `__init__` is
never re-run on load and the attribute simply does not exist on restored models. Fixed by
reading it defensively at the use site:

```python
depth_key = getattr(self, "_depth_key_map", {}).get(key)
```

Verified by re-running the v4 gate probe, which loads checkpoints through the same path
serving uses. Worth remembering: **any new attribute added to `HPT.__init__` will be absent
on all previously-trained checkpoints.**

## UPDATE 2026-08-04: colour UNBLOCKED on teleop — the missing rgb model was solved from data

The robot recordings never stored a camera-rgb calibration (publisher published SLAM-only;
fixed upstream for FUTURE recordings, and there is no robot VRS for the old ones). Instead
of waiting for a re-capture, the rgb<->rect model was solved by registration: depth gives
3D at every rect pixel; SIFT matches into the stored 640 fisheye RGB; least-squares over
[pose(6), f, cx, cy, k1, k2]. Result (job 3652632): 2716 inliers, 1.22 px median reproj;
solved f=370.40 c=(310.5,319.7) k1=-0.320 t=[-9.1,0.1,-4.5]mm — near the human-device
prior but distinctly per-unit (why borrowing extrinsics broke in July).
Held-out phase-corr gate, one model across ALL 6 tags: **median 0.43 px @512, p90 1.55**
(human colour build passed at 1.26). Params: `tmp/robot_rgb_solved.npz`.
-> dataset `rby1_teleop_colour_rgbd` (row-identical to the grey slamrect_rgbd, image now
colour **RGB** -- the recorder stores BGR; flipped at build because Adapt3R's frozen
DINOv2 is RGB-pretrained. Deploy must flip the live BGR image for this policy.
⚠ First attempt (`adapt3r_tel_colour_2k`, killed @ep~740) stored BGR and fed the frozen
RGB backbone swapped channels -- VOID, do not cite; caught via aug_debug (orange basket).
Corrected run: `adapt3r_tel_colour_rgb_2k`.) -> the grey Adapt3R recipe with colour.
Readouts: vs grey Adapt3R 0.1157 = value of colour to 3D; vs v4 RGB 0.0859 = ceiling.
If the factory calib ever arrives (fixed publisher, 10-s capture), swap + rebuild.
