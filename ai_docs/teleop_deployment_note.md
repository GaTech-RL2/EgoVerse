# Teleop Round — Real-Robot Rollout Guide (0724 robot manip data)

> Companion to `fpp_deployment_note.md` (that one is the HUMAN-data / full nav+pick+place
> round). THIS guide is the **manipulation-phase-only** policies trained on robot teleop.
> Pull branch `rby1_aria_policy` LATEST before serving.

> **CHECKPOINT RULE (user decision 2026-08-05): always deploy the LATEST checkpoint of a
> run (`epoch=1999`/`last.ckpt`), not the best-val one — a bit of overfitting is fine for
> IL policies. Earlier per-epoch entries below are kept as reference/fallback only.

## ★★★ v4 — RECTIFIED images (newest; NOT yet hardware-tested)

Same recipe as v3, but images are **undistorted** to a linear pinhole camera before
training. Motivation: v3 (72 demos) did not beat v2 (28 demos) on hardware, which points at
a train/deploy image mismatch rather than a data shortage. Dataset
`datasets/rby1_teleop_pp_rect_sg` (72 demos / 8,240 frames).

| tag | checkpoint | clean MAE | reliance | val all-MAE |
|---|---|---|---|---|
| **v4-A (primary)** | `logs/RBY1_wb_img_tel_v4/wb_img_pickplace_v4_rect_2k/checkpoints/epoch_epoch=1999.ckpt` | **0.0132** | ×1.01 | 0.0904 |
| v4-B | `.../epoch_epoch=1199.ckpt` | 0.0165 | ×0.99 | 0.0909 |
| v4-C (best val) | `.../epoch_epoch=299.ckpt` | 0.0311 | ×0.99 | **0.0859** |

**⚠ THE ROBOT MUST APPLY THE IDENTICAL RECTIFICATION, or v4 will be WORSE than v3.**

```
policy frame : linear pinhole, 640x640, f = 373 px, c = (320, 320), HFOV 81 deg
source       : each device undistorts ITS OWN fisheye (FISHEYE624) into that target
               recorded frame = centre-crop 1920x1920 of the 2560x1920 sensor, resized to 640
then         : resize 640 -> 224 (INTER_AREA), send BGR uint8 numpy
```
Exact params: `/coc/flash7/czhang883/tmp/rect_params.json` and the dataset HDF5 attrs.

**⚠ CRITICAL (found 2026-07-28): use THESE maps, NOT the robot device's own calibration.**
v4's undistortion was computed with the `camera-rgb` calibration from the `nav_pouring_test`
`.vrs`, which is a **different Aria unit** than the one that recorded the teleop data
(confirmed bit-exact: teleop slam FL focal 245.86935 / principal (253.335, 257.135) vs VRS
246.44195 / (256.656, 255.620)). The teleop HDF5s carry SLAM calib in
`obs/slam.attrs["calib"]` but **no `camera-rgb` entry**, so no correct RGB calib exists yet.

Consequence: v4's rectification is slightly wrong in absolute terms, but **identically** wrong
on every training frame. It therefore remains self-consistent *only if deployment applies the
exact same VRS-derived transform*. If the robot undistorts with its own (correct) device
calib, train and deploy diverge — re-introducing precisely the mismatch v4 exists to remove.

The proper fix is a 60-second VRS capture from the robot's own Aria; then re-rectify and
retrain, after which "each device uses its own calib" becomes the correct rule.

**Honest note:** rectification did **not** improve offline val (v4 best 0.0859 vs v3 0.0836)
— which is expected, because offline replay cannot see a train/deploy mismatch. The whole
point of v4 is the *hardware* gap. If v4 also disappoints on the robot, the mismatch
hypothesis is wrong and the next suspect is the action/base integration, not the image.

---

## ★★ v3 (72 demos, raw fisheye — hardware-tested, did NOT beat v2)

Same validated recipe as v2 (SG-smoothed actions, proprio dropout 0.9, 2000 ep, ResNet-18)
with **crop widened to 20–40 px** and trained on the **merged 0724+0726 set: 72 demos /
8,240 frames** (3.2× v2's data), 70 train / 2 val. Dataset `datasets/rby1_teleop_pp_0724_0726_sg`.

| tag | checkpoint | clean MAE¹ | reliance | val all-MAE² |
|---|---|---|---|---|
| **v3-A (primary)** | `logs/RBY1_wb_img_tel_v3/wb_img_pickplace_v3_72demo_2k/checkpoints/epoch_epoch=1999.ckpt` | **0.0134** | ×1.03 | 0.0896 |
| v3-B (hedge) | `.../epoch_epoch=1199.ckpt` | 0.0166 | ×1.06 | 0.0879 |
| v3-C (best held-out) | `.../epoch_epoch=599.ckpt` | 0.0223 | ×1.03 | **0.0836** |

¹ on SG-smoothed targets, not comparable to round-1 numbers. ² only 2 held-out episodes
(168 frames) — noisy. All three are vision-driven (reliance ≈ ×1.0) and image-shift stable.

**Why 1999 despite val being lowest at 599:** the two metrics disagree because val measures
generalisation to unseen episodes while clean-MAE measures fit. Deployment here is the *same*
lab, table and objects as training — the deployment distribution IS the training distribution
— so fit matters more than episode-generalisation, and the val spread (0.0836→0.0896) is
small and measured on only 2 episodes. If 1999 looks over-tight / brittle on hardware, drop
to 599, which is the genuinely best-generalising checkpoint.

**v2 remains the known-good fallback** (hardware validation was at `epoch=1499`; per the
latest-ckpt rule deploy `epoch=1999`, keep 1499 in your pocket) — v2 is the only run
confirmed working on the robot. Do not retire it until v3 proves out.

Serve v3-A:
```bash
python egomimic/scripts/serve_policy.py \
  --checkpoint logs/RBY1_wb_img_tel_v3/wb_img_pickplace_v3_72demo_2k/checkpoints/epoch_epoch=1999.ckpt --port 8000
```
Obs/action contract (§3–§5) unchanged.

### Preflight findings (2026-07-27, verified by actually serving v3-A)

1. **Serving deps were MISSING from the `emimic` venv** — `serve_policy.py` died at import.
   Installed `msgpack-numpy==0.4.8` and `websockets==16.1.1`. If you serve from a different
   host, verify both are present there first.
2. **Server metadata confirmed good:**
   `embodiment=rby1, action_horizon=32, action_dim=49`. (`camera_keys` prints `front_img_1`
   four times and `proprio_keys` includes `task_id` — both are cosmetic; `task_id` has no
   norm stats and is skipped.)
3. **`robot0_joint_pos` must be 22-D no-wheel.** Sending the raw 26-D vector fails with
   `The size of tensor a (26) must match the size of tensor b (22)`. The dataset contains
   both `obs.robot0_joint_pos` (26) and `obs.robot0_joint_pos_no_wheel` (22) — the model
   uses the latter.
4. **`test_serve_policy_client.py` is currently BROKEN for this model** — it hard-codes
   `obs.robot0_joint_pos` (26-D), so the §6 offline replay will crash. Not a model bug.
5. **Send the image as a numpy array, not a torch tensor.** `_obs_to_batch` only applies the
   BGR→RGB flip on the **numpy** path (`_prepare_image(..., bgr_to_rgb=True)`); a torch
   tensor is passed through unflipped and would silently be channel-swapped.

---

## ★ v2 (hardware-VALIDATED — the known-good fallback)

Round v2 addressed the user's post-rollout notes: **proprio dropout 0.9** (up from 0.5),
**Savitzky-Golay-smoothed action targets** (window 7 / poly 3 — jerk cut ~54%), **only 1
held-out val episode** (max training data), **2000 epochs**, ResNet-18. Trained on the
smoothed dataset `datasets/rby1_teleop_pp_0724_sg`.

| tag | checkpoint | clean MAE¹ | reliance | why |
|---|---|---|---|---|
| **v2-A (primary — latest)** | `logs/RBY1_wb_img_tel_v2/wb_img_pickplace_v2_2k/checkpoints/epoch_epoch=1999.ckpt` | 0.0052 | **×0.96** | purest vision-driving + tight fit; smoothest actions |
| v2-B (tightest fit) | `.../epoch_epoch=1999.ckpt` | 0.0038 | ×1.16 | best raw fit, leans very slightly more on proprio |
| v2-C | `.../epoch_epoch=999.ckpt` | 0.0083 | ×1.00 | earlier, if 1499 over-fits on HW |

¹ clean MAE is measured on the **SG-smoothed** targets, so it is much lower than the round-1
numbers below and NOT comparable to them; the reliance ratio (pzero÷clean) IS comparable and
is ×1.0 = pure vision. Image-shift probes (10/20 px) leave predictions unchanged.

Serve v2-A:
```bash
python egomimic/scripts/serve_policy.py \
  --checkpoint logs/RBY1_wb_img_tel_v2/wb_img_pickplace_v2_2k/checkpoints/epoch_epoch=1999.ckpt --port 8000
```
Everything in §3 (obs contract, **BGR**), §4 (action contract), §5 (start AT the table) below
still applies unchanged. The actions are now smoother, so expect less jerky execution.

---
### Round-1 checkpoints (superseded by v2 — kept for reference)

## 0. What these policies are (read first)

Trained on **28 robot teleoperation demos** (0724 pick-and-place, penguin → blue basket),
~9 s each, whole-body 49-D action. This is the **near-table OOD fix**: every human-data
policy failed the near-table hardware test because the close-up hand/object view was out of
distribution. This data *is* that near-table distribution, recorded on the robot itself.

Consequences for your rollout:
- **Start AT / right next to the table**, gripper roughly in the teleop start pose. These are
  NOT nav policies — base motion is small coordinating motion only (net ~0.02 m, path ~0.14 m,
  yaw path ~0.21 rad per demo). Do not expect a long approach drive.
- **Perception matches deployment exactly** — robot camera, robot optics. No appearance or
  perspective gap (that is why dropout was only 0.5 and crop only 10–20 px this round). The
  live image is the training distribution, so the image path is trustworthy here.
- **All three are vision-driven** (proprio-zero changes predictions by ~nothing, reliance
  ×1.0 — see gate table). Sending real proprio is fine; zeroing it is available as a debug
  isolation test on any of the three.

## 1. Checkpoints to roll out (all under repo root `/coc/flash7/czhang883/Documents/EgoVerse/`)

| tag | checkpoint | clean MAE | reliance | why |
|---|---|---|---|---|
| **A (primary)** | `logs/rby1_teleop_pp_0724/tel_resnet_1k/checkpoints/epoch_epoch=999.ckpt` | 0.0191 | ×1.01 | best fit, simplest (plain ResNet + transformer) |
| **B** | `logs/rby1_teleop_pp_0724_wam3/tel_wam3_1k/checkpoints/epoch_epoch=899.ckpt` | 0.0208 | ×1.02 | world-model head; best held-out val (0.0760) |
| **C (optional)** | `logs/rby1_teleop_pp_0724/tel_d3conv_1k/checkpoints/epoch_epoch=999.ckpt` | 0.0228 | ×1.06 | frozen DINOv3-S + ConvNeck (vision-only encoder) |

- **A vs B** is the same world-model question as the human round (identical recipe otherwise) —
  now on data whose perception matches the robot. A is the safe default; roll B to see if the
  aux world head helps grasping on hardware.
- All three offline-equivalent within ~2 %; **reliance and hardware behavior decide**, not val.

## 2. Dataset (for the pre-flight replay sanity check + scp)

`datasets/rby1_teleop_pp_0724` (28 eps / 2 570 frames, LeRobot format, 195 MB).
Images are 224² stored **RGB** (built from cv2-BGR robot frames with a BGR→RGB flip);
the live robot camera still sends cv2-native **BGR** and the server flips it — see §4.

scp it to the robot host if you want to replay there:
```bash
scp -r <thishost>:/coc/flash7/czhang883/Documents/EgoVerse/datasets/rby1_teleop_pp_0724 ./
```

## 3. Serve

```bash
cd /coc/flash7/czhang883/Documents/EgoVerse   # branch rby1_aria_policy, latest
git pull && source emimic/bin/activate
# NVS3D_DIR is NOT needed for A/B/C (d3conv uses standard timm DINOv3, cached in HF cache).

python egomimic/scripts/serve_policy.py --checkpoint logs/rby1_teleop_pp_0724/tel_resnet_1k/checkpoints/epoch_epoch=999.ckpt      --port 8000   # A
python egomimic/scripts/serve_policy.py --checkpoint logs/rby1_teleop_pp_0724_wam3/tel_wam3_1k/checkpoints/epoch_epoch=899.ckpt   --port 8001   # B
python egomimic/scripts/serve_policy.py --checkpoint logs/rby1_teleop_pp_0724/tel_d3conv_1k/checkpoints/epoch_epoch=999.ckpt      --port 8002   # C
```
On-connect metadata must read embodiment **rby1 / action_dim 49 / horizon 32**.
The wam3 world-model aux head is train-only — it is silent at inference, no extra output.

## 4. Obs contract (send exactly this)

- `front_img_1`: 224×224 uint8 **BGR** (cv2-native — the server flips to RGB). Raw robot
  frame resized square; no rectification, no center crop.
- `robot0_joint_pos`: 22-D no-wheel, order **[torso 6, r_arm 7, l_arm 7, head 2]**.
- `hand_left_qpos`: 12 · `hand_right_qpos`: 12.
- Send real proprio. Debug test (any of A/B/C): send all-zero proprio — offline this changes
  nothing (×1.0); if live behavior changes a lot under zeroed proprio, something else is wrong
  (capture the dump).

## 5. Action contract

`(1, 32, 49)` un-normalized @10 Hz. Layout:
`base[0:3]` **per-step deltas** (integrate by plain cumsum in the frame-0 heading, NO yaw
rotation — but base motion is small here), `torso[3:9]`, `head[9:11]`, `l_arm[11:18]`,
`r_arm[18:25]`, `l_hand[25:37]`, `r_hand[37:49]` = absolute joint targets.

## 6. Pre-robot sanity (5 min, once)

Serve A, then replay against the training corpus:
```bash
python egomimic/scripts/test_serve_policy_client.py \
  --episode-idx 0 --max-steps 30 --trajectory \
  --dataset-folder datasets/rby1_teleop_pp_0724
```
PASS: MAE ≈ 0.02 with smooth per-block curves (this is its training distribution).

## 7. What to expect / watch

- **Grasp is the key observable** — r_hand carries the largest error block (per-block
  clean ~0.04) simply because it moves the most; the arm/torso blocks fit tightly (≤0.02).
- Base/torso do small coordinating motion; if the base drives off, suspect the delta→cumsum
  integration, not the policy.
- Image is the single input the policy leans on — if a rollout misbehaves in a way that does
  NOT reproduce in the §6 replay, check BGR order / resize / exposure first.
- Robot pace ≈ 10 Hz chunk playback; start in safe mode.

## 8. Gate table (training-corpus probe, 20 samples; clean vs proprio-zero vs image-shift)

| policy | ep | clean | pzero | reliance | shift10 | shift20 |
|---|---|---|---|---|---|---|
| tel_resnet | 999 | 0.0191 | 0.0193 | ×1.01 | 0.0188 | 0.0186 |
| tel_resnet | 699 | 0.0210 | 0.0211 | ×1.00 | 0.0213 | 0.0211 |
| tel_wam3   | 899 | 0.0208 | 0.0213 | ×1.02 | 0.0205 | 0.0209 |
| tel_wam3   | 999 | 0.0212 | 0.0213 | ×1.00 | 0.0206 | 0.0210 |
| tel_d3conv | 999 | 0.0228 | 0.0241 | ×1.06 | 0.0228 | 0.0228 |
| tel_d3conv | 599 | 0.0281 | 0.0275 | ×0.98 | 0.0273 | 0.0273 |

reliance = pzero ÷ clean; ×1.0 = pure vision. Every teleop checkpoint is vision-driven —
dropout 0.5 on robot data achieved what needed 0.9 on human data. Held-out val (20 % episode
holdout) is flat across the three within ~2 % (wam3 0.0760 < resnet 0.0768 < d3conv 0.0774).
