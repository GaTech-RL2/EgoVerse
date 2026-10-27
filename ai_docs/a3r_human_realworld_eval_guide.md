# a3r_human — real-world evaluation guide (single policy, everything you need)

**The question this answers:** can a policy trained *only on human demonstrations*
(RGB + depth, Aria glasses, SEW-retargeted actions) drive the real RBY1?

Self-contained — no need to read the other guides. Final checkpoint:
`logs/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/epoch_epoch=1999.ckpt`
(ep1999, final val 0.0903).

---

## 0. Bottom line before you start

Honest risk assessment from what we've measured. Read this so a failure is informative
rather than confusing:

| evidence | says |
|---|---|
| offline on human data | works well (val 0.0903; dry-run MAE 0.0080) |
| scored on **real robot** observations | 0.204 — **does not beat** predicting the episode mean (0.121) |
| vs its image-only twin (h_rect) on robot data | **a3r_human is better** (0.204 vs 0.221) — depth helps cross-embodiment |
| depth sensitivity (§2) | **knife-edge**: a 10% depth scale error is as damaging as deleting depth |
| scene depth statistics (§2b) | robot table scene is **closer** than human training average |
| attention on robot frames | diffuse (entropy 0.78) — least object-focused of our policies |

**Expect a first rollout that reaches/gestures rather than completes the task.** The
value of this test is diagnostic: §6 tells you *which* of the three candidate causes
(depth mismatch / embodiment gap / speed) produced what you see. A "it didn't work"
with no attribution is the one outcome worth avoiding — the ladder in §4 prevents it.

---

## 1. What the robot side must provide

| # | requirement | why | verify |
|---|---|---|---|
| 1 | **Raw RGB camera frame**, 640×640 as streamed | LUT input (§3a) | publisher exposes it |
| 2 | **Live metric depth**, 512² or 224², **slam-rect frame**, metres, 0=invalid | the policy's 4th channel | §2 |
| 3 | Proprio: 22-D no-wheel joints + 2×12 hand qpos | obs | §3c |
| 4 | 10 Hz obs loop, action chunk executor | contract | existing |

Frame definition for #2 — rectified LEFT SLAM camera, pinhole:
`fx = fy = 307.336684`, `cx = cy = 256.0` at 512×512 (scale by 224/512 if you send 224).
Optical convention X-right, Y-down, Z-forward. Depth is **Z along the optical axis**,
not ray range.

> The encoder lifts depth with **identity extrinsics** — it builds its point cloud in
> this camera frame directly. No rotation into a device/glass frame (that's the DP3
> contract, don't reuse it here).

---

## 2. Depth — the make-or-break input

### 2a. Measured sensitivity of THIS checkpoint

Perturbing depth at inference, everything else held fixed (job 3681948):

| depth fed to the policy | MAE | vs correct |
|---|---|---|
| **correct** | **0.0080** | 1.0× (reference) |
| 50% of pixels dropped to holes | 0.0341 | 4.3× |
| **+0.10 m constant offset** | 0.0686 | **8.5×** |
| all-zero (no depth) | 0.0761 | 9.5× |
| **×1.10 scale error (10%)** | 0.0806 | **10.0×** |
| flat 1.0 m everywhere | 0.2868 | 35.7× |

**Read this carefully: a 10% scale error or a 10 cm offset is *worse than giving the
policy no depth at all*.** Metric accuracy is not a nice-to-have here. Conversely,
missing pixels are comparatively benign (4.3×) — holes are in-distribution, wrong
metric scale is not.

### 2b. Distribution check — what your live depth should look like

| | valid frac | median | p5 | p95 | frac < 1.5 m |
|---|---|---|---|---|---|
| **human training data** (what it learned on) | 1.000 | 1.79 m | 0.49 | 8.37 | 0.40 |
| robot near-table scene (measured) | 0.999 | 1.29 m | 0.40 | 4.97 | 0.60 |

The robot scene is **closer** than the training average (which includes nav phases). Not
fatal, but it is a real shift — and it interacts with §2a: the policy is most reliable in
the depth range it saw most.

### 2c. Pre-flight depth checks — do these before anything else

1. **Tape measure.** Point at the table. Sample depth at 3 known points (table edge,
   basket, far wall). **Each must agree within 3%** (at 1.5 m that's ±4.5 cm). If it
   fails, stop — §2a says you are already worse off than having no depth.
2. **Distribution.** One frame, valid pixels: median should land ~1.0–2.0 m, p95 under
   ~6 m, valid fraction > 0.95. Compare to the table above.
3. **Frame alignment.** Overlay depth on the rect image (§3a output). Object edges must
   coincide. A shifted depth map means the wrong frame — fix before proceeding.
4. **Units.** If median reads ~1290 instead of ~1.29, you're in millimetres. Divide.

---

## 3. Building the observation

### 3a. Image (raw → policy frame)

**CORRECTED 2026-08-17 (verified against the live publisher code):** with the
`lawrence_custom` streaming profile the raw RGB frame arrives **already 640×640** —
feed it to the LUT directly, no crop/resize. (The robot Aria's native calib is
2016×1512, f=877.96; the on-device 640² = centre-crop 1512² @x=252 scaled 640/1512.
Cross-check: that chain predicts f=371.6, cx=312.1 — our independently solved LUT
model is f=370.40, cx=310.5. Two derivations, 0.3% apart.) Only if a full-res
2016×1512 frame ever arrives: centre-crop 1512² at x=252, y=0, then INTER_AREA to
640. An earlier version of this section said 2560×1920 / crop@320 — that was the
HUMAN device's native mode, wrong for the robot unit.

```python
import numpy as np, cv2
L = np.load("ai_docs/assets_rect_lut/robot_rect224_lut.npz")
MX, MY = L["map_x"], L["map_y"]

def rect_rgb_224(raw_640_bgr):                          # 640x640 BGR as delivered
    rect = cv2.remap(raw_640_bgr, MX, MY, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rect[..., ::-1].copy()                       # BGR -> RGB. REQUIRED.
```
Self-check: save one frame — **basket blue, penguin feet orange**. (Human-trained policy;
BGR input silently degrades the frozen DINO backbone.) LUT validated to 1.44 px median
against the depth-exact training pipeline. ~9.5% black border is expected and harmless.

### 3b. Depth

```python
def depth_224(depth_512):
    return cv2.resize(depth_512, (224, 224), interpolation=cv2.INTER_NEAREST)
    # NEAREST only — bilinear invents flying pixels at edges. Never in-paint the zeros.
```

### 3c. Full obs dict

```python
obs = {
    "front_img_1":      rect_rgb_224(raw),      # (224,224,3) uint8 RGB
    "aria_depth":       depth_224(d512),        # (224,224) float32 METRES, 0=invalid
    "robot0_joint_pos": q22,                    # [torso 6, r_arm 7, l_arm 7, head 2]
    "hand_left_qpos":   hl,                     # (12,)
    "hand_right_qpos":  hr,                     # (12,)
}
```
Image and depth must be **pixel-aligned** — same frame, same rays.

### 3d. Actions out

`(1, 32, 49)` @10 Hz. `base[0:3]` = per-step deltas → integrate by cumsum in the frame-0
heading. `torso[3:9] head[9:11] l_arm[11:18] r_arm[18:25] l_hand[25:37] r_hand[37:49]` =
absolute joint targets.

⚠ **Executes at ~1/6 speed.** The human corpus is 60 Hz rows labeled 10 Hz, so a 32-step
chunk is ~0.53 s of human motion replayed over 3.2 s. Slow, smooth motion is *correct
behaviour*, not a hang. Every human-corpus rollout has this property.

⚠ If you compare predictions to logged actions, use
`actions.joint_base_torso_head_arm_hand` — **not** `actions.joint`. Both are 49-D and
different; the wrong one makes a healthy policy look ~3.7× broken.

---

## 4. Verification ladder — climb in order, no skipping

**Rung 1 — offline replay (zero hardware risk).**
```bash
python egomimic/scripts/serve_policy.py \
  --checkpoint logs/RBY1_adapt3r_human/adapt3r_human_2k/checkpoints/epoch_epoch=1999.ckpt --port 8002
python egomimic/scripts/test_serve_policy_client.py \
  --dataset-folder datasets/human_fullpp_rgbd --episode-idx 0 --max-steps 30 --trajectory
```
**Reference: full-chunk MAE 0.0070–0.0080, t1 0.0073, arms+hands 0.0082.** Match to
~1e-3 → your serving stack is faithful. (Flow-matching sampling is stochastic; not
bit-exact.) If this fails, nothing downstream is meaningful.

**Rung 2 — depth pre-flight.** §2c, all four checks.

**Rung 3 — live obs, no motion.** Feed live camera+depth into the policy with the robot
stationary; **do not execute**. Confirm:
- image self-check passes (blue basket) on a *live* frame
- depth stats in range on a *live* frame
- policy returns (1,32,49) at ≥10 Hz without stalling
- predicted arm targets are near the current joint positions (a policy predicting wild
  jumps from a rest pose is a red flag — check proprio ordering, §3c)

**Rung 4 — motion, safety first.** Reduced speed limit, hand on e-stop, table clear of
anything fragile. Start with the arms only if your executor supports masking the base.

---

## 5. Rollout protocol

- **Start pose:** the full task (approach + pick + place), not a near-table close-up.
  Human-corpus policies are known-OOD for close-up views — a near-table start tests the
  wrong thing.
- **Scene:** the trained scene — grey table, blue basket, penguin. Novel objects are a
  different experiment.
- **Runs:** 5 rollouts from slightly varied start poses. One rollout is noise.
- **Log per rollout:** start pose, and the furthest stage reached —
  `no motion / moves but wrong direction / reaches toward object / contacts / grasps /
  lifts / places`. That ladder is the result; a binary success/fail throws away the
  signal.
- **Optional but valuable:** record the obs stream. If it fails, §6 diagnosis is far
  easier with the actual frames the policy saw.

**If you can spare 20 more minutes:** run h_rect (image-only twin, `logs/RBY1_human_rect/
human_rect_resnet_2k/checkpoints/epoch_epoch=1999.ckpt`, same obs minus `aria_depth`)
back-to-back from the same start pose. That pair is the actual depth experiment — offline
says h_rect wins, robot-data scoring says a3r_human wins, and the hardware breaks the tie.

---

## 6. Failure diagnosis — which cause was it?

| symptom | most likely cause | check |
|---|---|---|
| Wild/jerky targets from rest | proprio ordering or units | is `robot0_joint_pos` the 22-D no-wheel order? |
| Motion is coherent but *very slow* | **nothing — this is expected** | §3d, 1/6 speed |
| Plausible but consistently mis-reaching by a fixed amount | **depth metric error** | re-run §2c tape check; 10 cm ⇒ 8.5× degradation |
| Behaves like the image-only policy; depth seems ignored | depth all-zero / not wired | print depth stats inside the obs builder |
| Erratic, unrelated to scene | wrong frame or channel order | §3a self-check; overlay depth on image |
| Reaches the right region, fails to grasp | **embodiment gap** — the real finding | compare with h_rect run |
| Nothing moves | serving/executor, not the policy | rung 1 + rung 3 |

The distinction that matters for your question: **rows 3–5 are your bugs; row 6 is the
science.** Only after the depth and frame checks pass does a failure tell you something
about human→robot transfer.

---

## 7. What the outcome means

- **Reaches the object region consistently** → human-only training transfers meaningfully.
  Big result; next step is a small amount of robot data for fine-tuning.
- **Moves plausibly but never converges on the object** → the expected outcome given
  §0. The cross-embodiment number predicted exactly this. Next lever is co-training with
  teleop data, not more human data.
- **Beats h_rect on the same starts** → depth earns its place cross-embodiment even
  though it costs 0.025 in-domain. That would be a genuinely new finding and would
  redirect the 3D round.
- **Nothing coherent, but rungs 1–3 all passed** → the human→robot appearance gap is
  larger than depth can bridge; the honest conclusion is that human-only pretraining
  needs robot data to be useful, and the 3D question should be settled on teleop data.

*Checkpoint ep1999 · guide written 2026-08-06 · sensitivity numbers from job 3681948 ·
dry-run reference from job 3655880.*
