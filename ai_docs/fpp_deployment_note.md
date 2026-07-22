# FPP Round — Real-Robot Rollout Guide (HD era, updated 2026-07-21 evening)

> Serving basics: `deployment_plan.md` §1–§6 · debugging: `deployment_debug_guide.md`
> · session protocol/scoring: `deployment_test_protocol_r6.md` §2.
> **Code: pull branch `rby1_aria_policy` LATEST before serving** — needs this round's
> classes (`egomimic/utils/image_augs.py`, `egomimic/models/custom_encoders.py` with
> the nvs3d cold-load fix). Older checkouts will NOT unpickle these checkpoints.

## 0. What changed since the last robot session (READ)

The 07-20 checkpoints over-attended proprio on hardware. The new **HD runs
(proprio dropout 0.9)** trained that out: offline, zeroing proprio changes their
predictions by ~nothing (reliance ×1.00 vs ×1.3–6.9 before). These are
functionally **vision-driven policies**. Practical consequences for rollouts:
- Proprio calibration issues can no longer cause the old failure mode.
- The camera feed is now the single point of failure — framing, lighting,
  BGR order, and image scaling deserve the pre-flight attention.
- Validation numbers quoted below are the OLD method (held-out human episodes +
  training-corpus gate probes). Teleop-val numbers are retired everywhere.

## 1. Rollout list (ranked, gate-verified)

| priority | tag | checkpoint (under `/coc/flash7/czhang883/Documents/EgoVerse/`) | clean | reliance |
|---|---|---|---|---|
| **A** | hd_wam3@1399 | `logs/aria_fullpp_wam3/fpp_hd_wam3_2k/checkpoints/epoch_epoch=1399.ckpt` | 0.025 | **×1.03** |
| **B** | hd_resnet@1499 | `logs/aria_fullpp/fpp_hd_resnet_2k/checkpoints/epoch_epoch=1499.ckpt` | 0.024 | **×1.00** |
| C (baseline) | wam3@1599 (0.6-era) | `logs/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=1599.ckpt` | 0.013 | ×1.28 |

- **A vs B** = world-model question, now with clean vision-driving in both.
- **C** = best balanced 0.6-era checkpoint; roll it only for continuity comparison
  with the July session (it fits tighter offline but leans on proprio ×1.28 — the
  A/B-vs-C behavioral difference at the table is itself a result).
- **Do NOT roll out**: `hd_nvs3d` linear (reliance ×3.4 at maturity — same proprio
  trap as July despite good-looking val; val is scored WITH real proprio and hides
  this); any 07-20 dropout-0.6 DINOv3 checkpoint (d3conv ×4.4, d3lora ×6.9).
- **hd_nvs3dneck**: reliance PLATEAUED ×1.5–1.8 (599/799/999) — better than linear
  (×3.4) but does NOT reach ×1.0; val also plateaus above leaders. NOT cleared.
- **hd_d3conv (HD twin)**: ×1.38 @899 (was ×4.4 at drop 0.6) and still training —
  best frozen-feature variant; candidate for the list at maturity (~ep1500+, re-gate).
- **hd_d3lora (HD twin)**: ×1.89 @799 (was ×6.9) — improved, still elevated.
- avoid `hd_wam3 epoch_epoch=1599.ckpt` — transient bad snapshot (post-resume);
  finals 1899/1999 are clean ×1.0 but @1399 remains wam3's best fit.

## 2. Serve (one port per checkpoint)

```bash
cd /coc/flash7/czhang883/Documents/EgoVerse   # branch rby1_aria_policy, latest
git pull && source emimic/bin/activate
# harmless for A/B/C; REQUIRED if serving any NVS-3D checkpoint later:
export NVS3D_DIR=/coc/flash7/czhang883/pretrained/nvs3d

python egomimic/scripts/serve_policy.py --checkpoint logs/aria_fullpp_wam3/fpp_hd_wam3_2k/checkpoints/epoch_epoch=899.ckpt  --port 8000   # A
python egomimic/scripts/serve_policy.py --checkpoint logs/aria_fullpp/fpp_hd_resnet_2k/checkpoints/epoch_epoch=1499.ckpt   --port 8001   # B
python egomimic/scripts/serve_policy.py --checkpoint logs/aria_fullpp_wam3/fpp_wam3_2k/checkpoints/epoch_epoch=1599.ckpt   --port 8002   # C
```
On-connect metadata must say embodiment rby1 / action_dim 49 / horizon 32.
(If serving from a different machine, copy the ckpt + set NVS3D_DIR to a local
folder containing `model.py` + `0981at.pt` for NVS-3D checkpoints only.)

## 3. Obs contract (unchanged from July — all confirmed this round)

- `front_img_1`: 224×224 uint8 **BGR** (cv2-native — the server flips to RGB).
  Raw frame resized square; no rectification, no cropping. This is now the input
  that carries ~all of the policy — double-check the live `img_compare` dump
  matches training frames (fisheye look, full FOV, not a center crop).
- `robot0_joint_pos`: 22-D no-wheel, order [torso 6, r_arm 7, l_arm 7, head 2].
- `hand_left_qpos`, `hand_right_qpos`: 12 + 12.
- Send real proprio as usual. **New debug tool**: for A/B you may send all-zero
  proprio as an isolation test — offline this changes nothing (×1.00); if live
  behavior changes a lot under zeroed proprio, something else is wrong (send me
  the dump). Don't zero proprio for C.

## 4. Action contract (unchanged)

(1, 32, 49) un-normalized @10 Hz: base[0:3] **per-step deltas** → integrate by
plain cumsum in the rollout-start (frame-0) heading, NO yaw rotation; torso[3:9],
head[9:11], l_arm[11:18], r_arm[18:25], l_hand[25:37], r_hand[37:49] = absolute
joint targets.

## 5. What to expect / watch

- **Nav**: solid for every listed checkpoint (heading-cosine ~0.9 class, correct
  turn signs). Policies drive at HUMAN pace ≈ 1.3–1.9× your teleop speed — start
  in safe mode.
- **At the table**: the human→robot pose-style offset (~0.15 rad pull toward
  human-like configurations) is a property of the data, not the dropout — expect
  the same initial pull as July when the chunk executor engages. Grasp attempts
  remain the key observable.
- **HD-specific**: if a rollout goes wrong, suspect the image path first (the
  policy has nothing else to lean on). Symptoms of image trouble: freezing or
  drifting that does NOT reproduce offline; check BGR order, resize, exposure.
- **A vs B**: same recipe, different eyes (world-model ResNet vs plain ResNet).
  July's C-vs-D question, now uncontaminated by proprio shortcuts.

## 6. Pre-robot sanity (5 min, once)

Serve A, then replay-eval against the training corpus from the repo:
```bash
python egomimic/scripts/test_serve_policy_client.py \
  --episode-idx 0 --max-steps 30 --trajectory \
  --dataset-folder datasets/aria_fullpp
```
PASS: MAE in the 0.02–0.06 range with smooth per-block curves (this corpus is
its training distribution; C should score ~0.01–0.02). Then receiver dry-run.

## 7. Scoring + capture (unchanged protocol)

Per protocol §2.2: ≥3 rollouts each, fixed start + object placement; score
NAV / APPROACH / GRASP-ATTEMPT / GRASP / PLACE; note failure mode
(skip-ahead / think-done / freeze / wrong-target).
**Capture for me**: frames-as-sent (img_compare dump), the (32,49) chunks, and
video — I can replay any failure offline against the gate probes next day.

## 8. Context for interpreting results

Offline this round: HD policies fit the human corpus at 0.024–0.029 with ZERO
proprio dependence; the 0.6-era fits tighter (0.011–0.013) but by leaning on
proprio (×1.3–6.9), which is what failed in July. Held-out val says all HD runs
sit ≈0.068 with real proprio — honest generalization, same ballpark as each
other. Hardware task progression (especially GRASP rate vs July's crop100 and
vs C) is the evidence offline numbers cannot provide. The open questions, in
order: (1) does vision-driving fix the table behavior you saw, (2) A-vs-B
world-model effect, (3) A/B-vs-C — does removing proprio reliance visibly
change behavior, (4) whether the 3D-aware neck (still training) earns a slot
tomorrow.
