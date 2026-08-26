# FPP HD-era Round — Sim→Real log (updated 2026-07-21 eve)

HD runs = proprio-dropout-0.9 → **vision-driven** (reliance ≈×1.0). If a rollout
misbehaves, suspect the **image path first** (BGR/resize/exposure). Head **not** frozen
for future runs (policy drives head; drop `FREEZE_HEAD`). Obs: `front_img_1` 224² BGR · joint_pos 22 ·
hands 12+12. Act (1,32,49)@10Hz, base=cumsum deltas. `EXEC_STEPS=16`, `SAFE_MODE=1` on HW.
Branch `rby1_aria_policy` LATEST (`git pull`) required to unpickle.

**Dataset = `aria_fullpp`** (135 eps, the HD training corpus). `aria_egoposer_firm` was
the OLD firm round and is the wrong distribution for GT-input; only use `aria_fullpp` now.
For live full-real the dataset is used only for the `dataset_avg` soft-reset pose + preview.

**This session: go straight to live full-real (S3)** — live Aria RGB + HW proprio — one
checkpoint at a time (A→B→C). Restart the server only when changing checkpoint.
Optional: add `VIZ_SAMPLES=20` to draw the parallel-denoising ghost fan during each preview.

## Checkpoints (pull: `bash pull_fpp_hd.sh`)

| pri | tag | ckpt (VARIANT) | clean | reliance |
|-----|-----|----------------|-------|----------|
| A | hd_wam3@1399 | `hd_wam3`  (world-model ResNet) | 0.025 | ×1.03 |
| B | hd_resnet@1499 | `hd_resnet` (plain ResNet) | 0.024 | ×1.00 |
| C | wam3@1599 | `hd_c` (0.6-era baseline; **don't zero proprio**) | 0.013 | ×1.28 |

Do NOT roll out: hd_nvs3d linear, any 07-20 dropout-0.6 DINOv3 (d3conv/d3lora). §1 vs §2
of guide disagree on A epoch (1399 table vs 899 serve) → default 1399, `EPOCH_A=899` to override.

## Per-ckpt flow: sim → GT-real → full-real (one at a time)

**Serve (once per ckpt, EgoVerse):**
```bash
cd ~/RB_Y1_workspace/EgoVerse && source emimic/bin/activate
VARIANT=hd_wam3 PORT=8000 bash serve_aria_egoposer.sh   # A; B=hd_resnet C=hd_c
```
Expect on-connect: embodiment **rby1** · action_dim **49** · horizon **32**.

**S1 — MuJoCo sim, GT input** (optional pre-check; dataset img+hands, no HW):
```bash
cd ~/RB_Y1_workspace/SEW-Geometric-Teleop
GT_MODE=gt_proprio PORT=8000 EXEC_STEPS=16 DEMO_NAME=0 \
  DATASET=~/RB_Y1_workspace/EgoVerse/datasets/aria_fullpp \
  bash projects/rby1_teleop/run_rollout_aria_egoposer_sim.sh
```

**S3 — full real, LIVE inputs** (live Aria RGB + HW proprio, head policy-driven) — the session driver:
```bash
cd ~/RB_Y1_workspace/SEW-Geometric-Teleop
SAFE_MODE=1 SHOW_CAMERA=1 EXEC_STEPS=16 PORT=8000 \
  DATASET=~/RB_Y1_workspace/EgoVerse/datasets/aria_fullpp \
  REC_COMPARE_DIR=/tmp/hd_wam3_1399_real_$(date +%H%M%S) \
  bash projects/rby1_teleop/run_rollout_aria_egoposer.sh
```
Switch ckpt: restart serve only (`hd_resnet` / `hd_c`); S3 identical (update tag).
Optional ghost fan during preview: prepend `VIZ_SAMPLES=20`.

## Results — score NAV/APPROACH/GRASP-ATTEMPT/GRASP/PLACE (Y/N)

| tag | S1 sim GT (opt) | S3 live #1 | S3 live #2 | S3 live #3 | fail mode | capture |
|-----|-----------------|------------|------------|------------|-----------|---------|
| A hd_wam3@1399 | | NAV/APPR/GRASP-ATT/GRASP Y — **promising** | GRASP-ATT Y, **almost success** (2 vids) | | aria-lost / near-miss | /tmp/hd_wam3_1399_real_* |
| B hd_resnet@1499 | | NAV weak — never reaches table (stops far) | | | nav-short / wrong-target | /tmp/hd_resnet_1499_real_* |
| C wam3@1599 | | bad — poor at task | | | other | /tmp/hd_c_1599_real_* |
| d3lora@1699 (retry) | | NAV Y, GRASP-ATT Y — over-rotates→table collision | | | pose-yank / over-rotate | /tmp/d3lora_1699_real_* |
| d3conv@1299 (retry) | | NAV ok but strays → hits side table | | | wrong-target / stray | /tmp/d3conv_1299_real_* |

**A hd_wam3@1399 (07-21 eve, S3 live #1):** OK most of the time and **grasped the target**;
run stopped due to **Aria connection lost** (USB/stream drop, seq stuck), not a policy failure.
Promising — needs a clean full run to score PLACE. Added `--hold-on-stale-frame` (default on) so
future runs freeze safe instead of driving blind when Aria drops; still need to fix the drop itself.

**A hd_wam3@1399 (07-21 eve, S3 live #2, no-safe-mode + full 32-chunk, head policy-driven):**
**almost success** — nearly completed the grasp. Looks like it would succeed within a few more
attempts. **2 videos recorded** for this ckpt. Best HD-era policy so far; strong candidate.

**B hd_resnet@1499 (07-21 eve, S3 live, no-safe-mode + full 32-chunk):** **clearly worse than A.**
Struggles even on **navigation** — approaches the table but always stops **too far away**, never
reaching grasp range. Trained to **epoch 1499**. Plain-ResNet (no world-model) underperforms the
world-model ResNet (A hd_wam3); consistent w/ A>B hypothesis.

fail: skip-ahead · think-done · freeze · wrong-target · pose-yank · aria-lost · other
Open Qs: (1) vision-driving fix table behavior? (2) A vs B world-model (3) A/B vs C proprio reliance.

## 07-20 round (superseded — proprio-trap) — 07-21 eve retries under HD conditions
Retried under current conditions (aria_fullpp, no-safe-mode, full 32-chunk, head policy-driven):
**d3lora@1699 (retry):** actually **decent at NAV** and **attempts the grasp** on the target, but
**over-rotates at the end → collides with the table.** Better than its earlier "bad" verdict, but
still not clean (pose-yank/over-rotate at grasp). Not beating A hd_wam3.
**d3conv@1299 (retry):** **also bad.** NAV ok but **strays too much**, drifts toward the **side
table and hits it** (wrong-target / lateral drift). Not beating A.
Prior verdicts (older conditions): d3lora@1699 bad · d3conv@1799 worse · (wam3/resnet not completed).

**Conclusion (07-21 eve):** across HD (A/B/C) + 07-20 retries (d3lora/d3conv), **A hd_wam3@1399 is
the only policy that cleanly reaches + grasps the target.** All others fail on nav-precision or
grasp-pose (stop-far / over-rotate / lateral-stray). Next: fix Aria drop (publisher watchdog) and
run A to a scored full success.

## 07-21 late — NEAR-TABLE (skip-nav) retests, SAFE_MODE=1
New setting: **robot physically placed right by the table** so the policy skips the nav phase and
goes straight to approach/grasp — isolates grasp capability from nav-precision failures. This round
uses **SAFE_MODE=1** (preview + y/N confirm each chunk), EXEC_STEPS=16, head policy-driven,
DATASET=aria_fullpp. 4 runs planned:

| # | tag | VARIANT= | ckpt | near-table result | capture |
|---|-----|----------|------|-------------------|---------|
| 1 | A hd_wam3@1399 | hd_wam3 | epoch=1399 | FAIL — confused by hand/object appearance | /tmp/hd_wam3_1399_nt_* |
| 2 | B hd_resnet@1499 | hd_resnet | epoch=1499 | FAIL — confused by hand/object appearance | /tmp/hd_resnet_1499_nt_* |
| 3 | glove@1999 | glove | exp1_glove newest | FAIL — proprio-reliant, no real manip | /tmp/glove_nt_* |
| 4 | bare@1899 | bare | exp1_bare newest | FAIL — proprio-reliant, no real manip | /tmp/bare_nt_* |

glove vs bare = exp1 glove-worn vs bare-hand data comparison (also near table).

**Near-table result (07-22 ~00:15): none succeeded.** Two distinct failure modes:
- **HD (A hd_wam3, B hd_resnet):** get **confused by the hand/object appearance** up close — the
  vision-driven policies mis-read the near-field scene (hand+object in frame) and don't produce a
  correct grasp. Note this contrasts with A's *from-distance* near-success — suggests the near-table
  visual distribution (large hand/object in view) is **out-of-distribution** for the image path.
- **non-HD (glove, bare):** **over-rely on proprioception** → with proprio held ~static at the table
  they produce **no meaningful manipulation** (essentially frozen/no grasp). Confirms the 07-20
  proprio-trap; not vision-driven enough to act from the near-table image alone.

**Takeaway:** near-table (skip-nav) is HARDER, not easier, for these ckpts — the up-close hand/object
appearance is OOD for HD vision, and non-HD can't act without a nav proprio ramp. A hd_wam3's best
result remains the **from-distance** run. Candidate fixes: (1) add near-table / hand-in-view frames
to training distribution; (2) for HD, check image path at close range (exposure/crop/scale).
