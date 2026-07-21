# FPP Round — Real-Robot Deployment Note (updated 2026-07-21, HD era)

> Serving basics unchanged: `deployment_plan.md` §1–§6 · debugging: `deployment_debug_guide.md`
> · session protocol/scoring: `deployment_test_protocol_r6.md` §2.
> **Code: pull branch `rby1_aria_policy` LATEST** — needs this round's classes
> (`egomimic/utils/image_augs.py`, `egomimic/models/custom_encoders.py` incl. the
> nvs3d cold-load fix). Older checkouts will NOT unpickle these checkpoints.

## 1. Checkpoints to test (2026-07-21 HD shortlist)

Hardware showed the 07-20 list (dropout 0.6) over-attends proprio; gate probes
confirmed (reliance = proprio-zero MAE ÷ clean MAE; ×1.0 = pure vision):

| priority | tag | checkpoint (under `/coc/flash7/czhang883/Documents/EgoVerse/`) | clean | reliance |
|---|---|---|---|---|
| **A** | hd_wam3@899 | `logs/aria_fullpp_wam3/fpp_hd_wam3_2k/checkpoints/epoch_epoch=899.ckpt` | 0.029 | **×1.00** |
| **B** | hd_resnet@1499 | `logs/aria_fullpp/fpp_hd_resnet_2k/checkpoints/epoch_epoch=1499.ckpt` | 0.024 | **×1.00** |
| later | hd_nvs3dneck@~1599 | training now (`fpp_hd_nvs3dneck_2k`) — gate-check before use | — | — |

Runs still training: later snapshots (~ep1599 region, historically the optimum)
supersede A/B when they land — same paths, higher epoch numbers; re-gate first.
**Do NOT deploy**: `hd_nvs3d` (linear probe) — reliance ×2.9, same proprio trap as
before (readout capacity too small to go vision-only); all 07-20 dropout-0.6
checkpoints (d3lora ×6.9, d3conv ×4.4 at maturity). 0.6-era fallback if HD
disappoints at the table: `wam3@1599` (`logs/aria_fullpp_wam3/fpp_wam3_2k/...=1599.ckpt`,
clean 0.013, ×1.28 — best balanced of that era).
NVS-3D serving needs `NVS3D_DIR` env → folder holding `model.py` + `0981at.pt`
(cluster copy: `/coc/flash7/czhang883/pretrained/nvs3d/`).

## 2. Serve

```bash
cd /coc/flash7/czhang883/Documents/EgoVerse   # branch rby1_aria_policy, dd74911d+
source emimic/bin/activate
python egomimic/scripts/serve_policy.py --checkpoint <ckpt-from-table> --port 8000
```
One port per checkpoint. On-connect metadata must say embodiment rby1 / action_dim 49 / horizon 32.

## 3. Obs contract (unchanged, with this round's confirmations)
- `front_img_1`: 224×224 uint8 **BGR** (cv2-native is correct — the server flips to
  RGB; this round PROVED the robot camera stack is BGR and the serving path handles it).
  Raw frame resized square; no rectification, no cropping.
- `robot0_joint_pos`: 22-D no-wheel, order [torso 6, r_arm 7, l_arm 7, head 2] — REAL
  measured values.
- `hand_left_qpos`, `hand_right_qpos`: 12 + 12.
- Send real proprio. (These policies mostly rely on vision — measured — but real
  proprio is the validated mode.)

## 4. Action contract (unchanged)
(1, 32, 49) un-normalized @10 Hz: base[0:3] **per-step deltas** → integrate by plain
cumsum in the rollout-start (frame-0) heading, NO yaw rotation; torso[3:9], head[9:11],
l_arm[11:18], r_arm[18:25], l_hand[25:37], r_hand[37:49] = absolute joint targets.

## 5. What to expect / watch (from this round's measurements)
- **Navigation should be good for every checkpoint** (heading-cosine ~0.9, 100% correct
  turn direction from epoch 99 onward). The policies drive at HUMAN pace ≈ 1.3–1.9×
  your teleop speed — start in safe-mode, expect brisker driving than teleop demos.
- **At the table**: the known pose-style offset (~0.15 rad toward human-style
  configurations) persists in all runs — expect an initial pull toward its preferred
  pose when the chunk executor engages; the receiver's interpolation smooths this but
  watch the first second at the table. Grasp attempts are the key observable.
- **Scoring** (per protocol §2.2, ≥3 rollouts each, fixed start + object placement):
  NAV / APPROACH / GRASP-ATTEMPT / GRASP / PLACE, note failure mode
  (skip-ahead / think-done / freeze / wrong-target).
- **Capture for me**: frames-as-sent (img_compare dump), the (32,49) chunks, and video
  — I can attention-map and replay any failure offline next day.

## 6. Quick sanity before robot time
Rung 0–2 of `deployment_debug_guide.md` §2 with checkpoint A once: serve → replay-eval
(`test_serve_policy_client.py --trajectory` vs `datasets/rby1_teleop_val_v2`) — PASS is
MAE ≈ the table numbers above and smooth per-block curves; then receiver dry-run.

## 7. Context for interpreting results
Offline, NO policy beats the stand-still MAE bar (0.058) — the metric ceiling is
pose-style, not competence; hardware task progression is exactly the evidence the
offline metric cannot provide. The interesting hardware questions, in order:
(1) does any run GRASP more reliably than July's crop100 did, (2) A-vs-D encoder gap,
(3) C's world model at the table, (4) E-vs-F glove effect in the real loop.
