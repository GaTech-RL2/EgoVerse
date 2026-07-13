# Deployment & Test Protocol — Round-6 Candidates (sim + real)

> How to deploy and evaluate the current checkpoint shortlist, in two stages:
> offline/"simulation" verification (no robot), then the real-world A/B session.
> Contract & serving basics: `deployment_plan.md` §1–§6. Debugging: `deployment_debug_guide.md`.

## 0. The candidates

| tag | checkpoint (`/coc/flash7/czhang883/Documents/EgoVerse/logs/…`) | why it's in the shortlist |
|---|---|---|
| **A. crop100** (primary) | `aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt` | only policy passing every offline gate; proven nav on hardware |
| **B. dino_full** | `aria_egoposer_firm/dino_full_2k/checkpoints/last.ckpt` | best clean accuracy; tests whether adapted-ViT features transfer better |
| **C. wam_dinofull** | `aria_egoposer_firm_wam/wam_dinofull_2k/checkpoints/last.ckpt` | world-model co-trained; tests whether future-prediction fixes task-progress confusion |
| (optional D. wam_res) | `aria_egoposer_firm_wam/wam_res_2k/checkpoints/last.ckpt` | only if curious — weakest offline clean (0.026) |

All four share the identical serving contract (§1): raw fisheye 224² uint8 BGR image +
22-D no-wheel proprio + 12+12 hand qpos → (1,32,49) un-normalized chunk. The WAM
checkpoints' world head is automatically skipped at inference (no receiver change).
⚠️ Serve all of them from branch `rby1_aria_policy` (clamp + DINOv2/aux code required).

## 1. Stage "SIM" — full-stack verification without the robot

Purpose: prove the *deployment stack* (server, protocol, receiver, execution math)
end-to-end before any hardware time. There is no physics simulator wired for this
task; "sim" here = replaying recorded episodes through the REAL serving path and
verifying the closed loop numerically. Four checks, ~30 min total per checkpoint:

1. **Serve** each candidate on its own port:
   `python egomimic/scripts/serve_policy.py --checkpoint <ckpt> --port 800X`
   Verify the on-connect metadata (embodiment rby1, action_dim 49, horizon 32).
2. **Replay eval (offline "rollout")**: `test_serve_policy_client.py --trajectory`
   against `datasets/aria_egoposer_firm` — feeds recorded obs through ws+msgpack,
   plots pred-vs-GT. PASS: MAE ≈ the offline numbers (A: 0.013 / B: 0.012 / C: 0.014),
   smooth per-block curves.
3. **Receiver dry-run**: run the actual robot-side receiver with obs from the dataset
   (or the robot powered but not executing). Log every (obs, chunk) pair. PASS:
   chunks smooth (see debug guide §3), base deltas |Δ|≤0.024/0.094, latency <150 ms.
4. **Execution-math unit check**: feed one recorded chunk into the receiver's
   integration/interpolation code and verify the commanded joint trajectory equals
   the chunk at 10 Hz nodes (esp. base cumsum, no yaw-rotation, 0.1 s spacing).

If the hardware side has its own simulator (MuJoCo/SEW), the same server + receiver
can drive it — the policy is sim-agnostic; only the receiver's robot API changes.
Treat sim success/failure as a STACK test, not a policy test (the policy never saw
sim visuals; expect degraded behavior from the appearance gap).

## 2. Stage "REAL" — the A/B(/C) session

### 2.1 Session protocol (per checkpoint: ~15 min)
1. Rungs 0–3 of the ladder (debug guide §2) once at session start with checkpoint A
   — this validates the day's setup; don't repeat for B/C.
2. **Safe-mode nav rollout** → if nav OK, safe-mode off, full rollout to the table.
3. **≥3 rollouts per checkpoint** (we learned from last session that n=1 per policy
   makes trends hard to trust). Fixed start pose, same cup/table placement.
4. Between checkpoints: only change `--checkpoint`/port. Nothing else.

### 2.2 What to score (per rollout, 0/1 each)
- NAV: reaches table without collision/intervention
- APPROACH: stops at a sensible manipulation distance/pose
- GRASP-ATTEMPT: reaches toward + closes on the cup (the R1 failure point)
- GRASP: lifts/holds the cup
- POUR: executes pour over the bowl
- Judgment call: note WHICH failure mode (skip-to-pour / think-done-back-away /
  freeze / wrong target) — this classifies the task-progress hypothesis.

### 2.3 The specific hypotheses this session tests
- **B vs A**: do adapted-ViT features close any of the appearance/perspective gap
  (esp. at the table)? Also: did full FT preserve DINO's viewpoint robustness (nav quality)?
- **C vs B**: same encoder, ± world-model co-training. If C attempts grasping where
  B skips ahead, that's direct evidence world supervision fixes task-progress
  blindness. This is the money comparison of the session.
- Proprio mode: run A with real proprio (its verified mode). B/C also real proprio.
  If anything looks proprio-suspect, A tolerates zeros (B/C degrade to ~0.05, usable
  as a diagnostic only).

### 2.4 Capture per rollout (for offline analysis; debug guide §7)
Live frames as sent (the img_compare dump you used before), the (32,49) chunks, and
video. With those three, I can attention-map + replay any failure the next day.

## 3. Known behavior notes for the WAM checkpoints
- World head is inert at inference — latency identical to BC checkpoints.
- wam_dinofull offline profile: clean 0.0143 / shift-flat / noise-flat / pzero 0.050.
- wam_res is offline-weakest on clean (0.026) but the most proprio-independent
  ResNet; treat as exploratory only.
- Offline in-dist cannot measure what WAM is supposed to buy (OOD/task-progress);
  hence the emphasis on the C-vs-B comparison above.
