# dp3c ablation campaign — final results (2026-09-21)

25 single-variable variants vs V0 (=RBY1_dp3c_basket_v2), chuye basket data (53 eps),
2000 ep each, h=64. Metrics on the 8 held-out val episodes, serving path, first-32
horizon: mae32 / reliance deltas (dP,dG,dL = ΔMAE with proprio/global/local zeroed) /
jit_ratio (pred |d²| ÷ GT |d²|). Scoreboard: tmp/abl/results.csv. wandb: dp3c_ablation.
Checkpoints: logs/RBY1_abl_<ID>/abl/checkpoints/last.ckpt (all ep1999 except J3b ~1100).

## Full ranking (mae32; V0 = 0.1405)

G1 .1303(dP+.018 LEAKY) | H5 .1311(jit 1.8x) | D3 .1356 | G7 .1359 | G6 .1360 |
A5 .1388 | G4 .1389 | J4b .1393(jit 4.3x!) | H4 .1400 | J1 .1400 | V0 .1405 |
D1 .1408 | A3 .1421 | J2 .1423 | C3 .1425 | J3 .1434 | E4 .1435 | J5 .1437(dP+.006) |
A1 .1438 | A4 .1442 | H3 .1456 | C4 .1474 | A2 .1499 | B2 .1501 | J4 .1530

## Family verdicts

- **Augmentation (H)**: cloud aug HURTS offline — none (H5) is best overall clean run
  AND smoothest (1.8x vs 2.5x). Strong aug (H4) ≈ V0; point dropout (H3) hurts.
  Open question = does aug buy hardware robustness → H5-vs-V0 is THE hardware A/B.
- **Proprio (G)**: dropout-0.9 already made proprio useless; dropping it entirely
  (G7) or hands-only (G6) slightly HELPS and removes the deploy-side proprio gap.
  G1 (drop 0.5) best raw number but leaky (+0.018 dP) — probe, don't deploy blind.
- **Points (D)**: 512/512 (D3) ≥ 1024 ≥ 2048. Encoder budget can be halved for free.
- **Trunk/head (J)**: 8 blocks (J1) = 16 (V0); 32 (J2) worse. Head-256 (J5) no gain.
  LR: 3e-4 (J4) WORST run overall; 3e-5 (J4b) mid-table but jit 4.3x (undertrained
  smoothing not converged) — keep lr 1e-4. Batch 64 (J3) no gain.
- **Crops (A/B)**: V0's Z-slab 0.25–2.0 near-optimal; every change hurts except the
  gentle 2.0 m lateral ball (A5, tiny gain). Local ball 0.5 m (B2) = worst on board;
  1.5 m context matters.
- **Colour (C)**: XYZ-only (C4) clearly worse → colour IS load-bearing. Neutralization
  off (C3) ≈ V0 → the 0.30 m grey-out costs nothing; keep it for embodiment safety.
- **Frames (E)**: base-frame global (E4) worse — keep glass frame (head motion is
  signal, not noise).

## Composite candidates (next round, not yet trained)
H5+G6+D3+J1 stack: no aug, no hand proprio, 512 pts, 8-block trunk — each individually
≥V0 and the combination is ~4x cheaper; combinations aren't guaranteed additive, needs
one confirmation run.

## Late additions (2026-09-22)

- **J3b (batch 128, final at ~ep2000)**: 0.1424 — confirms the batch story (64 and 128
  both ≤ V0; keep batch 32). Notably smooth (jit 1.9x).
- **I3 (DINO world-model head)**: **0.1373**, zero proprio reliance, jit 2.3x — a real
  improvement over V0 (0.1405) with a clean profile. Its BEST checkpoint (ep799) hit
  0.1210 on a smaller probe set. Aux head is train-only (dropped at inference), so
  serving is identical to V0. Ckpt: logs/RBY1_abl_I3/abl/checkpoints/last.ckpt
  (repaired to ep1999 after an NFS-stale-last incident — see ops note below).
- **I2 (flow world-model head)**: FINAL **0.1399** — ≈ V0 (0.1405), i.e. neutral.
  The promising ep999 probe (0.1247) was small-probe-set luck / didn't hold to ep1999.
  VERDICT on world-model heads: DINO (appearance) aux is the one that helps (0.1373);
  flow (motion) aux is a wash. Both had zero proprio leakage and normal jitter.
  (I2's last.ckpt also repaired to ep1999 — it had accumulated last-v1..v4.)

**OPS NOTE (two silent-progress-loss modes found via I2/I3, both fixed):**
(1) preemption slices shorter than startup(~40 min norm-stats on 12k-D aux cols) +
100-epoch ckpt gap -> zero banked progress; (2) an NFS-locked last.ckpt makes
Lightning write last-v1.ckpt, and by-name resume/eval silently rolls back (I3 scored
0.2000/50x-jitter from the stale file before correction). Fixes: resume AND eval pick
the newest ckpt by mtime (tmp/abl templates + eval_final); repaired I3's last.ckpt.

## Hardware shortlist (Monday)
1. V0 control  2. H5  3. G6 (or G7)  4. D3  5. G1 (leakage probe)  6. A5.
All plain dp3c_dual serving contract; D3 = FPS→512 both streams; D1 = global 2048.
G4/G6/G7 consume fewer proprio keys (build obs from ckpt's own key list).
Full robot-side spec: the "INFERENCE INFRA — IMPLEMENTATION SPEC" block (sent 09-21).

## FINAL WAVE (2026-09-24) — program complete: 34/34 runs, 0 config failures

**Dropout dose-response (complete)** — mae32 / ΔMAE-without-proprio:
0.1 → 0.1138 / +0.086 · 0.3 → 0.1182 / +0.060 · 0.5 → 0.1303 / +0.018 ·
0.7 → 0.1372 / −0.001 · 0.9 (V0) → 0.1405 / −0.001 · none (G7) → 0.1359 / —.
Monotone: every step of dropout you remove buys offline MAE by copying proprio the
robot won't reproduce (at 0.1, the global cloud is literally unused: dG −0.000).
THE textbook offline-metric-lies exhibit. Clean settings: 0.7, 0.9, or none.

**Stacks**:
- **S1 no-aug + no-hands = 0.1292 — NEW BEST CLEAN POLICY** (beats both parents:
  H5 0.1311, G6 0.1360; dP −0.000, jit 2.2x). These two compose.
- S4 (+512pts) 0.1322, S2 (no-aug+512) 0.1336 — 512pts stays a COST win, not an
  accuracy win, in combination.
- S3 (no-aug + DINO head) 0.1339 — worse than H5 alone: the aug and world-model
  gains OVERLAP (both regularize the representation); they don't compose.

**DEPLOY RECOMMENDATION**: **S1** (`logs/RBY1_abl_S1/abl/checkpoints/last.ckpt`),
plain dp3c_dual serving contract, no hand-qpos inputs needed. Fall back H5 / V0
as controls. Dropout-0.3/0.1 checkpoints are hardware PROBES of the leakage
hypothesis, never deploy candidates.
