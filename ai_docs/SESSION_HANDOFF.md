# Session Handoff — 2026-07-23 — HD ROUND CLOSED (offline)

> Read with: `ai_docs/fpp_deployment_note.md` (rollout guide, shortlist A–D) ·
> `ai_docs/pickplace_training_infra.md` (val history) · figures in
> `/coc/flash7/czhang883/deliverables_0707/` (arch_pipeline_hd, arch_hw_variants,
> arch_dino_usage, reliance_curves_0722, hd_loss_compare_0721, teleop_dataset_viz,
> teleop_ego_ep0.mp4). Branch `rby1_aria_policy`, all pushed.

## ROUND STATE: all 6 HD runs COMPLETED ep1999 (2026-07-23 13:30). No jobs running.

## FINAL GATE TABLE (old method: 10-frame probe on aria_fullpp; reliance = pzero/clean)
| run | trainable vision | best gate (clean / reliance @ep) | final @1999 | verdict |
|---|---|---|---|---|
| hd_wam3 | 11.3M ResNet ft + world-aux | **0.0247 / ×1.03 @1399** | 0.0271 / ×0.97 | **shortlist A** (avoid @1599 — transient-bad post-resume snapshot) |
| hd_resnet | 11.3M ResNet ft | **0.0241 / ×1.00 @1499** | 0.0286 / ×1.04 | **shortlist B** |
| hd_d3conv | 10.6M ConvNeck on frozen DINOv3-S | **0.0249 / ×1.44 @1399** | 0.0291 / ×1.19 | **optional D** (alt @1999 lower reliance, worse fit) |
| hd_nvs3dneck | 10.9M ConvNeck on frozen NVS-3D | 0.0259 / ×1.58 @1699 | 0.0277 / ×1.45 | not cleared (never reaches ×1.0); ≈ d3conv endpoint → 3D features gave NO offline edge |
| hd_d3lora | 0.44M LoRA on DINOv3-S | 0.0339 / ×1.89 @799 | 0.0319 / ×2.15 | closed negative (reliance worsens with training) |
| hd_nvs3d | 0.26M linear on NVS-3D | 0.0559 / ×2.90 @699 | (1499: 0.0367 / ×3.96) | closed negative (reliance WORSENS: 2.9→4.0) |
| wam3 (0.6-era) | — | 0.0134 / ×1.28 @1599 | — | **shortlist C** (only ckpt kept of 0.6 era) |

## THE CAPACITY LAW (round's main offline finding)
Proprio reliance under dropout-0.9 falls monotonically with trainable vision
capacity, and fine-tuned beats frozen+adapter at equal capacity:
ft-CNN 11M → ×1.0 · neck-on-frozen ~11M → ×1.4–1.6 · LoRA 0.44M → ×1.9–2.2 ·
linear 0.26M → ×2.9–4.0 (worsens with epochs — insufficient capacity ENTRENCHES
proprio). Dropout alone cannot cure frozen-feature policies; it needs somewhere
to push the load.

## HARDWARE VERDICTS (07-21/22, user's sim→real log)
- **A hd_wam3@1399 = only policy that reaches + GRASPS** (near-success ×2 from
  distance; one run ended by Aria USB drop, not policy). World-model aux is THE
  differentiator (A>B, same recipe otherwise).
- B stops short of table; C poor; d3lora over-rotates→collision; d3conv strays
  →side-table; glove/bare proprio-frozen.
- **NEAR-TABLE (skip-nav): ALL fail** — close-up hand/object view is OOD.
  → top next-round item: collect near-table / hand-in-view demos.

## LR-SCHEDULE FINDING
Cosine T_max=1400 → optimum lands ~ep1399–1499 in EVERY run; epochs 1400–2000 are
a slow drift (action-block losses +15–25%). Next round: T_max=2000 or stop ~1500.

## DISK / INFRA
flash7 volatile (66G–220G swings from external writers; user warned + given their
~1T old-project lever: im2Flow2Act 420G, dp3_sim2real 241G, human2any 294G,
miniconda3 76G). Our footprint minimal: 0.6-era ckpts DELETED (user-authorized;
only C wam3@1599 kept); HD ladders pruned to key ckpts (shortlist+final+last).
Watcher7 died of a bash bug (empty-NEWID array subscript) — replaced by SLURM
dependency chaining (`--dependency=afterany:<job>`), which is the better pattern
for future walls. Gate-eval pattern: tmp/eval_fpp_hdgate.py + tmp/fpp_hdgate.sbatch.

## NEXT-ROUND QUEUE (user decides)
1. **Near-table / hand-in-view data collection** (top item; cheap at-table demos).
2. **Composed config**: wam3-style future-DINO aux on other encoders (the aux is
   the hardware differentiator); e.g. hd_d3conv+aux or nvs3dneck+aux.
3. T_max=2000 in new configs.
4. Hardware: clean scored full run of A (Aria drop fixed robot-side with
   --hold-on-stale-frame); optional D test (does ×1.4 vs ×1.0 matter live?).
5. Deferred: delta-action space, teleop co-training, dense-grid PCA world targets.

## STANDING RULES
No login-node compute (sbatch/srun; light greps/plots/wandb OK) · 2000 epochs,
batch 32 · wandb personal (entity null) project sew_policy · teleop-val retired
for decisions (old-method gates + hardware only) · robot-side BGR contract per
deployment note · NVS-3D serving needs NVS3D_DIR (model.py + 0981at.pt).
