# RBY1 Whole-Body Image Policy — Training Status (single source of truth)

> **For any agent working in this repo.** Last updated 2026-07-16. This documents the
> full training-round history, every checkpoint on disk, measured error profiles, the
> code/config changes each round introduced, and what is currently running. The deploy
> contract + serve instructions live in `ai_docs/deployment_plan.md` (§1 contract,
> §8.0 current recommendation). Architecture/IO/hyperparameter reference:
> `ai_docs/policy_model_card.md`. Pipeline background: `ai_docs/data_pipeline_and_policy_guide.md`.

## TL;DR
- **Deployable now: `logs/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt`** —
  robust to image framing shifts, to ≥3° proprio error, AND to zeroed proprio. Send
  real live proprio (§1 contract; raw fisheye 224² uint8 BGR image).
- Rounds 3–7 complete (no jobs running). **R7 breakthrough: frozen-backbone + conv-neck
  policies (`d3_convneck_2k`, `lingbot_convneck_2k`) are the first non-ResNet ckpts to
  PASS the vision-only gate (pzero 0.026 / 0.023 ≤ 0.03).** Full 12-ckpt comparison chart:
  `/coc/flash7/czhang883/deliverables_0707/results_overview_all_rounds.png`.
- Data: `datasets/aria_egoposer_firm` (30 clean demos, 16,566 frames, LeRobot v2),
  built from `.../SEW-Geometric-Teleop/artifacts/aria_egoposer/aria_egoposer_train_firm.hdf5`.
  The older `datasets/aria_egoposer` is CONTAMINATED (right-arm IK corruption, 10/30 eps).

## Round history

| round | runs (logs/…/checkpoints/last.ckpt) | recipe delta | outcome |
|---|---|---|---|
| R0 06-23 | `aria_egoposer/{vanilla,hier}` | 1000 ep, contaminated data | ❌ do not use |
| R1 07-01 | `aria_egoposer_firm/vanilla`, `aria_egoposer_firm_v2/v2_hist_traj` | clean data; V2 added joint-hist+base-traj obs | data fixed; no robustness; V2 discarded by user |
| R2 07-03 | `aria_egoposer_firm/{v1_crop_2k,v1_dino_2k}` | +RandomResizedCrop 0-50px, dropout 0.8, 2000 ep | shift-brittleness FIXED; **proprio cliff found** (σ=0.01 rad → MAE 0.234) |
| R3 07-04 | `aria_egoposer_firm/{crop100_2k,dino100_2k}` | +crop 0-100px, dropout 0.9, **noise_std_raw 0.03**, **proprio_clamp 3.0** | **cliff FIXED** (crop100: noise 0.014 flat to 3°); dino still fails vision-only (pzero 0.157) |
| R4 07-05→07 | `aria_egoposer_firm/{dino_neck_2k,dino_lora_2k}` | +vision capacity: neck (1.9M, backbone untouched) vs LoRA r16 (0.54M, adapters on attn) | **vision-only goal FAILED both** (pzero 0.112/0.122 vs ≤0.03 gate) → frozen ViT-S can't carry the task alone; side win: **dino_lora_2k = best DINO overall** (clean 0.0177, shift+noise flat) → new A/B ckpt |

## Measured error profiles (MAE rad vs GT, 10 frames, in-dist; eval scripts in /coc/flash7/czhang883/tmp/)

| ckpt | clean | shift10/20px | pzero | noise σ=.01/.02/.03/.05 | verdict |
|---|---|---|---|---|---|
| crop100_2k ⭐ | 0.0126 | 0.0121/0.0128 | 0.0155 | 0.0142 at all σ | deploy primary |
| dino_lora_2k | 0.0177 | 0.0192/0.0182 | 0.1218 ✗ | 0.017-0.026 | **A/B backup (best DINO)**; never zero its proprio |
| dino_neck_2k | 0.0201 | 0.0209/0.0213 | 0.1118 ✗ | 0.021-0.025 | superseded by dino_lora |
| dino100_2k | 0.0245 | 0.0253/0.0241 | 0.1571 ✗ | 0.023-0.028 | superseded by dino_lora |
| v1_crop_2k | 0.0119 | 0.0106/0.0104 | 0.0145 | 0.234 CLIFF ✗ | vision-only mode only |
| v1_dino_2k | 0.0134 | 0.0140/0.0132 | 0.158 ✗ | 0.039→0.112 | superseded |
| firm vanilla (1k ep) | 0.0091 | 0.0190/0.0345 ✗ | 0.0167 | 0.139 @.05 ✗ | baseline only |

## Code changes (all in this checkout, all backward-compatible / default-off)
- `egomimic/models/hpt_nets.py`: **`DINOv2`** encoder class (timm
  `vit_small_patch14_dinov2.lvd142m`, frozen, patch tokens → Linear→256; drop-in for
  `ResNet`); optional `feature_indices` (multi-layer concat), `neck_blocks`
  (trainable transformer neck), `lora_rank/lora_alpha` (**`LoRALinear`** adapters on
  attn qkv+proj; base weights untouched). Needs `HF_HUB_OFFLINE=1` + warm HF cache
  (`/coc/flash7/czhang883/.cache/huggingface`) — set in the sbatch.
- `egomimic/algo/hpt.py`: per-key **`noise_std_raw`** (rad) → per-dim normalized noise
  vector `clamp(2σ/(q99-q1), max=1)` built from schematic stats at init (cause of the
  cliff: 2 near-static joints have quantile range ≈0, exploding small raw offsets);
  **`proprio_clamp`** (±3) on normalized proprio at train AND eval (baked into ckpt
  forward). getattr-guarded: old ckpts unpickle whole HPTModel objects without the
  new attrs.
- `submit_wb_img_training.sbatch`: auto-resume from `last.ckpt` on resubmission;
  `HF_HUB_OFFLINE=1` for Step 3.

## Configs map (egomimic/hydra_configs/)
- experiments/wholebody_image/`wb_img_proprio_{vanilla, v1_crop, v1_dino, v1_crop100, v1_dino100, v1_dino_neck, v1_dino_lora, v2_traj}.yaml`
  (+ matching `model/experiments/wholebody_image/rby1_wb_img_proprio_act32_*.yaml`,
  data module `data/experiments/wholebody_image/rby1_wb_img_act32.yaml`).
- Launch: `sbatch --export=ALL,DATASET_NAME=aria_egoposer_firm,RAW_DATA_PATH=<hdf5>,TRAIN_CONFIG=experiments/wholebody_image/<exp>,DESCRIPTION=<desc> submit_wb_img_training.sbatch`
  (2000 epochs, ~21 h on one a40, overcap; add `--exclude=...,sonny` — its GPU-0 was
  wedged on 07-03: "CUDA device busy" at init).
- WandB: personal account (entity null), project `sew_policy`, run id `aria_egoposer_firm_<desc>`.

## ROUND 5 RESULTS (2026-07-10 — encoder ablations, same R3 recipe)

| ckpt (`logs/aria_egoposer_firm/…/last.ckpt`) | trainable vision | clean | shift10/20 | pzero | noise ≤3° |
|---|---|---|---|---|---|
| **dino_full_2k** (ViT-S fully unfrozen) | 21.7M (100%) | **0.0119** | 0.013/0.015 ✓ | **0.067** ✗ | 0.013-0.016 ✓ |
| dinob_lora_2k (frozen ViT-B + LoRA r16) | 1.1M | 0.0155 | 0.017/0.020 ✓ | 0.098 ✗ | 0.017-0.020 ✓ |
| dino_mlp_2k (frozen ViT-S + 4-block res-MLP head) | 2.2M | 0.0180 | 0.021/0.019 ✓ | 0.091 ✗ | 0.017-0.022 ✓ |

**Findings:** (1) **dino_full is the new best DINO** — full fine-tuning at lr 1e-4 did NOT
wreck the ViT (heavy crop aug regularized it); it ties crop100's clean accuracy (0.0119
vs 0.0126) and has the best DINO vision-only score. (2) Clear monotone trend: the more
the backbone adapts (frozen < adapters < full FT), the better BOTH clean and pzero.
(3) Vision-only gate (≤0.03) still unbroken by any DINO (best 0.067 vs ResNet 0.016).
(4) ViT-B+LoRA beats ViT-S+LoRA on everything → scale helps, but less than unfreezing.
**Caveat:** shift-flatness only tests translation; whether full-FT preserved DINOv2's
VIEWPOINT invariance (the reason to use DINO) is unknown until hardware A/B.
**New deploy A/B suggestion: crop100_2k (primary) + dino_full_2k (challenger).**

## ROUND 6 RESULTS (2026-07-13 — EgoWAM-style world-model co-training, arXiv:2607.08436)
World head (train-only, dropped at inference) predicts future (t+1.0s) DINOv2-B 4x4x768
features via flow matching, lambda=1, through the auxiliary-head machinery. Dataset:
`datasets/aria_egoposer_firm_wam` (precomputed targets, add_dino_wm_target.py).

| ckpt (`logs/aria_egoposer_firm_wam/…/last.ckpt`) | clean | shift20 | pzero | noise 3° |
|---|---|---|---|---|
| wam_res_2k (ResNet + world head) | 0.0257 | 0.0109 | 0.0256 | 0.0251 |
| wam_dinofull_2k (ViT-S FT + world head) | 0.0143 | 0.0126 | **0.0504** | 0.0165 |

**Findings:** (1) World co-training improved the ViT's vision-only score 25% (0.067→0.050,
best DINO-family yet) at a small clean cost (0.012→0.014). (2) It HURT the ResNet
(clean doubled to 0.026; the 11M encoder likely lacks capacity for both tasks at
lambda=1 on 30 demos) — though wam_res became notably proprio-independent (pzero≈clean)
and oddly shift-favoring. (3) No offline in-dist metric beats crop100 — but EgoWAM's
claimed gains are OOD/hardware generalization, which offline in-dist CANNOT measure;
wam_dinofull is the right candidate to A/B on hardware against dino_full/crop100.
Ideas if pursued further: lambda<1 for ResNet, or a wider trunk.

## ROUND 7 RESULTS (2026-07-16 — new backbones: DINOv3 + LingBot-Vision, frozen; same R3 recipe)
Backbones: DINOv3-S/16 (timm `vit_small_patch16_dinov3`, 384-d) and LingBot-Vision ViT-L/16
(`robbyant/lingbot-vision-vit-large`, 1024-d, masked-boundary-modeling SSL). Per user
instruction backbones stay FROZEN; capacity added on top: LoRA r16 (adapters) or
**ConvNeck** (`hpt_nets.ConvNeck`: token grid → 14×14 map → 9 torchvision BasicBlocks
@256ch ≈ 10.8M ≈ ResNet-18 budget → tokens).

| ckpt (`logs/aria_egoposer_firm/…/last.ckpt`) | trainable vision | clean | shift10/20 | pzero | noise ≤3° |
|---|---|---|---|---|---|
| d3_lora_2k (frozen DINOv3-S + LoRA r16) | 0.54M | 0.0141 | 0.013/0.015 ✓ | 0.0556 ✗ | 0.014-0.015 ✓ |
| **d3_convneck_2k** (frozen DINOv3-S + 9-block ConvNeck) | 10.8M | 0.0182 | 0.017/0.016 ✓ | **0.0261 ✓** | 0.018-0.019 ✓ |
| **lingbot_convneck_2k** (frozen LingBot-L + 9-block ConvNeck) | 11.2M | 0.0198 | 0.016/0.014 ✓ | **0.0230 ✓** | 0.018-0.020 ✓ |

**Findings:** (1) **First non-ResNet ckpts to PASS the vision-only gate (≤0.03)** —
both ConvNeck variants, with backbones fully frozen. The ResNet's vision-only advantage
was therefore the *conv prior + trainable capacity*, NOT backbone fine-tuning; and it
can be had while preserving the frozen SSL backbone's (viewpoint-)invariances.
(2) Backbone-quality effect at fixed adapter capacity: DINOv3+LoRA halves DINOv2+LoRA's
pzero (0.122→0.056) and beats it on every condition — gram-anchored dense features
transfer better. (3) Trade-off: ConvNeck costs ~0.004-0.006 clean MAE vs the best FT
models; lingbot_convneck has the best pzero (0.0230) and shift profile of any DINO-family
ckpt but the weakest clean of the three. (4) Hardware A/B candidates from R7:
**d3_convneck_2k / lingbot_convneck_2k** (frozen invariances + passes vision-only) vs
crop100 — offline in-dist cannot rank them further.

## Previous round-5 run notes (submitted 2026-07-08)
- job 3441899 → `logs/aria_egoposer_firm/dino_mlp_2k` — frozen ViT-S + deep residual-MLP
  projection head (4 blocks 256↔1024; 2.2M vision-trainable, per-token, no spatial mixing)
- job 3441900 → `logs/aria_egoposer_firm/dino_full_2k` — ViT-S FULLY unfrozen (21.7M;
  lr=1e-4 same recipe — aggressive for a ViT, treated as an ablation finding either way)
- job 3441901 → `logs/aria_egoposer_firm/dinob_lora_2k` — frozen ViT-B/14 (86M) + LoRA r16
  (1.1M trainable) — scales up the R4 winner
- Gate on completion: same profile eval (clean/shift/pzero/noise) + compare vs dino_lora_2k.
- Deferred: Adapt3R-style 3D representations until point-cloud infra exists (RGB first).

## Hardware rollouts (2026-07-07, 1 rollout each — see ai_docs/presentation_rollout_0707.md)
- crop100_2k: nav ✓ (smooth, safe-mode off) / manip ✗ (skipped grasp, went straight to pour)
- dino_neck_2k: nav ✗ (offset, clipped whiteboard) / not reached
- dino_lora_2k: nav ✓ (smoothest of all) / manip ✗ (acted as if task done, backed away)
- Diagnosis: perspective+appearance gap (human-head vs robot-head views) + no task-progress
  signal (memoryless policy). Attention maps: ResNet=scene-blob, LoRA=diffuse-global,
  neck=object-centric — none has both scene and object reading.

## Previous deploy pair (still current until R5 gate)
- **crop100_2k (primary, real proprio) + dino_lora_2k (A/B, real proprio required).**
- If vision-only DINO is ever revisited: both a 1.9M trainable neck and r16 LoRA failed
  the pzero gate identically → next rung is unfreezing the last 2 ViT blocks (~3.6M,
  real invariance risk) or a frozen ViT-B/14; weigh against crop100 already covering
  vision-only at 0.016.

## Known open items
- Deploy-side: live image path must send raw fisheye resized to a 224² square in BGR
  (live publisher previously sent 1280×720) — see deployment_plan.md §1/§7.
- V2 (joint-hist + base-traj obs) discarded but documented in deployment_plan.md §9
  if odometry input is ever revisited; its `base_traj` convention = plain cumsum of
  base deltas from the rollout's first frame (NOT body-frame yaw rotation).
