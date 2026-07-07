# RBY1 Whole-Body Image Policy — Training Status (single source of truth)

> **For any agent working in this repo.** Last updated 2026-07-05. This documents the
> full training-round history, every checkpoint on disk, measured error profiles, the
> code/config changes each round introduced, and what is currently running. The deploy
> contract + serve instructions live in `ai_docs/deployment_plan.md` (§1 contract,
> §8.0 current recommendation). Architecture/IO/hyperparameter reference:
> `ai_docs/policy_model_card.md`. Pipeline background: `ai_docs/data_pipeline_and_policy_guide.md`.

## TL;DR
- **Deployable now: `logs/aria_egoposer_firm/crop100_2k/checkpoints/last.ckpt`** —
  robust to image framing shifts, to ≥3° proprio error, AND to zeroed proprio. Send
  real live proprio (§1 contract; raw fisheye 224² uint8 BGR image).
- Round 4 in progress (2 jobs): give the DINO variant enough trainable vision capacity
  to also work vision-only.
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

## Currently running
- Nothing. R4 completed + gate-evaluated 2026-07-07. Deploy pair:
  **crop100_2k (primary, real proprio) + dino_lora_2k (A/B, real proprio required).**
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
