# Model Card — RBY1 Whole-Body Image Policy (`crop100_2k` + `dino_lora_2k`)

> Full architectural/IO/hyperparameter reference for the deployed policies.
> Status & round history: `policy_training_status.md`. Deploy contract & serve/debug:
> `deployment_plan.md`. Last updated 2026-07-07.

## 1. High-level structure (HPT + flow matching)

```
obs.aria_image (224,224,3)      robot0_joint_pos(22)  hand_l/r_qpos(12,12)
        │                                 │ (normalized, quantile ±1)
   train/eval augs                 [noise σ=0.03rad/dim → clamp ±3 → dropout p=0.9]
        │                                 │
  ResNet-18 encoder                 state_* MLP stems (22/12/12 → 256)
  (49 tokens × 256)                       │
        │                          cross-attn: 16 learned latents each
  image MLP stem + cross-attn ──┐         │
  (16 latents × 256)            ├──► HPT trunk: 16 transformer blocks, d=256, 8 heads
                                │    (+ domain embedding, obs_horizon=1, drop_path 0.1)
                                │         │ action-token readout
                                └──► FMPolicy flow-matching head
                                     CrossTransformer denoiser: 6 blocks, cond 256,
                                     hidden 128, 4 heads → velocity field
                                          │  (10 Euler steps at serving)
                              actions (32, 49)  ← un-normalize (quantile) → robot
```

Total parameters: **31.8 M** (crop100, all trainable) / **42.6 M** (dino_lora, 21.0 M trainable — ViT backbone frozen).

## 2. Inputs (per control step, 10 Hz)

| key | shape / dtype | processing inside the model |
|---|---|---|
| `front_img_1` | (224,224,3) uint8 **BGR** | server: BGR→RGB, /255 → (1,3,224,224); NO resize. Then eval augs = ImageNet Normalize only. |
| `robot0_joint_pos` | (22,) float32 rad | = robot 26-D state `[4:26]` (base/wheel dims dropped). Order: torso(6), r_arm(7), l_arm(7), head(2). Quantile-normalized to ±1, then **clamped to ±3** (guards 2 near-static joints whose q99−q1≈0). |
| `hand_left_qpos` / `hand_right_qpos` | (12,) each, float32 rad | quantile-normalized, clamped ±3 |

- Normalization stats (q1/q99 per dim) are **baked into the ckpt** (`DataSchematic` via `save_hyperparameters`); the receiver sends RAW values, never normalizes.
- Robustness (measured): image translation ±20 px → no degradation; proprio error ≤3° (all joints) → no degradation; proprio all-zeros → near-clean (crop100 only; dino_lora REQUIRES real proprio).

## 3. Output

`{"actions": float32 (1, 32, 49)}` — **already un-normalized**, a 3.2 s trajectory at 10 Hz:

| block | idx | dim | meaning |
|---|---|---|---|
| base | 0:3 | 3 | per-step Δx, Δy, Δyaw — integrate by **plain cumsum** (frame-0 frame; never rotate by running yaw) |
| torso | 3:9 | 6 | joint position targets (rad) |
| head | 9:11 | 2 | joint targets |
| l_arm / r_arm | 11:18 / 18:25 | 7+7 | joint targets |
| l_hand / r_hand | 25:37 / 37:49 | 12+12 | joint targets |

Execution: receding horizon K=4–8 of the 32 steps, then re-query; interpolate between the 10 Hz targets at control rate (targets are trajectory samples, not instantaneous jumps).

## 4. Architecture details & important parameters

### Vision encoder
- **crop100_2k**: torchvision **ResNet-18**, ImageNet-pretrained, **fully fine-tuned**; truncated before avgpool/fc → (B,512,7,7) → reshape to **49 spatial tokens** → Linear 512→256. (`egomimic.models.hpt_nets.ResNet`)
- **dino_lora_2k**: timm `vit_small_patch14_dinov2.lvd142m` (ViT-S/14, 21.6 M) **frozen**, + **LoRA r=16 α=32** on attn qkv+proj of all 12 blocks (0.44 M trainable) → 256 patch tokens → Linear 384→256. (`egomimic.models.hpt_nets.DINOv2` + `LoRALinear`; serving needs `HF_HUB_OFFLINE=1` + warm HF cache.)

### Stems (per modality → 16 tokens each)
`MLPPolicyStem`, widths [256]; sinusoid positional embedding added over tokens; cross-attention pooling with **16 learned latents** (image: 8 heads × dim 64; proprio: 4 heads × dim 32), modality dropout 0.1. Proprio keys are batch-prefixed `state_*`; only keys present in `stem_specs` are consumed.

### Proprio robustness stack (order matters, in `preprocess_states`)
1. **Per-dim Gaussian noise** (train only): `noise_std_raw: 0.03` rad converted at init to normalized vector `clamp(2·0.03/(q99−q1), max=1.0)` per joint.
2. **Clamp ±3** (`proprio_clamp: 3.0`) — train AND eval (part of the deploy contract).
3. **Dropout p=0.9** (train only): whole proprio vector replaced by a learned null token for 90% of samples → policy is vision-first.

### Trunk
16 transformer blocks, embed 256, 8 heads, drop_path 0.1, learned domain embedding, `token_postprocessing: action_token`, observation_horizon 1, action_horizon 32.

### Head — flow matching (`FMPolicy`)
- Denoiser: `CrossTransformer` — 6 blocks, cond_dim 256, hidden 128, act 49×32, 4 heads, dropout 0.1, mlp_layers 4, mlp_ratio 4.
- Training: velocity-field regression, timestep distribution **beta**.
- Inference: **`num_inference_steps` forced to 10 at serving** (config value 50 is ignored by `serve_policy.py`). Sampling is stochastic → consecutive chunks differ slightly; use receding horizon / temporal ensembling.

### Image augmentation (train)
`RandomResizedCrop(224, scale=(0.31,1.0), ratio=(1,1), antialias)` — square crop removing 0–100 px, resized back → `ColorJitter(0.1,0.1,0.1,0.05)` → `Normalize(ImageNet)`. One param draw per batch (torchvision v1 behavior). **Eval/serving = Normalize only** (full frame is in-distribution because scale includes 1.0).

### Optimization
AdamW lr 1e-4, weight_decay 1e-4; CosineAnnealingLR **T_max 1400**, eta_min 1e-5 (repo convention: LR re-rises slightly after epoch 1400); **2000 epochs**, batch 32, 8 workers, bf16-mixed, single a40 (~21 h); ~100 optimizer steps/epoch; checkpoints every 200 epochs + `last.ckpt`.

### Data
`datasets/aria_egoposer_firm` — 30 demos, 16,566 frames @10 Hz (LeRobot v2), raw fisheye Aria RGB (NOT rectified), built from `aria_egoposer_train_firm.hdf5`. Action chunks assembled via `delta_timestamps` 0.0–3.1 s. Norm mode: quantile.

## 5. Checkpoints & measured profile (MAE rad, in-dist)

| | path (`/coc/flash7/czhang883/Documents/EgoVerse/logs/aria_egoposer_firm/…`) | clean | shift20 | pzero | noise ≤3° |
|---|---|---|---|---|---|
| ⭐ crop100_2k | `crop100_2k/checkpoints/last.ckpt` | 0.013 | 0.010 | 0.016 | 0.014 |
| A/B dino_lora_2k | `dino_lora_2k/checkpoints/last.ckpt` | 0.018 | 0.018 | 0.122 ✗ | 0.017–0.026 |

## 6. Serving stack

`egomimic/scripts/serve_policy.py --checkpoint <ckpt> --port 8000` → `EgoVersePolicy` → WebSocket + msgpack_numpy (openpi-compatible). On connect the server pushes metadata (embodiment, action_horizon/dim, camera/proprio keys). Health: `GET /healthz`.
⚠️ Serve from a checkout containing the **updated** `egomimic/algo/hpt.py` (proprio clamp) and `egomimic/models/hpt_nets.py` (DINOv2/LoRALinear) — both required for correct behavior of these checkpoints.

## 7. Config files (reproduce / retrain)
- Experiment: `egomimic/hydra_configs/experiments/wholebody_image/wb_img_proprio_v1_crop100.yaml` (/ `_v1_dino_lora.yaml`)
- Model: `egomimic/hydra_configs/model/experiments/wholebody_image/rby1_wb_img_proprio_act32_v1_crop100.yaml` (/ `_v1_dino_lora.yaml`)
- Data module: `egomimic/hydra_configs/data/experiments/wholebody_image/rby1_wb_img_act32.yaml`
- Launcher: `submit_wb_img_training.sbatch` via `wb_img_batch_workflow.sh` (auto-resume + HF_HUB_OFFLINE baked in)
