# Adapt3R 3D Observation Token Encoder — Design Doc

This document describes the EgoVerse port of Adapt3R's 3D observation
tokenisation. The encoder lives in
[`adapt3r_3d_encoder.py`](./adapt3r_3d_encoder.py) and plugs into HPT's
`encoder_specs` interface without touching the HPT backbone.

---

## 1. Motivation

The standard HPT image encoder (`egomimic.models.hpt_nets.ResNet`) flattens an
RGB frame into 2D spatial tokens. Adapt3R's contribution is to replace those
tokens with a single **3D-aware** token per camera per frame:

* Lift the depth map into a metric point cloud (pinhole back-projection).
* Encode each point's XYZ with a NeRF-style sinusoidal embedding.
* Fuse with co-located 2D backbone features (DINO/ResNet).
* FPS-downsample to a fixed point budget.
* Attention-pool to a single rich token.

That single 3D-aware token is then expanded to 16 trunk tokens by the existing
`MLPPolicyStem` cross-attention — so the HPT trunk, stems, heads, and stem
cross-attention are completely unchanged.

---

## 2. Pipeline

```
data["front_img_1"]   [B, T, I, 4, H, W]   (RGBD; HPT prepends T=I=1 dims)
   │
   ├── RGB[:3]  ── ImageNet-norm ── backbone (DINO or ResNet18)  ── [BN, F, h, w]
   │                                       │
   │                              flatten + Linear(F → hidden_dim)
   │                                       │
   │                                       ▼
   │                                 [BN, h*w, hidden_dim]   ◄── per-pixel feature
   │
   ├── Depth[3] ── lift_point_cloud_batch(K, E) ── [BN, H, W, 3]
   │                                              │
   │                          F.interpolate(nearest, (h, w))
   │                                              ▼
   │                                       [BN, h*w, 3]      ◄── per-pixel XYZ
   │
   ├── FPS over XYZ → indices ── gather(XYZ) and gather(features)
   │                                              │
   │                            ┌─────────────────┴─────────────────┐
   │                  NeRFSinusoidalPosEmb(XYZ)         RGB feature (already projected)
   │                            └──────────────────┬─────────────────┘
   │                                               concat
   │                                                ▼
   │                                       [BN, N, pc_in_dim]
   │
   └── AttentionExtractor (learned queries × N points) ──► [BN, 1, output_dim]
                                                                │
                                              reshape           ▼
                                              [B, T*I, output_dim]
```

`pc_in_dim = (do_pos + do_image) * hidden_dim + (3 if do_rgb else 0)`.

---

## 3. Backbone selection

| Backbone   | Source                                 | Feature dim | Spatial grid (input 360) | Notes |
|------------|----------------------------------------|-------------|--------------------------|-------|
| `dino`     | `torch.hub` `dinov2_vits14`            | 384         | 16 × 16 (input resized to 224) | Default. Frozen. Strong open-domain features. |
| `resnet18` | TorchVision ImageNet-pretrained        | 512         | 12 × 12                  | Lighter; useful for CPU tests. |

Both backbones are frozen by default. To enable fine-tuning:

```yaml
encoder_specs:
  front_img_1:
    _target_: egomimic.models.adapt3r_3d_encoder.Adapt3R3DEncoder
    backbone: dino
    finetune_backbone: true   # ← unfreezes the backbone
```

For the typical low-LR backbone fine-tuning convention, the encoder exposes
`get_finetune_param_groups(base_lr, backbone_lr_scale=0.1)`:

```python
groups = encoder.get_finetune_param_groups(base_lr=1e-4, backbone_lr_scale=0.1)
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
# → backbone params at 1e-5; everything else at 1e-4.
```

To wire this into HPT/PL we'll later need to:

1. Make `ModelWrapper.configure_optimizers()` aware of the encoder param-group
   helper (currently it just instantiates the optimizer over `self.parameters()`).
2. Optionally apply LoRA to the DINO ViT (only train low-rank adapters) — this
   is a simple wrap around `dino_vits14`'s attention/FFN linears.

---

## 4. Camera calibration

`intrinsics` (3 × 3) are configured for the original sensor resolution and
**auto-rescaled** to the encoder input resolution at init via the YAML fields
`image_size_orig` and `image_size_input`:

```
fx_in = fx_orig * (W_in / W_orig);  cx_in = cx_orig * (W_in / W_orig)
fy_in = fy_orig * (H_in / H_orig);  cy_in = cy_orig * (H_in / H_orig)
```

For RBY1 + Aria RGB:
* `image_size_orig: [2016, 1512]` (sensor)
* `image_size_input: [360, 360]` (rectified+resized lerobot stream)
* fx (rescaled): 877.96 × 360/2016 ≈ 156.7
* cx (rescaled): 989.30 × 360/2016 ≈ 176.7

⚠ **Caveat — fisheye distortion is ignored.** The Aria RGB cam is a
FISHEYE624 lens with 12 distortion coefficients; we use a pinhole
approximation. This is **fine for the verification dataset** (already
rectified) but must be addressed for raw-fisheye real-world deployment.
See `# TODO` markers in the encoder for the integration points.

---

## 5. Dummy depth

For RGB-only datasets, set `dummy_depth: <metres>` (e.g. `1.0`). The encoder
overwrites the depth channel with that constant, producing a flat plane at
z = `dummy_depth`. The 3D structure is uninformative but every downstream
component (shape, dtype, device, gradient flow, integration with HPT trunk)
is exercised end-to-end. **This is the verification mode** — see Section 7.

When a depth key is configured in `depth_key_map` but absent from the batch,
`HPT._robomimic_to_hpt_data` inserts a zero placeholder; `dummy_depth`
overwrites it.

---

## 6. Fidelity vs. original Adapt3R

| Component                              | Port         | Original    | Match |
|----------------------------------------|--------------|-------------|-------|
| `NeRFSinusoidalPosEmb`                 | bitwise      | reference   | ✅ verified by unit test |
| `AttentionExtractor`                   | structurally identical (Q/K/V MLPs, learned head queries, scaled dot-product) | reference | ✅ |
| `depth2fgpcd_batch`, `lift_point_cloud_batch`, `batch_transform_point_cloud` | identical math (float32 instead of original float16 buffer) | reference | ✅ |
| Farthest Point Sampling                | pure-torch O(N×k) | DGL `farthest_point_sampler` | ⚠ Same start_idx=0, deterministic, output set is equivalent |
| RGB backbone                           | DINOv2 ViT-S/14 (default) **or** ResNet-18 | ResNet-18 / 50 / CLIP / "fusion" | ⚠ Subset; user-selectable |
| Feature Pyramid Network                | omitted (single-scale) | FPN over a single scale | ⚠ FPN with one scale = pass-through; effectively equivalent |
| Scene bounding-box crop                | not implemented | LIBERO/MimicGen specific | ❌ TODO |
| Hand-frame transform (`hand_frame=True`)| not implemented | core Adapt3R cross-embodiment trick | ❌ TODO |
| Language fusion (`do_lang`)            | omitted | optional CLIP/lang projection | n/a — HPT has its own lang stem |

### Intentional scope reductions

* **No hand-frame**: the user wants to verify the *3D-aware tokenization*
  contribution; the cross-embodiment hand-frame transform requires per-batch
  hand poses and is a separate research question.
* **No scene crop**: the boundary values are LIBERO-specific. Generic
  bounding-box crop is on the TODO list.
* **No FPN**: a single-scale FPN over a single scale is a no-op; we
  project the last spatial map directly via `nn.Linear(feat_dim → hidden_dim)`.

---

## 7. Verification plan & status

### Phase 1 — Unit tests (✅ done)

`egomimic/tests/test_adapt3r_3d_encoder.py` (19 tests):

* Shape contract for all `(B, T, I, H, W)` configs.
* NeRF embedding bitwise-equal vs. original Adapt3R import.
* Pinhole roundtrip (project → unproject; recover XYZ).
* FPS endpoint test (line/grid).
* FPS clamp warning when `num_points > h*w`.
* Backbone freezing & gradient flow.
* `get_finetune_param_groups` returns correctly scaled LRs.
* DINO smoke test (downloads ~85 MB; marked `slow`).

```bash
cd /home/droid_robot/zhenyang/EgoVerse && source emimic/bin/activate
emimic/bin/python -m pytest egomimic/tests/test_adapt3r_3d_encoder.py -v
# 19 passed in ~3 s
```

### Phase 2 — Visualization (✅ done)

`egomimic/scripts/visualize_adapt3r_3d_encoder.py` renders, for one frame:

1. Input RGB
2. Backbone feature map (PCA-RGB over the spatial grid)
3. NeRF position embedding at FPS points (PCA-RGB, 3D scatter)
4. Fused [pos | rgb_feat] (PCA-RGB, 3D scatter)
5. Per-head attention weights overlaid on the RGB image (image-plane projection)
6. Output token statistics (mean/std/min/max) + rescaled K values

```bash
emimic/bin/python egomimic/scripts/visualize_adapt3r_3d_encoder.py \
  --dataset /home/droid_robot/zhenyang/EgoVerse/datasets/RBY1_egoengine_mustard_cropped_right_arm_eef_hand \
  --episode 0 --frame 30 --backbone dino \
  --out /tmp/viz_adapt3r3d_dino.png --save-ply
```

### Phase 3 — Full HPT integration (✅ done)

```bash
export TMPDIR=/tmp
emimic/bin/python egomimic/trainHydra.py \
  --config-name=train_rby1_egoengine_adapt3r3d \
  trainer=debug logger=debug \
  data.train_datasets.dataset1.datasets.rl2_lab.folder_path=$DATASET \
  data.valid_datasets.dataset1.datasets.eth_lab.folder_path=$DATASET
# 4 epochs, 20 train batches + 12 val batches, last.ckpt saved.
```

---

## 8. TODOs (intentionally deferred)

* [ ] **FISHEYE624 distortion** for Aria RGB. Either undistort upstream
      (rectified images already exist in this dataset) or replace
      `depth2fgpcd_batch` with a fisheye-aware unprojection.
* [ ] **Scene bounding-box crop** — add `crop_bounds: [[xmin,ymin,zmin],
      [xmax,ymax,zmax]]` config and apply a mask in `forward()`.
* [ ] **Hand-frame transform** — requires `obs.hand_mat_inv` per batch.
      Adapt3R's `_crop_point_cloud(... hand_mat_inv=...)` is the reference
      implementation.
* [ ] **DINO LoRA** for parameter-efficient fine-tuning (will pair with
      `get_finetune_param_groups`).
* [ ] **Real depth** — once the dataset includes a depth stream (e.g. from
      FoundationStereo or a depth sensor), drop `dummy_depth` and add the
      actual lerobot key to the DataSchematic.

---

## 9. Files

| File | Role |
|------|------|
| `egomimic/models/adapt3r_3d_encoder.py` | Encoder implementation (~600 lines, self-contained). |
| `egomimic/algo/hpt.py` | Plumbing: `depth_key_map` + dummy-depth fallback in `_robomimic_to_hpt_data`. |
| `egomimic/hydra_configs/model/experiments/rby1_egoengine/rby1_egoengine_adapt3r3d_dino.yaml` | Model config with DINO + Aria K. |
| `egomimic/hydra_configs/train_rby1_egoengine_adapt3r3d.yaml` | Top-level train config. |
| `egomimic/tests/test_adapt3r_3d_encoder.py` | 19 unit tests. |
| `egomimic/scripts/visualize_adapt3r_3d_encoder.py` | Visualisation tool. |
| `experiments/2026-04-28_adapt3r3d_verification.md` | Experiment log (this run). |
