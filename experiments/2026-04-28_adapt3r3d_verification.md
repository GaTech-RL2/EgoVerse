# 2026-04-28 — Adapt3R 3D Encoder Verification (DINOv2 + dummy depth)

**Branch:** `feat/adapt3r-3d-encoder` &nbsp;·&nbsp;
**Goal:** verify the Adapt3R 3D-tokenization port is correct (math + plumbing
+ HPT integration) on the egoengine dataset. Absolute depth correctness is
out-of-scope — depth is replaced with a constant via `dummy_depth: 1.0`.

---

## Setup

* Dataset: `/home/droid_robot/zhenyang/EgoVerse/datasets/RBY1_egoengine_mustard_cropped_right_arm_eef_hand`
  * 61 episodes, 6702 frames @ 10 Hz
  * RGB-only, 360 × 360, already rectified
* Camera: Aria RGB (FISHEYE624, 2016×1512 native, fx=fy=877.96, cx=989.30, cy=744.17).
  Fisheye distortion **ignored** (pinhole approximation; intrinsics rescaled to 360×360).
* Encoder: `Adapt3R3DEncoder(backbone="dino", dino_img_size=224, num_points=256, hidden_dim=60, dummy_depth=1.0)`
* Backbone: DINOv2 ViT-S/14 (frozen, ImageNet1k weights via `torch.hub`)

---

## What was done

### Phase 1 — Unit tests (✅)

```
emimic/bin/python -m pytest egomimic/tests/test_adapt3r_3d_encoder.py -v
# 18 passed (fast)
emimic/bin/python -m pytest egomimic/tests/test_adapt3r_3d_encoder.py::test_encoder_dino_shape_contract -v
# 1 passed (DINO download)
```

19/19 tests pass, including a **bitwise differential test** of
`NeRFSinusoidalPosEmb` against the original Adapt3R implementation imported
from `/home/droid_robot/zhenyang/Adapt3R`. Pinhole projection round-trip and
FPS endpoint correctness confirmed.

### Phase 2 — Visualization (✅)

```
emimic/bin/python egomimic/scripts/visualize_adapt3r_3d_encoder.py \
  --dataset /home/droid_robot/zhenyang/EgoVerse/datasets/RBY1_egoengine_mustard_cropped_right_arm_eef_hand \
  --episode 0 --frame 30 --backbone dino \
  --out /tmp/viz_adapt3r3d_dino.png --save-ply
```

**Output (frame 30, episode 0):**

* RGB shape: `(360, 360, 3)`
* Backbone feature dim: 384 (DINOv2 ViT-S/14)
* Encoder output token: shape `(1, 1, 256)`, mean = +0.0000, std = 0.0395
* PNG: `/tmp/viz_adapt3r3d_dino.png`
* PLY: `/tmp/viz_adapt3r3d_dino.ply` (full-res lifted cloud, 360×360 = 129 600 points)

**Cross-check with ResNet-18 backbone:**

```
emimic/bin/python egomimic/scripts/visualize_adapt3r_3d_encoder.py \
  --dataset $DS --episode 0 --frame 30 --backbone resnet18 --num-points 100 \
  --out /tmp/viz_adapt3r3d_resnet.png
# Output token: shape=(1, 1, 256), mean=+0.0032 std=0.0309
# feat_dim=512
```

### Phase 3 — Full HPT integration (✅)

```
export TMPDIR=/tmp
emimic/bin/python egomimic/trainHydra.py \
  --config-name=train_rby1_egoengine_adapt3r3d \
  trainer=debug logger=debug \
  data.train_datasets.dataset1.datasets.rl2_lab.folder_path=$DS \
  data.valid_datasets.dataset1.datasets.eth_lab.folder_path=$DS
```

* 4 epochs, 20 training batches, 12 validation batches, no NaN, no exception
* `last.ckpt` saved to
  `logs/RBY1_egoengine_adapt3r3d/train_2026-04-28_17-01-52/checkpoints/last.ckpt`
* Encoder forward → backbone (DINO) → FPS → AttentionExtractor → MLPPolicyStem
  cross-attn → trunk transformer → flow-matching head all completed cleanly.

---

## Findings

1. **Math is faithful** — bitwise match against original Adapt3R for the NeRF
   embedding; structurally identical AttentionExtractor; pinhole projection
   round-trip recovers depths within 1e-4.
2. **DINO patch features are well-behaved** at 360 → 224 resize. Token
   statistics (mean ≈ 0, std ≈ 0.04) are sane and similar to the ResNet path.
3. **HPT plumbing works without backbone changes.** The new
   `depth_key_map` config wires RGB+depth concatenation in the outer `HPT`
   class; the depth-key-missing fallback inserts a zero placeholder that the
   encoder fills with `dummy_depth`.
4. **FPS clamp warning fires** as expected when `num_points` > spatial grid;
   for DINO at 16×16 = 256 points, setting `num_points: 256` matches exactly.

## Risks / open issues

* **Fisheye distortion ignored** — fine for the rectified verification dataset
  but must be addressed for raw Aria fisheye deployment. TODO marked in code.
* **Depth is constant** — encoder is "doing something" but the 3D structure is
  unused in this verification. Real depth (sensor or FoundationStereo) is the
  next step.
* **DINO finetune wiring** — `get_finetune_param_groups()` exists on the
  encoder but the PL `ModelWrapper.configure_optimizers()` does not yet route
  to it. Out-of-scope for this run; tracked in the design doc.

---

## Artefacts

| Path | Description |
|------|-------------|
| `egomimic/models/adapt3r_3d_encoder.py` | Encoder implementation. |
| `egomimic/models/adapt3r_3d_encoder.md` | Design doc (long-form). |
| `egomimic/algo/hpt.py` | Plumbing edits (`depth_key_map` fallback). |
| `egomimic/hydra_configs/model/experiments/rby1_egoengine/rby1_egoengine_adapt3r3d_dino.yaml` | Model config. |
| `egomimic/hydra_configs/train_rby1_egoengine_adapt3r3d.yaml` | Train config. |
| `egomimic/tests/test_adapt3r_3d_encoder.py` | 19 unit tests. |
| `egomimic/scripts/visualize_adapt3r_3d_encoder.py` | Visualisation. |
| `/tmp/viz_adapt3r3d_dino.png`, `/tmp/viz_adapt3r3d_dino.ply` | Sample outputs. |
| `logs/RBY1_egoengine_adapt3r3d/train_2026-04-28_17-01-52/checkpoints/last.ckpt` | Smoke-train checkpoint. |

---

## Next steps

1. Add real depth (FoundationStereo or hardware) → swap `dummy_depth: null` in
   the model YAML and add a depth lerobot key to the DataSchematic.
2. Wire `get_finetune_param_groups()` into `ModelWrapper.configure_optimizers()`
   for low-LR DINO fine-tuning.
3. Implement the optional `crop_bounds` config (TODO in encoder).
4. Compare downstream task success (BC eval) with the standard ResNet HPT
   encoder — once real depth is available.
