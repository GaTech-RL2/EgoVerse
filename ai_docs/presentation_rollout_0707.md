# RBY1 Whole-Body Policy from Human Video — Progress & First Hardware Rollouts

> Presentation source material, 2026-07-07. Written to be understandable without project
> background. Figures referenced here live in `/coc/flash7/czhang883/deliverables_0707/`.

---

## 1. One-paragraph project summary

We train a **whole-body robot policy** for the RBY1 mobile manipulator (mobile base +
torso + head + two 7-DoF arms + two 12-DoF hands) **entirely from human demonstrations**:
a person wearing an Aria headset performs the task (walk to a table, grasp a cup, pour);
an off-the-shelf body-tracking + retargeting pipeline converts the human motion into
robot joint trajectories; we then train an end-to-end neural policy that maps the
**egocentric camera image + the robot's joint readings** to the **next 3.2 seconds of
whole-body motion**. No teleoperation, no robot data collection. Dataset: 30
demonstrations, ~16.5k frames at 10 Hz, one task (approach table, grasp cup, pour).

---

## 2. Policy structure (the part to explain carefully)

### 2.1 Big picture — figure `arch_pipeline.png`

```
image ──► vision encoder ──► image stem ─┐
joints ─► [robustness layer] ─► stem ────┼──► HPT trunk ──► flow-matching head ──► 32×49 action chunk
hands ──► [robustness layer] ─► stems ───┘   (16 blocks)      (denoiser)             (3.2 s of motion)
```

**Inputs** (every 0.1 s):
- `front_img_1` — 224×224 RGB, the raw fisheye egocentric camera (NOT undistorted)
- `robot0_joint_pos` — 22 joint angles (torso 6, right arm 7, left arm 7, head 2)
- `hand_left_qpos` / `hand_right_qpos` — 12 + 12 finger joints

**Output**: an **action chunk** — 32 future steps × 49 dims = base motion (Δx, Δy, Δyaw)
+ joint targets for torso/head/arms/hands, covering 3.2 s. The robot executes the first
~0.5 s and re-queries (receding horizon).

### 2.2 What is HPT?

**HPT = Heterogeneous Pre-trained Transformer** (Wang et al., NeurIPS 2024) — an
architecture template for robot learning with three parts:
1. **Stems** — small per-modality adapters. Each input (image features, body joints,
   each hand) is passed through a small MLP and then **cross-attention**: 16 learnable
   query tokens "read" the modality and summarize it into 16 tokens. This makes every
   modality the same size and shape for the trunk, regardless of its raw dimension.
2. **Trunk** — one shared transformer (here: 16 blocks) that fuses all modality tokens.
   The "heterogeneous/pretrained" idea is that one trunk can serve many robots; we train
   ours from scratch for RBY1 only.
3. **Head** — task-specific output module. Ours is a **flow-matching head**: a small
   transformer that starts from random noise shaped like an action chunk and iteratively
   refines it (10 steps) into the final trajectory, conditioned on the trunk's output.
   Flow matching (a cousin of diffusion) is used because a whole 3.2 s trajectory is
   multi-modal — averaging isn't valid, sampling is.

### 2.3 The three vision-encoder variants — figure `arch_encoders.png`

The ONLY difference between our three checkpoints is the vision encoder and how it is
fine-tuned. Everything else (stems/trunk/head/training data/augmentation) is identical.

| | **crop100_2k** | **dino_lora_2k** | **dino_neck_2k** |
|---|---|---|---|
| Backbone | ResNet-18 (CNN, ImageNet-pretrained) | DINOv2 ViT-S/14 (self-supervised vision transformer) | same DINOv2 |
| Backbone size | 11.3 M | 21.6 M | 21.6 M |
| Fine-tuning strategy | **full fine-tuning** — every layer updated | **LoRA adapters** — backbone frozen; tiny low-rank matrices injected beside every attention layer | **frozen + neck** — backbone untouched; extra trainable transformer added after it |
| Trainable vision params | 11.3 M (100%) | 0.54 M (2.4%) | 1.9 M (8%) |
| Total policy params (trainable) | 31.8 M (31.8 M) | 42.6 M (21.0 M) | 44.0 M (22.3 M) |
| Why this design | strongest task fit; risks memorizing training pixels | adapts features *at every depth* while preserving DINO's pretrained representation (adapters removable) | zero risk to pretrained features; tests "was the head just too small?" |

**What is DINOv2?** A vision transformer pretrained by Meta on 142 M images with
self-supervision (no labels). Its features are known to be unusually robust to viewpoint
and appearance changes — the motivation for trying it here. ViT-S/14 = "small" variant,
splits the 224² image into 16×16 = 256 patches of 14×14 pixels.

**LoRA vs neck, in one breath:** LoRA changes *how the backbone computes* (a learned
low-rank correction inside each attention layer, `W x + B·A·x`, original weights
untouched); the neck changes *what happens after* the backbone (2 extra transformer
blocks on top of frozen multi-layer features). LoRA won: better accuracy at ¼ the
trainable parameters, because adapting features at depth beats adding capacity at the top.

### 2.4 Robustness layer on proprioception (why it exists)

Early checkpoints had a **"proprio cliff"**: joint readings off by just 0.6° from the
demos collapsed the policy (error ×16). Root cause: two torso joints never move in the
demos, so normalization divides by ≈0 and amplifies tiny offsets enormously. Fix (baked
into all current checkpoints):
- **per-joint noise** during training equal to 1.7° of real-world error,
- **clamp** on normalized values (train *and* deploy),
- **90% proprio dropout** — in 9 of 10 training samples the joints are hidden entirely,
  forcing the policy to work from vision; joints become a hint, not a crutch.
Also relevant: in this human-retargeted data, the recorded "proprio" is *identical* to
the action (there is no real robot state in the pipeline) — one more reason not to let
the policy lean on it.

### 2.5 Training recipe (identical across variants)

30 demos → 2000 epochs (~21 h, one A40). Image augmentation: random crop removing up to
100 px (then resize back) + color jitter — this is what makes the policy tolerant to
camera framing shifts. Actions normalized per-joint (quantile). Loss: flow-matching
velocity regression on the 32×49 chunk.

---

## 3. Offline evaluation (before hardware)

MAE against ground truth on training data (rad; 1° ≈ 0.017):

| condition | crop100 | dino_lora | dino_neck |
|---|---|---|---|
| clean inputs | 0.013 | 0.018 | 0.020 |
| image shifted 10–20 px | 0.010–0.012 (no degradation) | 0.018–0.019 | 0.021 |
| joints zeroed (vision only) | **0.016** | 0.122 ✗ | 0.112 ✗ |
| joints corrupted up to 3° | 0.014 (flat) | 0.017–0.026 | 0.021–0.025 |

Takeaways: all three are shift-proof and noise-proof (the augmentation + robustness
layer worked). Only crop100 can run vision-only. The two DINO variants failed the
"vision-only" goal identically → frozen DINO features alone can't carry this task from
30 demos; but LoRA gave the best DINO accuracy overall.

---

## 4. First hardware rollouts (2026-07-07) — 1 rollout per checkpoint

| checkpoint | navigation (drive to table) | manipulation (grasp + pour) | notes |
|---|---|---|---|
| **crop100_2k** | ✓ smooth: straight, clean left turn, reached table (safe mode off) | ✗ skipped grasping; went **directly to the pouring motion** without holding the cup | policy doesn't realize it isn't holding the cup |
| **dino_neck_2k** | ✗ trajectory offset, clipped a whiteboard (safe mode kept on) | not reached | weakest nav, consistent with weakest offline clean MAE |
| **dino_lora_2k** | ✓ **smoothest of all three**, very clean nav | ✗ same failure class: acted as if the task was already done — reached the table then **stepped back** without pouring | |

**Score: navigation 2/3 success · manipulation 0/2 attempts.**

### 4.1 Failure analysis — why navigation works but manipulation doesn't

1. **Observation gap** (see `side_by_side_training_vs_live.gif`): the live egocentric
   view differs from training in **perspective** (camera height/tilt — robot head vs
   human head) and **appearance** (scene, lighting, table contents). Navigation survives
   this: it needs only coarse scene layout, which our crop augmentation covers.
   Manipulation needs *fine-grained* cues — exactly what the gap destroys.
2. **No task-progress signal**: the policy is memoryless (single frame, no history).
   Grasp-vs-pour phases look similar from the ego view, and the proprio (hand joints)
   is either dropped (training) or off-manifold (live). So near the table the policy
   guesses the phase — one checkpoint guessed "pouring", the other guessed "done".
3. **Attention evidence** (see `attention_maps_grid.png`) — the three models "read" the
   image in strikingly different ways, and it lines up with the behavior:
   - **crop100 (ResNet)**: one focused blob on the scene center / room divider / table
     region — a *scene-layout* reader. Good for steering; but at the table its focus
     sits on the divider ABOVE the table, not on the cup — consistent with never
     registering "cup not grasped yet".
   - **dino_lora**: diffuse, near-global attention spread over the whole frame — a
     *holistic layout* reader. Explains its excellent, smooth navigation; and equally
     its table failure: no object-level focus that could signal task phase.
   - **dino_neck**: razor-sharp *object-centric* attention — locks precisely onto the
     hand, bowl, and cup in both live and training frames. Ironically the best "eyes"
     for manipulation but the worst navigator (it under-reads scene geometry) — and it
     never got to the table.
   - Attention stays structured on live frames for all three (it doesn't collapse), so
     the failure is not "the policy sees noise" — it's that appearance/perspective
     shifts change what the downstream layers *make of* what they see, and no model has
     the memory to disambiguate task phase.

### 4.2 What this suggests (next steps, in rough order of value)

1. **Close the appearance/perspective gap at the source**: collect (or re-render) demos
   with the camera at the robot's actual head pose, or add stronger viewpoint
   augmentation (3D-aware crops / homography jitter) during training.
2. **Give the policy task-progress information**: a short observation history (2–4
   frames) or an explicit phase/hand-state input, so "am I holding the cup?" is
   answerable.
3. **Sub-task decomposition**: keep the (working) navigation policy, hand off to a
   dedicated manipulation policy triggered near the table.
4. Keep LoRA-DINO as the vision backbone going forward — best live smoothness + best
   DINO accuracy; combine with (1)+(2).

---

## 5. Infrastructure recap (one slide)

- **Training**: SLURM (A40), auto-resuming jobs; 2000 epochs ≈ 21 h; HDF5 → LeRobot
  conversion pipeline; wandb logging.
- **Serving**: WebSocket + msgpack server; checkpoint is self-contained (normalization
  and robustness baked in); robot sends 4 obs keys, receives a 32×49 chunk.
- **Evaluation harness**: offline error-profile suite (clean / image-shift / proprio
  noise-gradient / vision-only) that predicted the hardware ranking correctly.
- **Docs**: model card, deployment plan + contract, debug guide, training-status ledger
  (all in `ai_docs/`, branch `rby1_aria_policy`).

## 6. Deliverables index (absolute paths)

```
/coc/flash7/czhang883/deliverables_0707/live_rollout_dino_lora.gif
/coc/flash7/czhang883/deliverables_0707/training_episode0.gif
/coc/flash7/czhang883/deliverables_0707/side_by_side_training_vs_live.gif
/coc/flash7/czhang883/deliverables_0707/arch_pipeline.png
/coc/flash7/czhang883/deliverables_0707/arch_encoders.png
/coc/flash7/czhang883/deliverables_0707/attention_maps_grid.png
/coc/flash7/czhang883/Documents/EgoVerse/ai_docs/presentation_rollout_0707.md   (this file)
```
Checkpoints: `/coc/flash7/czhang883/Documents/EgoVerse/logs/aria_egoposer_firm/{crop100_2k,dino_lora_2k,dino_neck_2k}/checkpoints/last.ckpt`
