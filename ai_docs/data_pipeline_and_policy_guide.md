# EgoVerse — RBY1 Data Pipeline & Policy Guide

> One-stop guide for the RBY1 work in this repo: what the data looks like, how to
> turn raw recordings into trainable data, how to train, and how to inspect /
> serve / sanity-check a trained policy.
>
> Companion to the root `CLAUDE.md` (architecture) and `rby1.md` (Zhenyang's
> original notes). When they disagree, **the code wins** — citations below point
> at the source of truth.

---

## 0. The mental model (read this first)

```
 RAW RECORDING            INTERMEDIATE              FINAL TRAINING DATA         TRAIN            SERVE / CHECK
 ─────────────            ────────────              ───────────────────         ─────            ─────────────
 CSV (SEW body pose)  ─┐
 teleop HDF5          ─┼─►  robomimic HDF5   ──►   LeRobot dataset on disk  ──►  trainHydra  ──►  .ckpt  ──►  serve_policy
 DexMimicGen HDF5     ─┘    (group: data/demo_i)   (parquet + meta + mp4)        (Lightning)      (norm baked in)  + test client
 Aria VRS  ───────────────────────────────────►   (aria_to_lerobot, direct)
```

**The single most important fact:** **HDF5 is an intermediate, not the training
format.** Training reads a **LeRobot v2 dataset** (per-episode `*.parquet` +
`meta/*.json` + optional `*.mp4`). Every RBY1 path converts HDF5 → LeRobot first.
You were right to be suspicious of the HDF5.

There are **two-to-three stages** between a raw file and trainable data:

1. **(SEW only) Step 0** — `CSV → HDF5` (MuJoCo IK in the external SEW repo).
2. **Step 1** — `HDF5 → LeRobot "raw"` via the *universal* converter
   `robomimic_hd5.py`. This just copies keys 1:1 and renames `/`→`.`.
3. **Step 2** — `LeRobot "raw" → LeRobot "final"` via a path-specific transform
   that **builds the action/obs vectors the model actually consumes** (e.g. the
   49-D whole-body action, the 22-D no-wheel proprio, the black placeholder
   image). **This is "what's going on" — the raw HDF5 actions are re-sliced and
   re-ordered here.** DexMimicGen skips Step 2.

After that: **Step 3** train, **Step 4** visualize/sanity-check, **Step 5** serve
+ offline eval.

---

## 1. What this repo trains (RBY1)

Two RBY1 policy families (both = HPT trunk + flow-matching action head):

| Policy | Observation | Action | Where |
|---|---|---|---|
| **Single-arm image** | real `aria_image` only (ResNet), **no proprio stems** | `actions_joint_right_arm_hand` = **19-D** (R-arm 7 + R-hand 12) | model `rby1_0320_right_hand_img_only.yaml`; exp `rby_no_mobile/train_rby1_0320_03_right_hand_img_only.yaml` |
| **Whole-body AprilTag** | `april_tag` (6-D) or `april_tag_xyz` (3-D) + robot/hand proprio + **black placeholder image** (TinyCNN) | `actions_joint_base_torso_head_arm_hand` = **49-D** (whole body) | model `experiments/hierarchical/rby1_hierarchical_p1_masked_attn_april6d.yaml`; exp `experiments/hierarchical/rotatebox_p1_april6d.yaml` |

Embodiment id: **`EMBODIMENT.RBY1 = 12`** (`egomimic/rldb/utils.py:97`). All schematic /
norm-stat lookups key on the integer `12`, not the string `"rby1"`.

---

## 2. Data formats — HDF5 vs LeRobot

### 2.1 Raw / intermediate HDF5 (robomimic style)

The converter expects a robomimic-style HDF5:

```
<file>.hdf5
└── data/                      # top-level group
    ├── demo_0/                # one episode per child
    │   ├── obs/
    │   │   ├── aria_image          # (T, …) image frames (optional per config)
    │   │   ├── robot0_joint_pos    # (T, 26) float
    │   │   ├── hand_left_qpos      # (T, 12) float
    │   │   ├── hand_right_qpos     # (T, 12) float
    │   │   ├── april_tag           # (T, 6)  float  (SEW only)
    │   │   └── april_tag_xyz       # (T, 3)  float  (SEW only)
    │   └── actions/
    │       └── joint               # (T, 49) float  ← raw SEW command vector
    ├── demo_1/ ...
```

- Episodes = `list(data["data"].keys())` when `--ignore_episode_keys` is passed
  (`egomimic/rldb/scripts/robomimic_hd5.py:86`).
- A key is treated as an **image** if its name contains `img`/`image`/`rgb`
  (`robomimic_hd5.py:162,226`); compressed JPEGs are `cv2.imdecode`'d to CHW.
- **Which keys get pulled is defined entirely by a config JSON** (see §4.1).

### 2.2 LeRobot dataset (the FINAL on-disk training format)

```
<output-dir>/<name>/
├── meta/
│   ├── info.json        # features dict (dtype/shape/names), fps, robot_type, total_episodes,
│   │                    #   chunks_size, parquet/video path templates
│   ├── episodes.jsonl   # one line/episode: index, length, tasks
│   ├── tasks.jsonl      # task strings
│   └── stats.json       # per-feature mean/std/min/max
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # all per-frame columns incl. images (unless video-encoded)
│       └── episode_000001.parquet ...
└── videos/                          # ONLY if a feature has dtype "video" (--encode-as-video)
    └── chunk-000/<cam_key>/episode_000000.mp4
```

- Path templates: `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`,
  `episode_chunk = episode_index // chunks_size` (default `chunks_size=1000`)
  (`external/lerobot/lerobot/common/datasets/utils.py:41-47`).
- **By default images are stored *inside the parquet* as raw `image` frames, not
  mp4.** A `videos/` dir appears only with `--encode-as-video`. The RBY1 sbatch
  scripts do **not** pass it, so no `videos/` for SEW/egoengine/dexmimicgen.
- Key-name rule: HDF5 `obs/aria_image` → LeRobot `obs.aria_image`,
  `actions/joint` → `actions.joint` (just `/`→`.`; `robomimic_hd5.py:44-46`).
- `--robot-type rby1` sets the embodiment baked into `meta/info.json`.

### 2.3 How the LeRobot dataset is read at train time

`FolderRLDBDataset(folder_path=...)` detects a LeRobot root by `meta/info.json`
and wraps `RLDBDataset(LeRobotDataset)` (`egomimic/rldb/utils.py:184,588`):

- **Train/valid split** is derived on the fly from `valid_ratio` (default 0.2)
  via `_update_splits` — you do **not** pre-split on disk.
- **Action chunking** comes from `delta_timestamps` in the data YAML, e.g.
  `"actions.joint_base_torso_head_arm_hand": [0.0, 0.1, …, 3.1]` (32 steps @
  10 fps). The list length = the action horizon the model sees.

---

## 3. Data structure & data types — the `DataSchematic`

`DataSchematic` (`egomimic/rldb/utils.py:1167`) is the contract between the
on-disk LeRobot keys and the model. Every key has three coordinates:

```yaml
<embodiment>:            # e.g. rby1  (mapped to int id 12)
  <batch_key>:           # name the MODEL sees, e.g. front_img_1, robot0_joint_pos
    key_type: <type>     # camera_keys | proprio_keys | action_keys | metadata_keys
    lerobot_key: <col>   # exact column in the LeRobot dataset, e.g. obs.aria_image
```

- `camera_keys` → encoded by an image encoder (ResNet / TinyCNN) → image stem.
- `proprio_keys` → wrapped as `state_<batch_key>` → MLP+cross-attn stem.
  **Normalized.**
- `action_keys` → the prediction target. **Normalized.**
- `metadata_keys` → `embodiment` etc. Not normalized.

Two reusable lookups: `lerobot_key_to_keyname` and `keyname_to_lerobot_key`.
Shapes are filled at runtime by `infer_shapes_from_batch`; norm stats by
`infer_norm_from_dataset`.

> **Gotcha (the `state_` prefix):** a proprio batch key `robot0_joint_pos` is fed
> to the model as `state_robot0_joint_pos`. The matching entry in the model
> YAML's `stem_specs` must be `state_robot0_joint_pos`. A proprio key that is in
> the schematic but has **no** `state_*` stem is normalized and then **silently
> ignored** at forward time.

### 3.1 RBY1 observation keys

| LeRobot key | batch key | dim | dtype | notes |
|---|---|---|---|---|
| `obs.aria_image` | `front_img_1` | `(3,H,W)` | uint8 (real) / float16 (black) | real image (single-arm) **or** black placeholder (whole-body) |
| `obs.robot0_joint_pos` | `robot0_joint_pos` | **26** | float32 | base[0:4] torso[4:10] r_arm[10:17] l_arm[17:24] head[24:26] |
| `obs.robot0_joint_pos_no_wheel` | `robot0_joint_pos` | **22** | float32 | `robot0_joint_pos[4:]` (drops base/wheel) — used by whole-body |
| `obs.hand_left_qpos` | `hand_left_qpos` | 12 | float32 | |
| `obs.hand_right_qpos` | `hand_right_qpos` | 12 | float32 | |
| `obs.april_tag` | `april_tag` | **6** | float32 | xyz + rotation vector ("april6d") |
| `obs.april_tag_xyz` | `april_tag_xyz` | **3** | float32 | xyz only |
| `obs.right_arm` / `obs.left_arm` | — | 7 | float32 | derived from `robot0_joint_pos` |
| `obs.base_pose_integrated` | — | 3 or K·3 | float32 | dead-reckoned base pose (single or windowed) |
| `obs.eef_proprio` (egoengine) | `eef_proprio` | eef | float32 | |
| `obs.task_id` (optional) | `task_id` | one-hot | float32 | auto-included if present |

### 3.2 RBY1 action keys

`actions.joint` is the **raw 49-D SEW command**, in this order:

```
[0:7) L-arm   [7:14) R-arm   [14:20) torso   [20:22) head
[22:25) base Δ(dx,dy,dyaw)   [25:37) L-hand  [37:49) R-hand
```

Step 2 (`egoengine_lerobot_extract_arm_hand.py`) **re-slices & re-orders** this
into the keys the configs actually train on (`:228-268`):

| Action key | dim | composition | used by |
|---|---|---|---|
| **`actions.joint_base_torso_head_arm_hand`** | **49** | `[base, torso, head, L-arm, R-arm, L-hand, R-hand]` | **whole-body** policies |
| **`actions.joint_right_arm_hand`** | **19** | `[R-arm(7), R-hand(12)]` | **single-arm** policy |
| `actions.joint_left_arm_hand` | 19 | `[L-arm, L-hand]` | |
| `actions.joint_right_arm` / `_left_arm` | 7 | one arm | |
| `actions.joint_arm_hand_torso` | 38 | `[L-arm,R-arm,L-hand,R-hand,torso]` | |
| `actions.joint_hands` | 24 | `[L-hand, R-hand]` | hierarchical sub-block |
| `actions.joint_arm_head` | 16 | `[L-arm,R-arm,head]` | |
| `actions.joint_base_torso` | 9 | `[torso, base]` | |
| `actions.right_arm_eef_hand` (egoengine) | ~18 | `[eef, hand_right_cmd_qpos]` | egoengine policy |
| `actions.delta_joint_*` (optional) | = source | `target[t+1]−target[t]` | with `--add-delta-actions` |

> ⚠️ **Two different 49-D layouts.** Raw `actions.joint` ≠
> `actions.joint_base_torso_head_arm_hand`. The CLAUDE.md "Action Vector Layout"
> table describes the **output** key (base, torso, head, arms, hands), *not* the
> raw input. The reorder happens at `egoengine_lerobot_extract_arm_hand.py:245`.

**Hierarchical decoding** of the 49-D whole-body action uses
`block_dims: [3, 6, 2, 14, 24]` (base, torso, head, arms, hands) with DAG parents
`[[], [0], [0,1], [0,1], [0,1,3]]` (arms/hands skip the head dead-end).

### 3.3 Normalization (`norm_mode`)

Applied to `proprio_keys` + `action_keys` only (`normalize_data`/`unnormalize_data`,
`utils.py:1431`). Stats computed over the whole dataset per key per embodiment.

| mode | forward | notes |
|---|---|---|
| `zscore` | `(x − mean) / (std+ε)` | |
| `minmax` | `2·(x − min)/(max − min) − 1` → [−1,1] | single-arm config uses this |
| `quantile` | `2·(x − q1)/(q99 − q1) − 1` → [−1,1] | **default** (`train.yaml`); robust to outliers |

Keys with no computed stats **warn and pass through** (don't crash) — fixed
behavior for keys present in the schematic but absent from the export.

---

## 4. Preprocessing — step by step

There are three RBY1 ingestion paths plus a direct Aria path. Pick by raw input.

### 4.1 The conversion config JSONs (`egomimic/rldb/configs/`)

Step 1 is driven by a tiny JSON listing **only** `action_keys` + `obs_keys` (which
HDF5 groups to copy). No fps, no rename map, no image flags.

| Config | obs_keys | action_keys | path |
|---|---|---|---|
| `RBY1_SEW_lowdim_HDF5_config.json` | `april_tag_xyz, april_tag, hand_left_qpos, hand_right_qpos, robot0_joint_pos` (**no image**) | `joint, joint_arm, joint_arm_hand, joint_arm_head_torso_base` | SEW (default) |
| `RBY1_SEW_img_HDF5_config.json` | adds `aria_image` | same | SEW (with real image) |
| `RBY1_egoengine_HDF5_config.json` | `images, hand_right_qpos, eef_proprio` | `eef, hand_right_cmd_qpos` | EgoEngine |
| `RBY1_dexmimicgen_no_mobile_HDF5_config.json` | `frontview_image, robot0_joint_pos, robot0_left/right_gripper_qpos` | `joint, joint_arm, joint_arm_hand, joint_arm_head_torso_base, hand_left/right_cmd_input` | DexMimicGen |
| `RBY1_HDF5_config.json` | `aria_image, hand_left_qpos, hand_right_qpos, robot0_joint_pos` | `joint, …` | generic |

> Note SEW-lowdim has **no image** → the black placeholder is injected in Step 2
> (`--black-image`). That's exactly why the whole-body policy "sees" a black image.

---

### 4.2 PATH A — SEW (→ whole-body AprilTag policy)

Orchestrators: `SEW_data_workflow.sh` (manual, Steps 1–2),
`sew_batch_workflow.sh`, `sew_hierarchical_batch_workflow.sh` (SLURM, Steps 1–3),
`sew_mink_batch_workflow.sh` (adds Step 0).

**Step 0 (optional) — CSV → HDF5** (only if starting from raw body-pose CSVs):
```bash
# Runs inside the EXTERNAL SEW-Geometric-Teleop repo + its venv (MuJoCo IK).
python egomimic/scripts/sew_process/csv_to_hdf5_sew_custom.py \
  --input_folder <csv_folder> --output_folder <hdf5_out_dir> \
  --hdf5_name robot_data_sew.hdf5 \
  --mobile_base_config_path <.../sew_solver_mobile_base_user.yaml>
```
Creates the HDF5 with `obs/{april_tag,april_tag_xyz,aria_image,robot0_joint_pos,…}`
and `actions/joint` (49-D). `--no_aria_black_image` to keep a real image
(default bakes a black one).

**Step 1 — HDF5 → LeRobot raw:**
```bash
python egomimic/rldb/scripts/robomimic_hd5.py \
  --name "${DS}_raw" --raw-path /path/to/robot_data.hdf5 \
  --dataset-repo-id "${DS}_raw" \
  --config-path ./egomimic/rldb/configs/RBY1_SEW_lowdim_HDF5_config.json \
  --output-dir ./datasets/${DS}_lerobot_raw \
  --fps 10 --ignore_episode_keys --robot-type rby1
```

**Step 2 — build the final training keys** (49-D action, 22-D proprio, black img):
```bash
python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py \
  ./datasets/${DS}_lerobot_raw/ \
  --output-path ./datasets/${DS}_human_data \
  --black-image            # whole-body/SEW: synthesize black aria_image
  # --add-delta-actions    # optional: add actions.delta_joint_*
```

**Step 2b (optional) — base pose** (for mobile / push-cart base prediction):
```bash
python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_base_pose.py \
  ./datasets/${DS}_human_data --output-path ./datasets/${DS}_basepose \
  # --window-size 6 --window-delta 10   # windowed body-frame history (18-D)
```

**Step 3 — train** (see §5).

### 4.3 PATH B — EgoEngine (→ single-arm eef+hand policy)

Orchestrator: `egoengine_batch_workflow.sh` → `submit_egoengine_training.sbatch`.

```bash
# Step 1
python egomimic/rldb/scripts/robomimic_hd5.py \
  --name "${DS}_raw" --raw-path /path/to/teleop.hdf5 --dataset-repo-id "${DS}_raw" \
  --config-path ./egomimic/rldb/configs/RBY1_egoengine_HDF5_config.json \
  --output-dir ./datasets/${DS}_lerobot_raw --fps 10 --ignore_episode_keys --robot-type rby1

# Step 2 — concat eef + hand → actions.right_arm_eef_hand
python egomimic/scripts/egoengine_process/egoengine_lerobot_combine_action.py \
  ./datasets/${DS}_lerobot_raw/ --output-path ./datasets/${DS}_right_arm_hand
```

### 4.4 PATH C — DexMimicGen (no Step 2)

Orchestrator: `dexmimicgen_batch_workflow.sh`. Step 1 with
`RBY1_dexmimicgen_no_mobile_HDF5_config.json`, then **train directly off the raw
LeRobot output** — training consumes raw `actions.joint` (no extract step).

### 4.5 PATH D — Aria glasses (direct to LeRobot)

`egomimic/scripts/aria_process/aria_to_lerobot.py` converts Aria VRS + MPS output
straight to LeRobot (no robomimic HDF5). See `data_processing.md`. Not part of the
RBY1 robot pipeline above.

---

## 5. Training

```bash
source emimic/bin/activate
export TMPDIR=/tmp                       # avoid /tmp space issues

# single config
python egomimic/trainHydra.py model=hpt_bc_flow_rby1

# a full experiment (@package _global_ configs → use --config-name, NOT +experiment)
python egomimic/trainHydra.py \
  --config-name=experiments/hierarchical/rotatebox_p1_april6d \
  name=<DS> description=run1 \
  data.train_datasets.dataset1.datasets.rl2_lab.folder_path=./datasets/<DS>_human_data \
  data.valid_datasets.dataset1.datasets.eth_lab.folder_path=./datasets/<DS>_human_data

# debug (one node, fast)   |   SLURM (submitit)
python egomimic/trainHydra.py trainer=debug logger=debug
python egomimic/trainHydra.py -m launch_params.gpus_per_node=4 launch_params.nodes=1 name=<n> description=<d>

# resume
python egomimic/trainHydra.py ckpt_path=path/to/last.ckpt
```

**Flow** (`egomimic/trainHydra.py`): instantiate `DataSchematic` → instantiate
datasets (schematic injected) → `infer_shapes_from_batch(dataset[0])` +
`infer_norm_from_dataset(dataset)` → instantiate model with the populated
schematic injected as `robomimic_model.data_schematic` → `trainer.fit()`.

> **The `.ckpt` is the source of truth at inference.** `ModelWrapper` calls
> `save_hyperparameters()`, so the **entire HPT model — including the
> `DataSchematic` with all norm stats — is serialized into the checkpoint.**
> Editing `.hydra/config.yaml` after training does **nothing** at inference.

---

## 6. How to CHECK a policy

### 6.1 Inspect a LeRobot dataset (before training)

```bash
# list every feature key + shape
python egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py \
  /path/to/<DS>/LeRobot --list-keys

# plot one action key over one episode (writes PNGs to <dataset>/viz_preview/)
python egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py \
  /path/to/<DS>/LeRobot -k actions.joint_base_torso_head_arm_hand --dims 0:14 -e 0
```
- `-k/--action-key`, `--dims` (`all` | `0:14` | `0,1,5`), `-e/--episode`,
  `--list-keys` (note: **no `-l` short flag**), `--time-axis {seconds,frame}`,
  `--no-images`.
- **Output is matplotlib PNG files, not an interactive rerun viewer** (the
  "rerun" phrasing in `rby1.md`/`CLAUDE.md` does not match this script).

### 6.2 Inspect what's baked into a `.ckpt` (no dedicated script — use Python)

```python
from egomimic.pl_utils.pl_model import ModelWrapper
m   = ModelWrapper.load_from_checkpoint("path/to/last.ckpt", weights_only=False)
hpt = m.model
sch = hpt.data_schematic

print(hpt.domains)                          # ['rby1']
print(hpt.ac_keys, hpt.camera_keys, hpt.proprio_keys)
print(sch.norm_mode)                        # zscore | minmax | quantile
print(sch.norm_stats[12].keys())            # per-key stats for RBY1 (id 12)
print(sch.norm_stats[12]['actions_joint_base_torso_head_arm_hand']['mean'])
```
`weights_only=False` is **required** (the schematic is a pickled object).
`EgoVersePolicy(m).metadata` is a convenient one-shot summary.

### 6.3 Serve a checkpoint

```bash
python egomimic/scripts/serve_policy.py --checkpoint path/to/last.ckpt --port 8000
# --host (default 0.0.0.0)
```
- Loads the ckpt, wraps in `EgoVersePolicy`, serves over **WebSocket + msgpack**
  (openpi-compatible). `GET /healthz` → 200.
- ⚠️ For flow/diffusion heads it **forces `num_inference_steps = 10`** (overrides
  the trained `50`) for speed — relevant if comparing to training-time validation.
- `EgoVersePolicy` is **single-embodiment only**.

**Wire protocol:** on connect the server sends a metadata dict
(`{embodiment, action_horizon, action_dim, camera_keys, proprio_keys, methods}`).
Then send an observation dict (or a list → batched), receive
`{actions: (B,T,D), embodiment, server_timing}`.

**Client must send** (RBY1, keys = the baked `camera_keys`/`proprio_keys`):
- `front_img_1`: `(H,W,3)` **uint8, BGR** (converted to RGB internally).
- proprio keys, e.g. `robot0_joint_pos`: `(D,)` float32.

**Returns:** `actions` shaped `(1, action_horizon, action_dim)` — e.g. `(1,10,49)`
for the whole-body policy — **already un-normalized** (`forward_eval` calls
`unnormalize_data`). Layout = the `actions.joint_base_torso_head_arm_hand` table
in §3.2.

### 6.4 Offline eval — compare predictions vs ground truth

```bash
# Against a running server:
python egomimic/scripts/test_serve_policy_client.py \
  --episode-idx 0 --max-steps 30 --dataset-folder ~/path/to/<DS> --trajectory

# Or fully local (no server): load the ckpt directly
python egomimic/scripts/test_serve_policy_client.py \
  --local --checkpoint path/to/last.ckpt --dataset-folder ~/path/to/<DS> --trajectory --episode-idx 0
```
- `--trajectory` rolls an episode frame-by-frame; otherwise random samples.
- Writes per-dim GT-vs-pred plots + MSE/MAE/RMSE to
  `logs/test_serve_policy_<timestamp>/`.
- ⚠️ **Edit the hardcoded key maps** near the top of the script
  (`RBY1_LEROBOT_TO_OBS`, `RBY1_LEROBOT_ACTION_KEYS`) to match your dataset's keys,
  or frames get skipped. This is a known rough edge.
- The local loader **also** forces `num_inference_steps = 10`.

### 6.5 On-robot eval

`egomimic/scripts/evaluation/eval.py` is an abstract base; `eval_eve.py` is a
real-robot rollout for the **EVE/Aloha** rig (not RBY1). There is no in-tree RBY1
real-robot rollout yet — RBY1 eval goes through the serve + client path above
(`rby1.md` notes "test with rollout_sim in SEW_teleop" as TODO).

---

## 7. Gotchas (bite-sized)

- **HDF5 ≠ training data.** LeRobot (parquet+meta) is final. Always run Step 1
  (+ Step 2 for SEW/egoengine).
- **Two 49-D layouts.** Raw `actions.joint` (L-arm,R-arm,torso,head,base,hands)
  vs final `actions.joint_base_torso_head_arm_hand` (base,torso,head,arms,hands).
- **`.ckpt` is source of truth** — schematic + norm stats are baked in; editing
  YAML post-train does nothing.
- **`state_` prefix** — proprio stems in the model YAML must be
  `state_<batch_key>`; otherwise the key is normalized then ignored.
- **Schematic vs stems** — a key only affects the forward pass if it has a stem in
  `stem_specs`; otherwise it's silently dropped.
- **Black placeholder image** — SEW-lowdim has no real image; `--black-image`
  injects a `(3,64,64)` zero frame. The whole-body policy relies on AprilTag +
  proprio, not vision.
- **`num_inference_steps` is overridden to 10** at serving / local test (trained
  value is 50).
- **Embodiment is an int** — lookups key on `12`, not `"rby1"`.
- **Visualize = PNG, not rerun.**
- **`logs/` is gitignored** — no checkpoints live in the repo tree.

---

## 8. File index

| Purpose | Path |
|---|---|
| Train entry | `egomimic/trainHydra.py` |
| Main config | `egomimic/hydra_configs/train.yaml` |
| Schematic / datasets | `egomimic/rldb/utils.py` (`DataSchematic` :1167, `RLDBDataset` :184, `EMBODIMENT` :84) |
| HDF5→LeRobot converter | `egomimic/rldb/scripts/robomimic_hd5.py` |
| Conversion configs | `egomimic/rldb/configs/RBY1_*_HDF5_config.json` |
| SEW Step 2 (arm/hand, 49-D) | `egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py` |
| EgoEngine Step 2 (combine) | `egomimic/scripts/egoengine_process/egoengine_lerobot_combine_action.py` |
| Base-pose Step 2b | `egomimic/scripts/egoengine_process/egoengine_lerobot_extract_base_pose.py` |
| CSV→HDF5 (SEW Step 0) | `egomimic/scripts/sew_process/csv_to_hdf5_sew_custom.py` |
| Dataset visualizer | `egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py` |
| Serve | `egomimic/scripts/serve_policy.py` · `egomimic/serving/egoverse_policy.py` · `egomimic/serving/API.md` |
| Offline eval client | `egomimic/scripts/test_serve_policy_client.py` |
| HPT model / forward_eval | `egomimic/algo/hpt.py` |
| Lightning wrapper | `egomimic/pl_utils/pl_model.py` |
| Orchestrators | `*_batch_workflow.sh`, `SEW_data_workflow.sh`, `submit_*.sbatch` (repo root) |
