# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Environment

The project uses a **UV venv** at `emimic/` (not conda). Always activate before running anything:

```bash
source emimic/bin/activate
```

The Python interpreter is at `emimic/bin/python`. Use it directly for one-off commands: `emimic/bin/python egomimic/...`.

---

## Common Commands

### Training

```bash
# Standard training (override model config on CLI)
python egomimic/trainHydra.py model=hpt_bc_flow_rby1

# Debug mode (fast, on current node)
python egomimic/trainHydra.py trainer=debug logger=debug

# Multi-GPU / SLURM via submitit
python egomimic/trainHydra.py -m launch_params.gpus_per_node=4 launch_params.nodes=1 name=<name> description=<desc>

# Resume from checkpoint
python egomimic/trainHydra.py ckpt_path=path/to/last.ckpt

# Set TMPDIR before training to avoid /tmp space issues
export TMPDIR=/tmp
```

### Dataset processing (RBY1)

```bash
# Single HDF5
python egomimic/rldb/scripts/robomimic_hd5.py \
  --name <name> --dataset-repo-id <repo_id> \
  --config-path ./egomimic/rldb/configs/RBY1_HDF5_config.json \
  --output-dir ./datasets --fps 10 --ignore_episode_keys \
  --robot-type rby1 --raw-path /path/to/file.hdf5

# Folder of HDF5s
python egomimic/rldb/scripts/robomimic_hd5.py \
  --name <name> --raw-path /path/to/folder/ \
  --dataset-repo-id <repo_id> \
  --config-path ./egomimic/rldb/configs/RBY1_HDF5_config_0309.json \
  --output-dir ./datasets --fps 10 --ignore_episode_keys --robot-type rby1

# Visualize a LeRobot dataset (with rerun)
python egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py \
  /path/to/dataset/LeRobot -k actions.joint_arm --dims 0:14 -e 0
```

### Serving / Inference

```bash
# Serve a trained checkpoint
python egomimic/scripts/serve_policy.py \
  --checkpoint path/to/checkpoints/last.ckpt \
  --port 8000

# Test inference against a dataset
python egomimic/scripts/test_serve_policy_client.py \
  --episode-idx 0 --max-steps 30 \
  --dataset-folder ~/path/to/dataset --trajectory
```

---

## Architecture Overview

### Training pipeline

`egomimic/trainHydra.py` is the sole entry point. The flow is:

1. **Hydra** composes config from `egomimic/hydra_configs/train.yaml` + CLI overrides.
2. `DataSchematic` is instantiated from `cfg.data_schematic` — it registers all dataset keys, their types, and the lerobot↔batch key mapping.
3. Datasets are instantiated; `data_schematic.infer_shapes_from_batch()` and `infer_norm_from_dataset()` populate shape and normalization stats from the actual data.
4. The model (`ModelWrapper` wrapping an algo like `HPT`) is instantiated with the now-populated `DataSchematic` object injected as `robomimic_model.data_schematic`.
5. PyTorch Lightning `Trainer.fit()` runs the training loop.

**Critical:** The `DataSchematic` object (including norm stats) is baked into the saved `.ckpt` via `save_hyperparameters()`. Editing `.hydra/config.yaml` in a checkpoint directory has **no effect** at inference time.

### DataSchematic (`egomimic/rldb/utils.py:DataSchematic`)

Central schema object that:
- Maps each dataset key (lerobot format, e.g. `obs.aria_image`) to a **batch key** (e.g. `front_img_1`) with a **key type** (`camera_keys`, `proprio_keys`, `action_keys`, `metadata_keys`).
- Holds per-embodiment normalization stats (zscore/minmax/quantile) for `proprio_keys` and `action_keys`.
- Is embodiment-aware — one schematic can hold entries for multiple embodiments simultaneously.

Key methods:
- `normalize_data` / `unnormalize_data`: apply/invert normalization. Keys with no computed stats warn and pass through (don't crash).
- `keys_of_type(key_type)`: returns all registered batch keys of a given type.
- `infer_norm_from_dataset(dataset)`: computes stats by iterating the full dataset.

### HPT model (`egomimic/algo/hpt.py:HPT`)

At init, HPT reads the DataSchematic to populate:
- `self.camera_keys[embodiment_id]`
- `self.proprio_keys[embodiment_id]`
- `self.action_keys` / `self.ac_keys`

At forward time, `_robomimic_to_hpt_data()` prepares the batch:
- Camera keys → encoded by `ResNet` encoders → fed to image stems.
- Proprio keys → wrapped as `state_{key_name}` → fed to MLP+cross-attn stems.

**Stems** are defined in `stem_specs` in the model YAML. Only keys listed in `stem_specs` are consumed by the model — proprio keys in the schematic but not in `stem_specs` are silently ignored at forward time.

### Config system (Hydra)

- Config root: `egomimic/hydra_configs/`, main file: `train.yaml`.
- Defaults merge sub-configs: `model: hpt_bc_flow_rby1` loads `model/hpt_bc_flow_rby1.yaml` under `cfg.model`.
- All classes are wired via `_target_: fully.qualified.ClassName` — no Python-side ConfigStore registration needed.
- Custom resolver: `${eval: 'expr'}` (registered in `trainHydra.py`).

### Inference server

`egomimic/scripts/serve_policy.py` → `EgoVersePolicy` (`egomimic/serving/egoverse_policy.py`) → `WebsocketPolicyServer`.

The server loads a `.ckpt`, wraps it in `EgoVersePolicy`, and serves via WebSocket + msgpack (openpi-compatible protocol). `EgoVersePolicy._obs_to_batch()` iterates `self._proprio_keys` and `self._cam_keys` (read from the baked-in schematic), normalizes, and calls `forward_eval`.

---

## Adding a New Embodiment

1. Add to `EMBODIMENT` enum in `egomimic/rldb/utils.py`.
2. Add the embodiment block to `data_schematic.schematic_dict` in `train.yaml`.
3. Add `domains`, `stem_specs`, `head_specs`, and `encoder_specs` entries in the model YAML.
4. For proprio stems, prefix the stem key with `state_` (e.g. `state_robot0_joint_pos`); image stems use the bare key name.

See `model.md` for a detailed walkthrough.

---

## Key Gotchas

- **Schematic vs. stems**: A key registered as `proprio_keys` in the schematic will go through normalization. If it also lacks a corresponding `state_{key}` entry in `stem_specs`, it is normalized but then silently ignored during the forward pass. This was the source of the `task_id` normalization crash — schematic entry present, no stats computed (key absent from dataset), old code raised `ValueError`. Fixed to warn and pass through.
- **`.ckpt` is the source of truth at inference**: The `DataSchematic` (with all norm stats) is serialized into the checkpoint. Editing `.hydra/config.yaml` post-training does nothing.
- **`state_` prefix**: Proprio keys are stored in the batch as `state_{key_name}` (added in `_robomimic_to_hpt_data`). Stem keys in `stem_specs` must match this pattern for low-dim proprio inputs.
- **Embodiment integer IDs**: `EMBODIMENT.RBY1 = 12`. All schematic and norm-stats lookups key on the integer ID, not the string name.
