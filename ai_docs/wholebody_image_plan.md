# Plan: Whole-Body (49-D) + REAL Image RBY1 Policy

> **New training pattern**: predict the 49-D whole-body action
> (`actions.joint_base_torso_head_arm_hand`) from a **real aria image** (ResNet)
> **+ proprio** (robot joints 22-D no-wheel, hand qpos), **no april_tag**.
> This is the existing whole-body policy with the black-placeholder image swapped
> for real vision (the april-tag crutch dropped).
>
> Status: configs + launcher + venv are prepared **now**; when the HDF5 arrives,
> training is one command. See §3 for the staged, rollback-friendly run plan.

---

## 1. What was created (all in your checkout)

| Artifact | Path | Purpose |
|---|---|---|
| Model config (vanilla) | `egomimic/hydra_configs/model/experiments/wholebody_image/rby1_wb_img_proprio_act32_vanilla.yaml` | ResNet image + proprio stems + flat 49-D flow head |
| Model config (hier) | `.../rby1_wb_img_proprio_act32_hier.yaml` | same obs, hierarchical 49-D head `[3,6,2,14,24]` |
| Data config | `egomimic/hydra_configs/data/experiments/wholebody_image/rby1_wb_img_act32.yaml` | LeRobot loader, 32-step action chunks (folder overridden at launch) |
| Experiment (vanilla) | `egomimic/hydra_configs/experiments/wholebody_image/wb_img_proprio_vanilla.yaml` | composes model+data+**schematic (img+proprio, no tag)**, wandb `entity: null` |
| Experiment (hier) | `.../wb_img_proprio_hier.yaml` | hierarchical variant |
| Launcher (sbatch) | `submit_wb_img_training.sbatch` | Step 1+2 (data) + Step 3 (train) on a GPU, **your** checkout/venv |
| Dispatcher | `wb_img_batch_workflow.sh` | submit vanilla + hier for one dataset |

**Design choices baked in** (per your answers): obs = image + proprio, **no april_tag**;
**both** vanilla and hierarchical heads; **your own venv**; wandb → **your personal
account** (`entity: null`).

### Differences from the existing whole-body (april-tag) pipeline
| | existing (april6d) | this (wb image) |
|---|---|---|
| Step 1 config | `RBY1_SEW_lowdim` (no image) | **`RBY1_SEW_img`** (keeps `obs.aria_image`) |
| Step 2 flag | `--black-image` | **omitted** (real image preserved) |
| image encoder | `TinyCNN` (64×64 black) | **`ResNet`** (real frame) |
| obs stems | april_tag + proprio | **proprio only** (no april_tag) |
| trunk | d=192, 8 blocks | d=256, 16 blocks |

---

## 2. Prerequisites (doing now, before data)

1. **Venv** at `emimic/` (uv, py3.11) — building in the background now; verify with
   §3 Phase 0. The sbatch does `source emimic/bin/activate`.
2. **Incoming HDF5 must contain a real `obs/aria_image`** (robot camera frame) under
   `data/<demo>/obs/aria_image`, plus the usual `actions/joint` (49-D),
   `obs/robot0_joint_pos` (26-D), `obs/hand_left_qpos`, `obs/hand_right_qpos`.
   The sbatch **fails fast** if `obs/aria_image` is missing.
   - If your data is human SEW demos with **no** real camera, this pattern does not
     apply (there is no real image) — go back and reconsider obs (use april_tag).
3. **GPU** is via SLURM (`overcap`, `a40:1`, account `rl2-lab`) — already wired into
   the sbatch. No extra setup.
4. **WandB**: your `~/.netrc` key is set. Configs use `entity: null` → your personal
   account, `project: sew_policy`. (Rename the project per experiment if you want.)

---

## 3. Staged run plan (with verify + rollback)

> Philosophy you asked for: **do a step, verify it; if good continue; if not, roll
> back one or two steps and retry.** Each phase below has a **PASS** condition and a
> **FAIL → go back to** pointer. The one-shot command (Phase 5) chains Steps 1–3; use
> the manual phases when you want to inspect between steps.

Run everything from `/coc/flash7/czhang883/Documents/EgoVerse` with the venv active:
```bash
cd /coc/flash7/czhang883/Documents/EgoVerse
source emimic/bin/activate
export TMPDIR=/tmp
```

### Phase 0 — Environment sanity (no data needed)
```bash
# 0a. venv imports
python -c "import egomimic, torch, lightning, datasets; print('OK', torch.__version__)"
# 0b. configs compose (prints merged config; validates YAML wiring + overrides)
python egomimic/trainHydra.py --config-name=experiments/wholebody_image/wb_img_proprio_vanilla --cfg job | head -60
python egomimic/trainHydra.py --config-name=experiments/wholebody_image/wb_img_proprio_hier   --cfg job | head -60
```
- **PASS**: 0a prints a torch version; 0b prints a config tree with
  `ac_keys.rby1: actions_joint_base_torso_head_arm_hand`, `front_img_1` →
  `obs.aria_image`, ResNet encoder, and **no** `april_tag` key.
- **FAIL** → fix the venv (re-run §Appendix A) or the config typo, then redo Phase 0.

### Phase 1 — Convert HDF5 → LeRobot raw (Step 1)
```bash
DS=RBY1_0623_wb_img                 # pick a dataset name
RAW=/path/to/robot_data.hdf5        # the file you give me
python egomimic/rldb/scripts/robomimic_hd5.py \
  --name "${DS}_raw" --raw-path "${RAW}" --dataset-repo-id "${DS}_raw" \
  --config-path ./egomimic/rldb/configs/RBY1_SEW_img_HDF5_config.json \
  --output-dir ./datasets/${DS}_lerobot_raw --fps 10 --robot-type rby1 --ignore_episode_keys
```
- **Verify**:
  ```bash
  python -c "import json; i=json.load(open('datasets/${DS}_lerobot_raw/${DS}_raw/meta/info.json')); \
print('eps', i['total_episodes']); print('has_img', 'obs.aria_image' in i['features']); \
print('img_shape', i['features'].get('obs.aria_image',{}).get('shape'))"
  ```
- **PASS**: `total_episodes` == #demos in the HDF5; `has_img True`; `img_shape` is the
  real resolution (e.g. `[3,H,W]` with H,W ≫ 64), **not** `[3,64,64]`.
- **FAIL** → if `has_img False`: your HDF5 lacks a real image (Prereq #2) — stop, get
  proper data. If counts mismatch: `rm -rf datasets/${DS}_lerobot_raw` and rerun Phase 1.

### Phase 2 — Build training keys (Step 2, **no** --black-image)
```bash
python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py \
  ./datasets/${DS}_lerobot_raw/ --output-path ./datasets/${DS}
```
- **Verify**:
  ```bash
  python -c "import json; i=json.load(open('datasets/${DS}/meta/info.json')); f=i['features']; \
print('eps', i['total_episodes']); \
print('action49', f['actions.joint_base_torso_head_arm_hand']['shape']); \
print('proprio22', f['obs.robot0_joint_pos_no_wheel']['shape']); \
print('img_shape', f['obs.aria_image']['shape'])"
  ```
- **PASS**: `action49 == [49]`, `proprio22 == [22]`, `img_shape` still the **real**
  resolution (not `[3,64,64]`), episode count unchanged.
- **FAIL** → `rm -rf datasets/${DS}` and rerun Phase 2. If `img_shape` came out
  `[3,64,64]`, you accidentally passed `--black-image` — rerun without it. If keys are
  missing, go back to **Phase 1** (wrong Step-1 config).

### Phase 3 — Eyeball the data (Step 4 in your numbering)
```bash
python egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py \
  ./datasets/${DS} --list-keys
python egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py \
  ./datasets/${DS} -k actions.joint_base_torso_head_arm_hand --dims 0:14 -e 0
# images grid (confirm frames are REAL, not black):
python egomimic/scripts/egoengine_process/visualize_lerobot_dataset.py ./datasets/${DS} -e 0
```
- **PASS**: `preview_images.png` shows a real scene (not black); action curves are
  smooth and in a sane range.
- **FAIL** (black images / garbage actions) → go back to **Phase 1–2**; if the raw
  data itself is bad, stop and re-collect.

### Phase 4 — Smoke-test training (1 short run, catches shape/norm bugs)
```bash
python egomimic/trainHydra.py \
  --config-name=experiments/wholebody_image/wb_img_proprio_vanilla \
  trainer=debug logger=debug \
  name=${DS} description=smoketest \
  data.train_datasets.dataset1.datasets.rl2_lab.folder_path=$(pwd)/datasets/${DS} \
  data.valid_datasets.dataset1.datasets.eth_lab.folder_path=$(pwd)/datasets/${DS}
```
- **PASS**: it gets past "infer norm" + builds the model + runs ≥1 train step with a
  finite loss, no shape errors. (Confirms image stem(256), proprio stems(22/12/12),
  49-D head all line up.)
- **FAIL** → read the traceback:
  - shape mismatch on a stem → fix `input_dim` in the model config, redo Phase 4.
  - missing key / norm warning for an action/proprio key → fix the **schematic** in the
    experiment config (`lerobot_key`), redo Phase 4.
  - data not found → fix the `folder_path`, redo Phase 4.

### Phase 5 — Full training on GPU (Step 5)
**One-shot (recommended; does Steps 1+2+3 for both heads):**
```bash
./wb_img_batch_workflow.sh ${DS} ${RAW} both     # submits vanilla + hier
squeue -u czhang883
```
**Or a single variant / from already-converted data:**
```bash
sbatch --job-name=wbimg_${DS}_vanilla \
  --export=ALL,DATASET_NAME=${DS},RAW_DATA_PATH=${RAW},TRAIN_CONFIG=experiments/wholebody_image/wb_img_proprio_vanilla,DESCRIPTION=vanilla \
  submit_wb_img_training.sbatch
```
- **Verify**: `squeue -u czhang883` shows the job(s) running; `logs/slurm/<jobid>/out.log`
  shows Steps 1–3 progressing; wandb run appears under **your** account / `sew_policy`,
  loss decreasing.
- **FAIL** → `scancel <jobid>`; inspect `logs/slurm/<jobid>/err.log`; fix; resubmit.
  Conversion is cached, so resubmits skip Steps 1–2 (episode-count guarded).

### Phase 6 — Check the trained policy
Follow `ai_docs/data_pipeline_and_policy_guide.md` §6: inspect the `.ckpt` (norm stats,
keys), serve it, and run the offline eval client against `./datasets/${DS}`.
```bash
python egomimic/scripts/serve_policy.py --checkpoint logs/${DS}/vanilla/checkpoints/last.ckpt --port 8000
python egomimic/scripts/test_serve_policy_client.py --local \
  --checkpoint logs/${DS}/vanilla/checkpoints/last.ckpt \
  --dataset-folder ./datasets/${DS} --trajectory --episode-idx 0
```
> Note: `test_serve_policy_client.py` has hardcoded RBY1 key maps near the top
> (`RBY1_LEROBOT_TO_OBS`, `RBY1_LEROBOT_ACTION_KEYS`) — point the action key at
> `actions.joint_base_torso_head_arm_hand` for this policy before running.

---

## 4. Tuning knobs (once it trains)
- **Batch / workers**: `data.train_dataloader_params.dataset1.batch_size=64` (ResNet on
  real images is heavier than the black-TinyCNN path; drop to 16 if OOM on a40).
- **LR**: `model.optimizer.lr=4e-4` for larger batches.
- **Action horizon**: currently 32 (data chunks + head `act_seq` + trunk `action_horizon`
  must all match — change all three if you alter it).
- **Proprio dropout**: stems use 0.4 (joints) / 0.7 (hands) to push reliance on vision;
  lower if proprio should matter more.

---

## Appendix A — Rebuild the venv (if Phase 0 fails)
> **Critical:** the NFS home (`/nethome/czhang883`) has a tight quota — the default uv
> cache there blows up on `torch` (`Disk quota exceeded`). Keep ALL caches on flash.
```bash
cd /coc/flash7/czhang883/Documents/EgoVerse
export PATH=/coc/flash7/czhang883/.local/bin:$PATH       # uv lives here
export UV_CACHE_DIR=/coc/flash7/czhang883/.cache/uv      # <-- cache on FLASH, not home
export TMPDIR=/coc/flash7/czhang883/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
rm -rf emimic ~/.cache/uv                                # drop broken venv + home cache
uv venv emimic --python 3.11
source emimic/bin/activate
uv pip install -r requirements.txt
uv pip install -e external/lerobot
uv pip install -e .
python -c "import egomimic, torch; print('OK', torch.__version__)"
```
`mujoco-py` is the most fragile dep but is **not needed for training** (sim/IK only); if
it fails, the rest of the env can still train. The training sbatch sets
`XDG_CACHE_HOME`/`HF_HOME`/`UV_CACHE_DIR` to flash for the same quota reason.

## Appendix B — Open assumptions to confirm when data lands
1. HDF5 has a real `obs/aria_image` (else this pattern is moot).
2. `actions/joint` is the 49-D SEW layout (L-arm,R-arm,torso,head,base,L-hand,R-hand);
   `obs/robot0_joint_pos` is 26-D. (Phase 2 verify catches mismatches.)
3. fps = 10 (matches the 32-step / 3.2 s action chunk). Change `--fps` + `delta_timestamps`
   together if different.
