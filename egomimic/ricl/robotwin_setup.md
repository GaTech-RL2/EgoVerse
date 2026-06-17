# RoboTwin PI-RICL — cluster runbook

RoboTwin docs: https://robotwin-platform.github.io/doc/usage/robotwin-install.html

**Fresh cluster from scratch?** See `robotwin_new_cluster_setup.md` (full install guide).
This file is the per-run cheatsheet + current status.

**Status: training + closed-loop eval both verified end-to-end.** Commands below target
the Georgia Tech PACE cluster (repo at `/storage/project/r-dxu345-0/rco3/EgoVerse`).
A 250-step checkpoint scored 1/10 on `beat_block_hammer`
(undertrained — train longer for a real number; the pipeline is the deliverable).

Setup once per cluster: `git submodule update --init external/RoboTwin`; vendor the
PaliGemma tokenizer to `pg_tokenizer/` (gitignored) via
`AutoTokenizer.from_pretrained("google/paligemma-3b-mix-224").save_pretrained(...)`
(check the HF cache first). pi0.5 base ckpt: `egomimic/algo/pi_checkpoints/pi05_base_pytorch`.
**Install packages with `uv pip install` — the venv has no `pip`.**

## TODO 1 — data → zarr → train  (DONE)

```bash
source emimic/bin/activate
python -m egomimic.ricl.scripts.download_robotwin --mode hf --out egomimic/ricl/outputs/robotwin_raw
python -m egomimic.ricl.scripts.robotwin_to_zarr \
  --root egomimic/ricl/outputs/robotwin_raw/extracted --out egomimic/ricl/outputs/robotwin_zarr --limit 5  # optional validation
python egomimic/ricl/scripts/train_robotwin_ricl.py --stage cpu \
  --root egomimic/ricl/outputs/robotwin_raw/extracted --embed fake          # CPU smoke
sbatch egomimic/ricl/scripts/train_robotwin_ricl.sbatch                      # GPU training (gpu-h200, qos inferno)
```
The trainer reads the HDF5 corpus directly (`--root` = parent of the nested
`<task>/data`). Writes checkpoints + `quantiles.json` under
`outputs/robotwin_train/robotwin_ricl/version_*/`. The h200's ~141 GB fits full fp32
AdamW of the 3.6 B model (~57 GB), so no 8-bit optimizer is needed.

## TODO 2 — closed-loop eval on RoboTwin  (DONE)

1. Build the demo-bank DINOv2 index (GPU):
   ```bash
   sbatch -p gpu-h200 -A gts-dxu345-rl2 -q inferno --gres=gpu:h200:1 -c8 --mem=64G --wrap=\
     "emimic/bin/python egomimic/ricl/scripts/build_robotwin_bank_index.py \
        --root egomimic/ricl/outputs/robotwin_raw/extracted \
        --out egomimic/ricl/outputs/robotwin_bank_index --embed dinov2"
   ```
2. Sim deps — install **selectively** into `emimic` (NOT RoboTwin's `_install.sh`, which
   pins `torch==2.4.1`): `uv pip install sapien==3.0.0b1 mplib==0.2.1 open3d trimesh
   "pyglet<2"`. curobo/pytorch3d are NOT needed (see patches). Assets (~16GB):
   `emimic/bin/python external/RoboTwin/assets/_download.py` then unzip
   `{background_texture,embodiments,objects}.zip` into `assets/` and run
   `external/RoboTwin/script/update_embodiment_config_path.py`.
3. **Patches to the `external/RoboTwin` fork** (policy eval needs no motion planning, so
   we skip curobo entirely):
   - `envs/robot/planner.py`: in the `except` (curobo import failed), define a stub
     `CuroboPlanner` (constructible; real `plan_grippers` interpolation; other methods
     no-op) so the env imports + the robot builds a planner.
   - `envs/robot/robot.py` `set_planner`: force `self.communication_flag = False` (use
     the in-process stub, not curobo worker subprocesses).
   - `script/eval_policy.py`: `expert_check = False` (skip the curobo expert solvability
     run); guard the `episode_info`/`generate_episode_descriptions` block and fall back
     to the task name as the instruction; honor `EVAL_TEST_NUM` to bound episodes.
   - `task_config/demo_clean.yml`: `eval_video_log: false` (skip per-episode ffmpeg).
   - `policy/pi_ricl_egoverse/deploy_policy.yml`: set absolute `egoverse_checkpoint`,
     `bank_root`, `bank_index_dir`, `quantiles_path`.
4. Run:
   ```bash
   CKPT=<abs .ckpt> EVAL_TEST_NUM=10 sbatch egomimic/ricl/scripts/eval_robotwin_ricl.sbatch
   ```
   SAPIEN renders headless ("Render Well"); eval runs eager (`TORCH_COMPILE_DISABLE=1`,
   in the sbatch — pi0.5 `sample_actions` otherwise triggers a multi-minute compile).
   Prints `Success rate: k/n` per episode.
