# RoboTwin pi0.5 RICL — new-cluster setup

End-to-end guide to stand up **training + closed-loop eval** of EgoVerse's pi0.5 RICL
on RoboTwin 2.0 on a fresh SLURM/GPU cluster. Verified on the Georgia Tech PACE cluster
(`/storage/project/r-dxu345-0/...`). For the per-run command cheatsheet see `robotwin_setup.md`; for code
layout see `CLAUDE.md`.

Throughout, let `REPO=/path/to/EgoVerse` and run from `$REPO`.

---

## 0. Assumptions

- Linux + SLURM, NVIDIA GPUs, CUDA 12.x driver (see gotcha G1 — some nodes lag).
- Python 3.11, the project venv managed by **`uv`** (no `pip` inside it — see G2).
- AWS/R2 creds only if you also pull the SQL episode registry; **not needed** for the
  RoboTwin path (it's local HDF5/zarr).
- GPU memory: the h200s here (~141 GB) run full fp32 AdamW of the 3.6 B model (~57 GB)
  comfortably (step 6).

---

## 1. Repo + submodules

```bash
git clone <egoverse remote> $REPO && cd $REPO
git checkout ryanco/robotwin-sim
git submodule update --init external/RoboTwin external/openpi
```
`external/RoboTwin` is the `GaTech-RL2/RoboTwin` fork and carries the
`policy/pi_ricl_egoverse/` deploy shim. (Ignore any stray `external/robocasa`/
`external/robosuite` gitlinks — not used.)

---

## 2. Project venv (egomimic)

The venv is `uv`-managed and installs the package editable as `egomimic`.

```bash
cd $REPO
uv venv emimic --python 3.11
source emimic/bin/activate
uv pip install -e .            # or `uv sync` if the repo ships a uv.lock workflow
python -c "import torch; print(torch.__version__)"   # expect 2.7.x +cuXXX
```
**Always install with `uv pip install`, never `pip`** (G2).

---

## 3. PaliGemma tokenizer (offline-safe)

pi0.5 needs the PaliGemma tokenizer. Vendor it locally so compute nodes don't hit the
gated HF download:

```bash
source emimic/bin/activate
python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("google/paligemma-3b-mix-224").save_pretrained("egomimic/ricl/pg_tokenizer")
PY
```
`pg_tokenizer/` is gitignored. If the model isn't already in the HF cache you'll need an
HF token (`huggingface-cli login`) once.

---

## 4. pi0.5 base checkpoint

Training fine-tunes from the PyTorch pi0.5 base at:
```
egomimic/algo/pi_checkpoints/pi05_base_pytorch/{config.json, model.safetensors}   (~6.8 GB)
```
Provision it by copying from an existing cluster, or convert the JAX/openpi base to
PyTorch (see the JAX→PyTorch conversion command in the repo history /
`egomimic/algo/pi_checkpoints/`). The trainer's `--ckpt` defaults to this repo-relative
path.

---

## 5. RoboTwin data → zarr (TODO 1)

```bash
source emimic/bin/activate
# ~230 MB: beat_block_hammer / aloha-agilex / clean_50 from HF TianxingChen/RoboTwin2.0
python -m egomimic.ricl.scripts.download_robotwin --mode hf --out egomimic/ricl/outputs/robotwin_raw
# optional: validate the EgoVerse-zarr converter on a few episodes
python -m egomimic.ricl.scripts.robotwin_to_zarr \
  --root egomimic/ricl/outputs/robotwin_raw/extracted --out egomimic/ricl/outputs/robotwin_zarr --limit 5
# CPU smoke (no GPU): validates collate + RICL prompt + token budget
python egomimic/ricl/scripts/train_robotwin_ricl.py --stage cpu \
  --root egomimic/ricl/outputs/robotwin_raw/extracted --embed fake
```
The data lands one level deep (`.../extracted/<task>__<emb>_<setting>/<...>/data/`); the
trainer's recursive `**/data` glob finds it, so always pass `--root .../extracted`.

---

## 6. Train (TODO 1)

The h200's ~141 GB fits full fp32 AdamW of the 3.6 B model (~57 GB), so no 8-bit
optimizer is needed. (On a ≤ 48 GB GPU, add `--adam8bit` to the sbatch — it needs
`uv pip install bitsandbytes`.)

Edit `egomimic/ricl/scripts/train_robotwin_ricl.sbatch` for your cluster
(`--partition`/`--account`/`--gres`, the `cd $REPO`, and `source emimic/bin/activate`);
it ships preset for PACE (`gpu-h200` / `gts-dxu345-rl2` / `qos inferno`). Then:
```bash
sbatch egomimic/ricl/scripts/train_robotwin_ricl.sbatch
# knobs via env: CKPT_EVERY, VAL_EVERY, MAX_STEPS, BATCH_SIZE, EMBED, ROOT
```
The sbatch sets `--batch-size 2` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `--embed dinov2` builds the
retrieval cache via DINOv2 (`torch.hub`; pre-cache if compute nodes are offline).
Outputs: `egomimic/ricl/outputs/robotwin_train/robotwin_ricl/version_*/`
(`checkpoints/*.ckpt` + `quantiles.json`). Expect loss to fall from ~1.5 toward <0.1.

---

## 7. Closed-loop eval (TODO 2)

### 7a. Demo-bank DINOv2 index
```bash
sbatch -p gpu-h200 -A gts-dxu345-rl2 -q inferno --gres=gpu:h200:1 -c8 --mem=64G --wrap=\
 "emimic/bin/python egomimic/ricl/scripts/build_robotwin_bank_index.py \
    --root egomimic/ricl/outputs/robotwin_raw/extracted \
    --out egomimic/ricl/outputs/robotwin_bank_index --embed dinov2"
```
(Use the venv python directly — `--wrap` runs under `dash`, where `source` fails, G4.)

### 7b. Sim stack — install SELECTIVELY (do NOT run RoboTwin's `_install.sh`, it pins `torch==2.4.1`, G3)
```bash
uv pip install sapien==3.0.0b1 mplib==0.2.1 open3d trimesh "pyglet<2"
# verify torch is still 2.7.x AND `import sapien` works
# curobo / pytorch3d are NOT installed — see patches below.
```
Assets (~16 GB):
```bash
emimic/bin/python external/RoboTwin/assets/_download.py        # background_texture/embodiments/objects .zip
# unzip each into external/RoboTwin/assets/, then:
emimic/bin/python external/RoboTwin/script/update_embodiment_config_path.py
```

### 7c. Patch the RoboTwin fork (lets policy eval run without curobo)
Policy eval drives joints directly (qpos), so no motion planning is needed:
1. `external/RoboTwin/envs/robot/planner.py` — in the curobo-import `except`, define a
   stub `class CuroboPlanner` (constructible; a real `plan_grippers` that linearly
   interpolates `{num_step, per_step, result}`; `__getattr__` returns a no-op
   `{"status":"Fail"}` for the unused arm planner).
2. `external/RoboTwin/envs/robot/robot.py` `set_planner` — force
   `self.communication_flag = False` (in-process stub, not curobo worker subprocesses).
3. `external/RoboTwin/script/eval_policy.py` — `expert_check = False`; guard the
   `episode_info`/`generate_episode_descriptions` block and fall back to the task name
   as the instruction; honor `EVAL_TEST_NUM` to bound episode count.
4. `external/RoboTwin/task_config/demo_clean.yml` — `eval_video_log: false`.
5. `external/RoboTwin/policy/pi_ricl_egoverse/deploy_policy.yml` — set absolute
   `egoverse_checkpoint`, `bank_root`, `bank_index_dir`, `quantiles_path`.

### 7d. Run
Edit `egomimic/ricl/scripts/eval_robotwin_ricl.sbatch` partition/account, then:
```bash
CKPT=<abs path to .ckpt> EVAL_TEST_NUM=10 \
  sbatch egomimic/ricl/scripts/eval_robotwin_ricl.sbatch
```
The sbatch activates the venv, sets `TORCH_COMPILE_DISABLE=1` (eval-only — pi0.5
`sample_actions` otherwise triggers a multi-minute compile, G5) and runs `python -u`.
Success: log shows `Render Well` then `Success rate: k/n` per episode and `eval exit: 0`.

---

## 8. Gotchas / troubleshooting

- **G1 — bad GPU node.** `nvidia-smi` shows the GPU but torch errors `No CUDA GPUs are
  available` → that node's driver is too old for torch cu126. Exclude it:
  `sbatch --exclude=<node> ...`.
- **G2 — venv has no `pip`.** Use `uv pip install`. Bare `pip` silently hits a stray
  system Python; `python -m pip` says "No module named pip".
- **G3 — RoboTwin deps pin old torch.** Never install RoboTwin's `requirements.txt` or
  `_install.sh` wholesale (pins `torch==2.4.1`, breaks the shared env). Install sim deps
  selectively and verify torch is unchanged after every install (`uv pip install
  --dry-run ...` first).
- **G4 — `sbatch --wrap` uses `dash`.** `source` fails; call `emimic/bin/python` (or
  `. emimic/bin/activate`) directly.
- **G5 — torch.compile.** Leave compile ON for training; set `TORCH_COMPILE_DISABLE=1`
  for eval/smoke (avoids minutes of max-autotune warmup).
- **G6 — inferno preemption.** The shared `inferno`/`gpu-h200` queue preempts
  allocations mid-run, which can wedge a held `salloc`/`srun`. Submit unattended jobs
  with `sbatch`, not interactive `salloc`.
- **G7 — buffered logs.** Run `python -u`; otherwise stdout (episode prints) only flushes
  on exit when redirected to a file.
- **G8 — `--root` nesting.** Always point data `--root` at `.../extracted` (the recursive
  `**/data` glob finds the nested task dir).

---

## 9. Quick verification ladder

1. `pytest egomimic/ricl/tests/robotwin_*_test.py -q` → all pass (CPU, synthetic).
2. `--stage cpu` smoke → `ALL CHECKS PASSED`.
3. Short GPU train (`MAX_STEPS=10`) → produces a `.ckpt`, loss prints.
4. Build bank index → `vectors.npy`/`refs.npz`/`manifest.json`.
5. Eval `EVAL_TEST_NUM=2` → `Render Well` + `Success rate: k/2` + `eval exit: 0`.
