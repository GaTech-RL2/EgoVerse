# Rollout install (Python 3.10)

End-to-end setup for running policy rollouts on a real robot. The training stack
(zarr, lerobot, etc.) is intentionally **not** required here — this guide is the
minimum set of dependencies to load a checkpoint and drive an ARX arm.

## Why 3.10

The Stanford ARX `arx5_interface` C extension is shipped as a
`cpython-310-x86_64-linux-gnu.so`, so the rollout host must run Python 3.10.
The Docker image we use already has 3.10 system-wide and CUDA / ROS / ARX
pre-built against it; this guide installs everything into that same
interpreter rather than into a venv.

openpi upstream is developed on 3.11 and pulls in lerobot, which transitively
requires FFmpeg 7. Both are problems on 3.10 + Ubuntu 22.04, so we use a
rollout-only fork of openpi with two small patches (Python version + lerobot
removed). See `external/openpi` instructions below.

## Prerequisites

- Docker container based on the team's Eva image (system Python 3.10, CUDA 12.x,
  ROS Humble at `/opt/ros/humble`, ARX prebuilt under
  `/opt/ros/humble/lib/python3.10/site-packages/arx5/`).
- A working `python3 -c "import jax; print(jax.devices())"` returning a CUDA
  device (JAX is needed for the optional checkpoint conversion path).
- A Hugging Face access token with read access to
  [`google/paligemma-3b-mix-224`](https://huggingface.co/google/paligemma-3b-mix-224).
  This repo is **gated** — you must accept the license on the model page once,
  then mint a token. The collate function in `pl_data_utils.py` downloads the
  PaliGemma tokenizer at rollout startup, so without a token rollout aborts
  with a `GatedRepoError`.

## Branches

This guide assumes:

- **egomimic** — checked out on the rollout branch (the one carrying
  `rollout-requirements.txt` and the relaxed `requires-python` in
  `pyproject.toml`).
- **external/openpi** — checked out on a `rollout` branch of our fork
  (`https://github.com/GaTech-RL2/openpi`) with these modifications already
  committed:
  - `.python-version` changed `3.11` → `3.10`.
  - `pyproject.toml` `dependencies` no longer lists `lerobot`.
  - `[tool.uv.sources]` no longer pins the lerobot git rev.
  - `packages/openpi-client/pyproject.toml` `requires-python` raised from
    `>=3.7` to `>=3.10`.
  - `src/openpi/shared/download.py` uses `datetime.timezone.utc` instead of
    the 3.11-only `datetime.UTC`.

If you're recreating these patches by hand from upstream, the diffs are small —
search this guide for the exact symbols above.

## Step 1 — clone with submodules

```bash
git clone <egomimic-repo>
cd egomimic
git checkout <rollout-branch>
git submodule update --init --recursive external/openpi
git -C external/openpi checkout <openpi-rollout-branch>
```

`external/openpi` ends up at the openpi rollout-branch commit and pulls in its
own nested submodules (`third_party/aloha`, `third_party/libero`).

## Step 2 — install egomimic rollout deps

Install into system Python 3.10 (not a venv — see "Why 3.10").

```bash
pip install -r rollout-requirements.txt
pip install -e .
```

`rollout-requirements.txt` is the trimmed sibling of `requirements.txt` — no
`zarr` (training-only). `pyproject.toml` has its `requires-python` relaxed to
`>=3.10` and `zarr` removed from `dependencies`, matching the rollout-branch
edits.

## Step 3 — install openpi deps

We use uv to resolve openpi's lockfile against 3.10, then hand the constraints
to pip so we install into the system interpreter.

```bash
# uv only used for resolution
curl -LsSf https://astral.sh/uv/install.sh | sh   # if not already present

cd external/openpi
uv lock                                            # rewrites uv.lock for 3.10
uv export --no-hashes --no-emit-workspace --quiet > /tmp/openpi-constraints.txt

pip install -e . -c /tmp/openpi-constraints.txt
```

`--no-emit-workspace` strips editable workspace entries that pip's `-c` would
choke on. The first `uv lock` is required because the bundled `uv.lock` was
generated against 3.11 and contains lerobot transitives.

## Step 4 — patch transformers

openpi ships replacement files for a handful of transformers internals (Gemma
attention, etc.). They have to be copied into the active transformers package
on disk.

```bash
cd external/openpi
python3 scripts/patch_transformers.py
```

The script auto-resolves the destination via `importlib.util.find_spec`, so it
works regardless of where transformers lives.

## Step 5 — pin torchvision to match torch

openpi pins `torch==2.7.1`. The pre-existing system `torchvision` was built
against an older torch, so its `_meta_registrations` references C++ ops that
no longer exist on 2.7.1 (`RuntimeError: operator torchvision::nms does not
exist`). Pin to the matching wheel:

```bash
pip install torchvision==0.22.1
```

If pip can't find a wheel, force the PyTorch index:

```bash
pip install torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

(Replace `cu124` with whatever `python3 -c 'import torch; print(torch.version.cuda)'`
prints — `12.1` → `cu121`, etc.)

## Step 6 — expose ARX to the active interpreter

The arx5 binding lives at `/opt/ros/humble/lib/python3.10/site-packages/arx5/`,
which is on `PYTHONPATH` only when ROS is sourced. For the rollout we add a
`.pth` file so it's visible whether or not the user remembers to source ROS:

```bash
echo "/opt/ros/humble/lib/python3.10/site-packages" \
  > $(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/ros_arx5.pth

python3 -c "import arx5.arx5_interface; print('arx5 OK')"
```

If `arx5` imports cleanly but ARX initialization later fails to find native
shared libs (e.g. `libArxJointController.so`), source ROS once per shell so
`LD_LIBRARY_PATH` is populated:

```bash
source /opt/ros/humble/setup.bash
```

The `.pth` fix only handles Python-level imports; LD lookups for the C++
libraries still need the env var.

## Step 7 — Hugging Face authentication

Required because the rollout's collate function instantiates a PaliGemma
tokenizer, and `google/paligemma-3b-mix-224` is a gated repo.

1. Visit https://huggingface.co/google/paligemma-3b-mix-224, click "Agree and
   access repository" (one-time per HF account).
2. Mint a read token at https://huggingface.co/settings/tokens.
3. Authenticate the host:

   ```bash
   huggingface-cli login
   # or set non-interactively:
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
   ```

   `transformers.AutoTokenizer.from_pretrained` reads `HF_TOKEN` (or the cached
   credentials from `huggingface-cli login`) automatically.

Sanity check:

```bash
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-mix-224'); print('hf OK')"
```

## Step 8 — get the PyTorch checkpoint

The rollout loads a converted PyTorch checkpoint. There are two paths:

### 8a — rsync from cloud (preferred)

If a teammate has already converted weights for the same openpi commit, just
copy the directory:

```bash
rsync -av <cloud-host>:/path/to/pi05_base_pytorch/ \
       egomimic/algo/pi_checkpoints/pi05_base_pytorch/
```

The output is a deterministic function of the JAX checkpoint + openpi
revision, so a copy from another machine is byte-identical to running the
conversion locally. `safetensors` files are torch-version-independent.

### 8b — convert from JAX locally

Only needed if you don't have a pre-converted copy. The JAX checkpoint
(`pi05_base/`) is fetched from the team's GCS bucket; you'll need `gsutil`
configured. Then:

```bash
cd external/openpi
python3 examples/convert_jax_model_to_pytorch.py \
  --config_name pi05_aloha \
  --checkpoint_dir ../../egomimic/algo/pi_checkpoints/pi05_base \
  --output_path ../../egomimic/algo/pi_checkpoints/pi05_base_pytorch
```

The conversion needs JAX with CUDA (12 GB GPU is enough), the patched
transformers from Step 4, and a working torchvision from Step 5.

## Verification

```bash
python3 -c "
import torch, torchvision, transformers, openpi
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import arx5.arx5_interface
print('torch', torch.__version__, 'torchvision', torchvision.__version__)
print('transformers', transformers.__version__)
print('openpi', openpi.__file__)
print('arx5', arx5.arx5_interface.__file__)
print('all OK')
"
```

All five lines should print without errors.

## Known incompatibilities to keep an eye on

- `datetime.UTC`, `Self` (from `typing`), and `type X = ...` syntax are 3.11+.
  We've patched the one occurrence in
  `src/openpi/shared/download.py`. If you `git pull` upstream openpi changes,
  re-grep for `datetime.UTC` and similar.
- `transformers` major-version bumps occasionally invalidate the
  `models_pytorch/transformers_replace/` files. Re-run
  `scripts/patch_transformers.py` after any transformers upgrade.
- The `pynvml` deprecation warning at startup is harmless and unrelated;
  ignore it.
