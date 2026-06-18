# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Maintaining this file

Whenever you learn something important that would help future work in this repo — a non-obvious convention, a gotcha, a fix to a recurring problem, a corrected assumption, or a workflow that isn't documented here — update this CLAUDE.md to capture it. Keep additions concise and place them in the relevant section. Don't wait to be asked.

## This branch: RICL (`ryanco/in-context-learning`)

This branch exists for **one experiment — RICL** (retrieval-based in-context learning on
pi0.5). Start at `egomimic/ricl/CLAUDE.md` (navigation) and `egomimic/ricl/README.md`
(architecture). The RICL working set is: `egomimic/ricl/**`, `algo/pi_ricl.py` (+ its
parent `algo/pi.py`), `eval/pi_ricl_eval.py`, `pl_utils/pl_data_utils.py`, the
`*_ricl*` / `ricl_stats_*` / `eva_pi` / `pi0.5_ricl` configs,
`scripts/embedding_process/zarr_embedding.py`, `scripts/human_robot_pairs.json`, and
the shared infra `trainHydra.py` + `rldb/**` + `utils/action_utils.py`.

**Not part of this branch — skip unless explicitly needed** (avoid filling context):
other algos/models/evals (`algo/{act,hpt}.py`, `models/{act_nets,hpt_nets,denoising_*,
diffusion_policy,ddim_scheduler}.py`, `eval/{eval_act,eval_hpt,eval_latent,eval_video}.py`);
`egomimic/robot/**`; most `egomimic/scripts/*` subdirs (`aria_process`, `eva_process`,
`tutorials`, `language_process`, `mecka_process`, `mps_process`, `data_download`,
`data_upload`, `data_visualization`, `backfill_scripts`, `benchmark`, `plotting`,
`calibrate_camera`, `evaluation`); `external/{lerobot,scale,rpl_vision_utils}/**`;
all `*.ipynb`. **Never read into context**: venvs (`emimic/`, `.venv/`), caches
(`**/__pycache__`, `.pytest_cache`, `.ruff_cache`, `egomimic.egg-info`), outputs
(`outputs/`, `egomimic/logs/`, `egomimic/ricl/outputs/`, `egomimic/ricl/pg_tokenizer/`,
`assets/`), any `*.zarr`, and the large data files noted in
`egomimic/ricl/CLAUDE.md`.

## Environment

- **You are on a shared SLURM cluster.** Don't run GPU/CPU-intensive work on the login node. Grab an interactive node first with `salloc`, e.g.:
  ```
  salloc -A gts-dxu345-rl2 -N1 -q inferno -t 1:00:00 --mem=75G --gres=gpu:h200:1
  ```
  Always use the `inferno` queue (`-q inferno`) rather than `ember` — it's faster. Adjust `-t`, `--mem`, and `--gres` to the job. `salloc` is best for interactive / iterative work (smoke tests, debugging) where you hold the node and run into it repeatedly. For large or long-running jobs (real training runs), submit through Hydra's submitit launcher instead (`hydra/launcher/submitit.yaml`) so the job queues and runs unattended. Lightweight read-only work (lint, type checks, small unit tests, file edits, single-file syntax checks) is fine on the login node.
- **Pick the GPU variant by job — don't always queue for `h200`.** `gpu-h200` is the most contended partition, so jobs sit. Choose by what the job needs to fit in VRAM (pi0.5 is 3.6 B params ≈ **14.5 GB fp32 / ~7 GB bf16** for the weights alone). Each type is its own partition `gpu-<type>` with `--gres=gpu:<type>:N`, same `-A gts-dxu345-rl2 -q inferno`:
  | partition | `--gres` type | GPU mem | use for |
  |---|---|---|---|
  | `gpu-h200` | `h200` | 141 GB | **pi0.5 full fp32 AdamW training** (model+optimizer ≈ 57 GB) |
  | `gpu-rtxpro-blackwell` | `rtx_pro_6000_blackwell` | 96 GB | training (newer, fewer nodes) |
  | `gpu-h100` | `h100` | 80 GB | training |
  | `gpu-l40s` | `l40s` | 48 GB | **pi0.5 eval/inference** (the eval sbatch was originally sized for a 48 GB a40) |
  | `gpu-a100` | `a100` | 40 GB | **pi0.5 eval/inference**; training only with `--adam8bit` + small batch |
  | `gpu-rtx6000` | `rtx_6000` | 24 GB | small / bf16-only inference — tight for pi0.5, plentiful nodes |
  | `gpu-v100` | `v100` | 16 GB | too small to load pi0.5 comfortably — avoid |
  Rule of thumb: **training needs ≥80 GB** (h200/h100/blackwell); **eval/inference fits 40–48 GB** (l40s/a100, with room for DINOv2 retrieval + SAPIEN) and those queues schedule far faster than h200. **Each partition caps the CPU:GPU ratio differently** — `gpu-h200` is 8:1, `gpu-l40s` is **4:1** — so `--cpus-per-task` must be ≤ ratio × #GPUs or the job is rejected with "Invalid generic resource (gres) specification" (e.g. 8 CPUs / 1 l40s fails; use 4). Validate a header without queuing via `sbatch --test-only <script>`.
- **Short GPU runs (eval-only, smoke, a few hundred forward passes): export `TORCH_COMPILE_DISABLE=1`.** pi0.5's `sample_actions` triggers a `torch.compile` max-autotune compile on the first call — minutes of warmup that only pays off across a long training run. Disabling it runs eager (slower per call, no warmup), a net win when you're not training for a while. Leave compile ON for real training.
- Python 3.11. Activate the project venv before any Python tooling: `source emimic/bin/activate`.
- Package is installed editable as `egomimic` (see `pyproject.toml`). Linting is `ruff` via pre-commit.
- AWS/Cloudflare R2 credentials are required for SQL episode registry + data download. Bootstrap with `aws configure` then `./egomimic/utils/aws/setup_secret.sh` (writes `~/.egoverse_env`). `load_env()` from `egomimic.utils.aws.aws_data_utils` is called automatically at the top of `trainHydra.py`.

## Architecture

### Training entrypoint
`egomimic/trainHydra.py` is the only training entrypoint. It is Hydra + PyTorch Lightning, supports DDP, and is composed entirely via config groups under `egomimic/hydra_configs/`:

- `train_zarr_cartesian.yaml`, `train_zarr_cartesian_pi.yaml`, `train_zarr_keypoint*.yaml` — top-level configs that pick a default `data=`, `model=`, `trainer=`, `evaluator=`, `logger=`, `callbacks=`.
- `data/` — one YAML per dataset recipe. Each instantiates `MultiDataset._from_resolver` with an `S3EpisodeResolver` + `key_map` + `transform_list` + `filters`. Modify these (or override inline) to change which episodes are pulled.
- `model/` — model recipes. Two families: HPT-based (`hpt_*.yaml`) and Pi/Pi0.5 flow-matching (`pi0.5_*.yaml`). Cotrain configs combine a robot + human dataset into a shared/separate-head model.
- `hydra/launcher/submitit.yaml` — SLURM partition/GPUs/time. Edit to match the cluster.
- `logger=debug` and `trainer=debug` exist for fast iteration.

### Data pipeline (read this before touching any data code)
Code lives under `egomimic/rldb/`. Flow is:
1. **SQL filter** (`rldb/filters.py`) selects rows from the `app.episodes` Postgres table.
2. **`S3EpisodeResolver`** (`rldb/zarr/zarr_dataset_multi.py`) maps filtered rows → S3 Zarr URIs and lazily downloads to a local cache.
3. **`ZarrEpisode`** reads each `<episode_hash>.zarr` (Zarr v3, see `CONTRIBUTING_DATA.md` for the schema contract).
4. **Key map** (per-embodiment, `rldb/embodiment/{eva,human}.py`) renames raw zarr keys to pre-transform names.
5. **Transform list** (same files) applies frame conversions (SLAM world → head frame via `ActionChunkCoordinateFrameTransform`), action chunking, normalization, concatenation. All poses are stored in SLAM world frame and re-expressed to head frame at load time — do **not** pre-transform when writing.
6. **`MultiDataset`** virtually merges per-episode datasets; `mode` ∈ {`train`, `valid`, `percent`, `total`} controls sampling.
7. **`DataSchematic`** in the top-level config maps post-transform keys → batch keys consumed by the model.
8. **Norm stats** are computed on-the-fly over `norm_stats.sample_frac` of the data and cached to the run's `norm_stats/` dir. Lower `sample_frac` for large datasets; reuse via `norm_stats.precomputed_norm_path`.

`egomimic/rldb/zarr/zarr_writer.py` is the only supported way to *produce* Zarr stores. Don't roll a custom writer — sharding/chunking conventions must match the rest of the corpus.

### Algorithms and models
- `egomimic/algo/` — top-level policy classes: `act.py`, `hpt.py`, `pi.py` (Pi/Pi0.5). They wrap nets from `egomimic/models/`.
- `egomimic/models/` — `act_nets.py`, `hpt_nets.py`, `denoising_*`, `*_policy.py`, `preprocess_pi_obs.py`. `fm_policy.py` is flow-matching, `diffusion_policy.py` is DDIM.
- `egomimic/pl_utils/pl_model.py` — Lightning `ModelWrapper` used by `trainHydra.py`.
- `egomimic/eval/` — evaluators dispatched by `+evaluator=eval_act|eval_hpt|eval_pi|eval_video`.

### Embodiments
Defined as strings (see `CONTRIBUTING_DATA.md` §8 for the full list): `aria_bimanual`, `eva_bimanual`, `mecka_bimanual`, `scale_bimanual`, plus single-arm variants. The `embodiment` value in both the SQL row and `zarr.attrs` must match exactly. `rldb/embodiment/{eva,human}.py` expose `get_keymap()` and `get_transform_list()` per embodiment.
