# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Environment

- **You are on a shared SLURM cluster.** Do not run anything GPU- or CPU-intensive yourself unless told to (no training, no eval, no large data conversions, no full dataset loads, no heavy `pytest` runs that spin up models or pull data). Defer to the user to actually execute those commands — your job is to prepare the command and explain it. Lightweight read-only work (lint, type checks, small unit tests, file edits, single-file syntax checks) is fine on the login node.
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
