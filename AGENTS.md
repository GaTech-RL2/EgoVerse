# Repo Agent Rules

## Shell / Command Execution
Use the uv environment that `pyproject.toml` and `uv.lock` specify. Run
`uv sync` to create or update the environment. Run Python tools through uv, for
example, `uv run pytest` or `uv run ruff check egomimic`. You can also run
`source .venv/bin/activate` before you use the tools.

Do not use `pip install`. Add each dependency to `pyproject.toml`, and then run
`uv sync`. CI uses `uv sync --locked` and fails if the manifest and lock file do
not agree.

Set `UV_PROJECT_ENVIRONMENT` if the environment directory is not `.venv`.
The locked environment includes several gigabytes of CUDA packages. If the home
directory has a small quota, set `UV_CACHE_DIR` and
`UV_PYTHON_INSTALL_DIR` to directories on scratch storage.

## Tests
New tests go under `tests/`, never next to source in `egomimic/`. The embodiment-registry and
Zarr-schema suites in `egomimic/rldb/embodiment` and `egomimic/rldb/zarr` predate this layout;
`pyproject.toml` (`testpaths`) and CI still collect them until they move.
- `tests/unit/`: hermetic, CPU-only, no network. Run
  `pytest tests/unit egomimic/rldb/embodiment egomimic/rldb/zarr` (or a bare `pytest`) before
  every commit.
- `tests/integration/`: needs cluster data, credentials, GPUs, or optional extras. Skipped by
  default; run with `pytest --integration tests/integration`. The `gpu` marker only labels tests:
  `-m gpu` selects them, `-m "not gpu"` leaves them out, and without a CUDA device they skip. Data
  paths come from `EGOVERSE_TEST_*` env vars, not hardcoded home directories.
- `tests/unit/test_train_step.py` is the per-vendor train-step matrix (HPT + Pi x eva/aria/mecka/scale)
  on synthetic episodes; run it after touching keymaps, transforms, data/model configs or the algos.
  The Pi half needs `openpi` and the PaliGemma tokenizer in the venv and skips without them; CI
  installs both (see the unit-tests job in `.github/workflows/ci.yml`) and fails instead of
  skipping. Its GPU twin on the real recipes is
  `pytest --integration -m gpu tests/integration/test_train_step_gpu.py` on a GPU node.
  `tests/unit/test_data_configs_compose.py` checks every `data/*.yaml` composes.

## Model settings
use plan mode for anything except extremely simple tasks

## Slurm rules
If you're on a slurm cluster, request a GPU before running or testing training.
On sky1/sky2: salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G
