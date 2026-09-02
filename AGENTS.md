# Repo Agent Rules

## Shell / Command Execution
Use the uv environment that `pyproject.toml` and `uv.lock` specify. Run
`uv sync` to create or update the environment. Run Python tools through uv, for
example, `uv run pytest` or `uv run ruff check egomimic`. You can also run
`source .venv/bin/activate` before you use the tools.

Do not use `pip install`. Add each dependency to `pyproject.toml`, and then run
`uv sync`. CI uses `uv sync --locked` and fails if the manifest and lock file do
not agree.

Set `UV_PROJECT_ENVIRONMENT` if the environment directory is not `.venv`. This
machine uses `ev`. The locked environment includes several gigabytes of CUDA
packages. If the home directory has a small quota, set `UV_CACHE_DIR` and
`UV_PYTHON_INSTALL_DIR` to directories on scratch storage.

## Model settings
use plan mode for anything except extremely simple tasks

## Slurm rules
If you're on a slurm cluster, request a GPU before running or testing training.
On sky1/sky2: salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G
