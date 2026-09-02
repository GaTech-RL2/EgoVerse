# Repo Agent Rules

## Shell / Command Execution
The project environment is the uv environment described by `pyproject.toml` +
`uv.lock`. Create or update it with `uv sync`, and run project Python tooling
through it — `uv run pytest`, `uv run ruff check egomimic` — or activate it
(`source .venv/bin/activate`) first. Do not `pip install` into it; add the
dependency to `pyproject.toml` and re-run `uv sync`, or CI's `uv sync --locked`
will fail.

If your environment lives under a different name (this machine uses `ev`), set
`UV_PROJECT_ENVIRONMENT` to it. Point `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`
at scratch storage on any machine with a small home quota — the locked
environment carries the CUDA torch stack and is several GB.

## Model settings
use plan mode for anything except extremely simple tasks

## Slurm rules
If you're on a slurm cluster, request a GPU before running or testing training.
On sky1/sky2: salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G