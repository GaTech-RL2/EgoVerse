# Repo Agent Rules

## Shell / Command Execution
to run commands in the interactive shell make sure to source emimic/bin/activate

Apply this before running project Python tooling (for example: `python`, `pytest`, `pip`).

## Tests
Tests live under `tests/`, never next to source in `egomimic/`.
- `tests/unit/`: hermetic, CPU-only, no network. Run `pytest tests/unit` before every commit.
- `tests/integration/`: needs cluster data, credentials, GPUs, or optional extras. Skipped by
  default; run with `pytest --integration tests/integration`. Data paths come from
  `EGOVERSE_TEST_*` env vars, not hardcoded home directories.
- `tests/unit/test_train_step.py` is the per-vendor train-step matrix (HPT + Pi x eva/aria/mecka/scale)
  on synthetic episodes; run it after touching keymaps, transforms, data/model configs or the algos.
  Its GPU twin on the real recipes is `pytest --integration -m gpu tests/integration/test_train_step_gpu.py`
  on a GPU node. `tests/unit/test_data_configs_compose.py` checks every `data/*.yaml` composes.

## Model settings
use plan mode for anything except extremely simple tasks

## Slurm rules
If you're on a slurm cluster, request a GPU before running or testing training.
On sky1/sky2: salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G