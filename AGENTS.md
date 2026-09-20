# Repo Agent Rules

## Shell / Command Execution
to run commands in the interactive shell make sure to source emimic/bin/activate

Apply this before running project Python tooling (for example: `python`, `pytest`, `pip`).

## Tests
Tests live under `tests/`, never next to source in `egomimic/`.
- `tests/unit/`: hermetic, CPU-only, no network. Run `pytest tests/unit` before every commit.
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

## Cancelled jobs and leaked GPU memory
A cancelled DDP job can leave whole GPUs allocated, with `nvidia-smi` blaming PIDs that are not in
`/proc`. Nothing is wrong with the driver: the memory is held by **live** `pt_data_worker` processes
reparented to init.

Dataloader workers are forked after CUDA init, so each inherits the rank's `/dev/nvidia*` fds — and
ranks open *every* visible GPU, so one survivor pins the whole node. The worker cannot exit on its
own: it inherited both ends of its result pipe, so a dead rank never gives it an `EPIPE`, it blocks
in `pipe_write`, and PyTorch's parent check (which only runs between `index_queue` timeouts) is
never reached. `ProctrackType=proctrack/linuxproc` then walks the PPID tree and never sees it.

Whole ranks leak the same way. A rank wedged in `torch.cuda.synchronize()` inside inductor's
`cudagraph_trees` does not answer SIGTERM, Slurm hits `UnkillableStepTimeout` (60s) and gives up
(`srun: error: Timed out waiting for job step to complete`), and the job leaves the queue with its
ranks still spinning on every GPU. Enabling `torch.compile` makes this more likely, not less.

`egomimic/utils/gpu_orphans.py` owns the fix:

- `orphan_guarded(params)` / `die_with_parent` set `PR_SET_PDEATHSIG` on every worker, so the kernel
  kills it with its rank regardless of what it is blocked on. **Any new `DataLoader` on a training
  path needs one of them** — they are already on the train/val loaders and the norm-stats pass.
- `arm_rank_pdeathsig()` does the same for the rank process itself, so a launcher that gives up
  takes its ranks with it. Slurm-gated, so a detached interactive run is not shot when its shell
  exits.
- `reap_orphans()` runs from `trainHydra.main` before CUDA init and clears what an earlier job left
  on the node, so a leak cannot outlive the job that made it. It only ever signals processes that
  are yours, reparented to init and holding an nvidia fd; a live job's workers have a live parent,
  so they are never touched. Off outside Slurm and on non-zero `SLURM_LOCALID`; disable with
  `EGOMIMIC_REAP_GPU_ORPHANS=0`.
- `./reap_gpu_orphans.sh [--kill] <node>` is the manual version, for nodes you are not about to
  submit to. It attaches to your own job on a busy node rather than queueing behind it.

The cluster-level fix is `ProctrackType=proctrack/cgroup`, which sweeps by cgroup instead of by PPID
tree and would make all of this unnecessary. That needs admin; `/etc/slurm` is read-only to us.
