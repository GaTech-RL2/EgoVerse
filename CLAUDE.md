# EgoVerse — Claude Code Instructions

## Modal (version 1.4.x)

This project uses Modal for cloud GPU training. Modal 1.x has breaking changes from 0.x.
Always follow the patterns below — do NOT use the old API.

### Container detection

Modal 1.x removed `modal.is_remote()` and `modal.is_local()`. Detect whether code is
running inside a Modal container via the environment variable:

```python
import os
is_inside_modal = os.environ.get("MODAL_IS_REMOTE") == "1"
```

### Renamed / removed APIs (0.x → 1.x)

| Old (0.x) | New (1.x) |
|-----------|-----------|
| `modal.is_remote()` | `os.environ.get("MODAL_IS_REMOTE") == "1"` |
| `modal.is_local()` | `os.environ.get("MODAL_IS_REMOTE") != "1"` |
| `Function.with_options(...)` | Not supported — pass args as function parameters |
| `modal.gpu.A100()` | `"A100"` (string literal) |
| `keep_warm=N` | `min_containers=N` |
| `concurrency_limit=N` | `max_containers=N` |
| `container_idle_timeout=N` | `scaledown_window=N` |
| `allow_concurrent_inputs=N` | `@modal.concurrent(max_inputs=N)` |
| `.lookup()` | `.from_name()` |
| `modal.web_endpoint` | `modal.fastapi_endpoint` |
| `Image.copy_local_dir` | `Image.add_local_dir` |
| `Image.copy_local_file` | `Image.add_local_file` |

### Core patterns

```python
import modal

app = modal.App("my-app", image=image)

# Function decorator
@app.function(gpu="A100", timeout=3600, secrets=[modal.Secret.from_name("my-secret")])
def my_func(arg):
    import heavy_dep  # imports go INSIDE functions, not at module level
    ...

# Invocation
my_func.remote(arg)      # blocking remote call
my_func.spawn(arg)       # fire-and-forget, returns handle
my_func.local(arg)       # run locally (same process)
list(my_func.map(args))  # parallel map

# Volumes
vol = modal.Volume.from_name("my-volume")

@app.function(volumes={"/mnt/data": vol})
def write_data():
    ...
    vol.commit()  # required after writes to persist to volume

# Secrets — injected as env vars inside container
modal.Secret.from_name("my-secret")  # references a secret stored in Modal dashboard
```

### Project-specific Modal setup

- **App name**: `egomimic-training` (training), `egomimic-ingest-zarr` (data ingestion)
- **Environment**: `robotics` (set via `MODAL_ENVIRONMENT=robotics`)
- **Volume**: `egoverse-zarr-data`, mounted at `/mnt/zarr-data` inside containers
- **Shared secrets**: `egoverse-r2` (R2 creds), `egoverse-mongodb` (MongoDB URI)
- **WandB**: NOT a shared secret — each user injects their own `WANDB_API_KEY` from
  `~/.egoverse_env` at submission time via `_local_wandb_key()` in `test_run.py`
- **Container detection**: `os.environ.get("MODAL_IS_REMOTE") == "1"`
- **Trainer flag**: `trainer._modal: true` in `ddp_modal.yaml` signals `trainHydra.py`
  to submit to Modal instead of running locally

### Key files

- `egomimic/modal/modal_config.py` — image, volume, app, and `CFG` (gpu/cpu/memory/timeout)
- `egomimic/modal/test_run.py` — training submission entrypoints (`submit`, `run`, `verify`)
- `egomimic/modal/ingest_zarr.py` — standalone script; the ONLY place that queries the
  legacy AWS RDS SQL table; downloads zarr episodes from old `rldb` R2 bucket into the volume
- `egomimic/hydra_configs/trainer/ddp_modal.yaml` — Lightning trainer config for Modal
- `egomimic/hydra_configs/data/mecka_all_zarr.yaml` — `LocalEpisodeResolver` over `/mnt/zarr-data`

### Credential files (local machine)

- `~/.egoverse_env` — new Mecka R2 + DigitalOcean DB (used by training and data processing)
- `~/.egoverse_env_old` — legacy rldb R2 + AWS RDS (used by `ingest_zarr.py` ONLY)

### Running training

```bash
python egomimic/trainHydra.py \
  name=<run_name> \
  description=<description> \
  data=mecka_all_zarr \
  model=<model_config> \
  trainer=ddp_modal
```

`trainHydra.py` detects `trainer._modal=true` + `MODAL_IS_REMOTE != 1`, collects all
Hydra overrides, and calls `run_hydra_train.spawn(...)` — then exits immediately.
Inside the Modal container `MODAL_IS_REMOTE=1` so training runs normally.

### Modal docs reference

- Full API reference and guides: https://modal.com/docs/guide
- LLM-optimised reference: https://modal.com/llms-full.txt
- Migration guide (0.x → 1.x): https://modal.com/docs/guide/modal-1-0-migration.md
