"""Modal training entrypoints for EgoVerse.

Usage (recommended — via trainHydra.py)
----------------------------------------
    python egomimic/trainHydra.py trainer=ddp_modal data=<data> model=<model> \\
        name=<run> description=<desc> \\
        [+modal_gpu=H100] [+modal_cpu=32] [+modal_memory_gb=128]

Direct fire-and-forget:
    modal run egomimic/modal/run.py::submit -- \\
        data=<data> trainer=ddp_modal model=<model> name=<run>

Direct blocking (streams logs, downloads artifacts when done):
    modal run egomimic/modal/run.py::run -- \\
        data=<data> trainer=ddp_modal model=<model> name=<run>

Verify container setup:
    modal run egomimic/modal/run.py::verify

Volume configuration (optional)
---------------------------------
Set EGOVERSE_MODAL_ZARR_VOLUME to mount a pre-existing zarr dataset volume.
Set EGOVERSE_MODAL_OUTPUT_VOLUME to persist checkpoints/logs (default: "egoverse-training-outputs").
Set either to "" to disable that volume.

GPU / compute
--------------
MODAL_GPU (default "A100"), MODAL_CPU (default 12.0), MODAL_MEMORY_GB (default 64).
trainHydra.py sets these from +modal_gpu= / +modal_cpu= / +modal_memory_gb= overrides.

App name (Modal dashboard / logs)
----------------------------------
MODAL_APP_NAME sets the Modal App name (default ``egoverse-train``). trainHydra.py sets it
from Hydra ``name`` and ``description`` as ``<name>-<description>`` when using trainer._modal.

Credentials
------------
~/.egoverse_env is copied into the container automatically if it exists.
WANDB_API_KEY: read from ~/.egoverse_env or local env.

Notes
------
- This file is self-contained (no egomimic imports at module level) because Modal
  mounts it before the repo is cloned inside the container.
- Modal runs the *committed* git state. Push before submitting a real run.
- SSH remote URLs are converted to HTTPS automatically for container cloning.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "main")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_EGOVERSE_ENV_REMOTE_PATH = "/root/.egoverse_env"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    remote_repo_dir: str = "/root/EgoVerse"
    python_bin: str = "python3"

    zarr_volume_mount_path: str = "/mnt/egoverse-data"
    output_mount_path: str = "/root/EgoVerse/logs"

    gpu: str = field(default_factory=lambda: os.environ.get("MODAL_GPU", "A100"))
    cpu: float = field(
        default_factory=lambda: float(os.environ.get("MODAL_CPU", "12.0"))
    )
    memory_mb: int = field(
        default_factory=lambda: (
            int(float(os.environ.get("MODAL_MEMORY_GB")) * 1024)
            if os.environ.get("MODAL_MEMORY_GB")
            else int(os.environ.get("MODAL_MEMORY_MB", "65536"))
        )
    )
    timeout_seconds: int = 86400

    secret_names: list[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.environ.get(
                "EGOVERSE_MODAL_SECRETS", "egoverse-env,egoverse-aws"
            ).split(",")
            if s.strip()
        ]
    )

    @property
    def train_script(self) -> str:
        return f"{self.remote_repo_dir}/egomimic/trainHydra.py"

    @property
    def venv_python(self) -> str:
        return f"{self.remote_repo_dir}/.venv/bin/python"


CFG = _Config()

# ---------------------------------------------------------------------------
# Volumes — always mounted when running on Modal
#   egoverse-data             → /mnt/egoverse-data   (all zarr episode data)
#   egoverse-training-outputs → /root/EgoVerse/logs  (checkpoints, logs, videos)
# ---------------------------------------------------------------------------

zarr_volume = modal.Volume.from_name("egoverse-data")
training_outputs_volume = modal.Volume.from_name(
    "egoverse-training-outputs", create_if_missing=True
)
_volumes: dict = {
    CFG.zarr_volume_mount_path: zarr_volume,
    CFG.output_mount_path: training_outputs_volume,
}

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------


def _resolve_egoverse_env_local_file() -> Path | None:
    override = os.environ.get("EGOVERSE_MODAL_EGOVERSE_ENV_FILE", "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(REPO_ROOT / ".egoverse_env")
    candidates.append(Path.home() / ".egoverse_env")
    for p in candidates:
        try:
            if p.is_file():
                return p.resolve()
        except OSError:
            continue
    return None


_base_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    .apt_install("git", "libgl1", "libglib2.0-0")
    # uv for fast, reproducible env management; modal for the auto-restart callback
    .pip_install("uv", "modal")
    # Bake requirements.txt into the image layer so the per-run venv creation is instant.
    # torch/torchvision are excluded — they're already present in the pytorch base image
    # with CUDA support and reinstalling from PyPI would pull the CPU-only build.
    .add_local_file(
        str(REPO_ROOT / "requirements.txt"),
        "/tmp/requirements.txt",
        copy=True,
    )
    .run_commands(
        # setuptools_scm is a build-time dep required by gpustat and similar packages
        "uv pip install --system setuptools_scm && "
        "grep -vE '^(torch|torchvision)' /tmp/requirements.txt | "
        "uv pip install --system -r /dev/stdin"
    )
)

_env_src = _resolve_egoverse_env_local_file()
if _env_src is not None:
    image = _base_image.add_local_file(
        str(_env_src),
        remote_path=_EGOVERSE_ENV_REMOTE_PATH,
        copy=True,
    )
    print(
        f"Modal image will include AWS/R2 env file: {_env_src} -> {_EGOVERSE_ENV_REMOTE_PATH}"
    )
else:
    image = _base_image

_MODAL_APP_DEFAULT = "egoverse-train"
_modal_app_name = (
    os.environ.get("MODAL_APP_NAME", _MODAL_APP_DEFAULT).strip() or _MODAL_APP_DEFAULT
)
app = modal.App(_modal_app_name, image=image)

# ---------------------------------------------------------------------------
# Local helpers (run on the submitting machine)
# ---------------------------------------------------------------------------


def _local_wandb_key() -> str:
    """Read WANDB_API_KEY from ~/.egoverse_env or local environment."""
    env_file = Path("~/.egoverse_env").expanduser()
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("WANDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    os.environ["WANDB_API_KEY"] = key
    key = os.environ.get("WANDB_API_KEY", "")
    if not key:
        print(
            "Warning: WANDB_API_KEY not found — W&B logging will be disabled. "
            "Add it to ~/.egoverse_env to enable."
        )
    return key


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def _resolve_git_state() -> tuple[str, str, bool]:
    """Return (remote_url, commit_sha, is_dirty).

    Exits with a clear error if HEAD has not been pushed — the container
    clones from GitHub so unpushed commits would cause a checkout failure there.
    """
    git_remote = _git_output(["git", "config", "--get", "remote.origin.url"])
    git_commit = _git_output(["git", "rev-parse", "HEAD"])
    is_dirty = bool(_git_output(["git", "status", "--porcelain"]))

    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "branch", "-r", "--contains", git_commit],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            raise SystemExit(
                f"ERROR: commit {git_commit[:12]} has not been pushed to {git_remote}.\n"
                "Push your branch first, then re-run:\n"
                "  git push --set-upstream origin <branch>"
            )
    except subprocess.CalledProcessError:
        pass  # fetch failed (no network / auth) — let the container surface the error

    return git_remote, git_commit, is_dirty


def _build_train_cmd(hydra_args: tuple[str, ...]) -> list[str]:
    return [CFG.venv_python, CFG.train_script, *hydra_args]


def _resolve_volume_paths(hydra_args: tuple[str, ...]) -> tuple[str, ...]:
    """Prepend the output volume mount path to relative checkpoint/norm-stat paths.

    Allows passing  ckpt_path=my_run/checkpoints/last.ckpt
    instead of the full container path  ckpt_path=/root/EgoVerse/logs/...
    """
    _PATH_KEYS = {"ckpt_path", "norm_stats.precomputed_norm_path"}
    fixed = []
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        if (
            sep
            and key in _PATH_KEYS
            and val
            and val != "null"
            and not val.startswith("/")
        ):
            val = f"{CFG.output_mount_path}/{val}"
            arg = f"{key}={val}"
        fixed.append(arg)
    return tuple(fixed)


def _download_run_artifacts(output_rel_path: str) -> None:
    """Download training artifacts from the output volume to ./modal-outputs/ locally."""
    if not output_rel_path:
        return
    local_dest = REPO_ROOT / "modal-outputs" / output_rel_path
    local_dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts to {local_dest} ...")
    modal_environment = os.environ.get("MODAL_ENVIRONMENT", "main")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            "--env",
            modal_environment,
            "egoverse-training-outputs",
            output_rel_path,
            str(local_dest),
        ],
        cwd=REPO_ROOT,
    )
    if result.returncode == 0:
        print(f"Artifacts saved to: {local_dest.resolve()}")
    else:
        print(
            f"Download failed — pull manually:\n"
            f"  modal volume get --env {modal_environment} "
            f'egoverse-training-outputs "{output_rel_path}" "{local_dest}"'
        )


# ---------------------------------------------------------------------------
# Container helpers (run inside the Modal container)
# ---------------------------------------------------------------------------


def _ssh_to_https(url: str) -> str:
    """Convert git@github.com:org/repo.git → https://github.com/org/repo.git"""
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:") :]
        return f"https://github.com/{path}"
    return url


def _prepare_repo(git_remote: str, git_commit: str) -> None:
    """Clone (or update) the repo and check out the exact commit."""
    clone_url = _ssh_to_https(git_remote)
    repo_dir = Path(CFG.remote_repo_dir)

    if (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "--all", "--tags"],
            check=True,
        )
    elif repo_dir.exists():
        # Directory exists but has no .git — can happen when a volume is mounted
        # at a subdirectory, causing Modal to create the parent before the clone.
        subprocess.run(["git", "init", CFG.remote_repo_dir], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                CFG.remote_repo_dir,
                "remote",
                "add",
                "origin",
                clone_url,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "origin", "--tags"],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "clone", clone_url, CFG.remote_repo_dir],
            check=True,
        )

    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "checkout", git_commit],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            CFG.remote_repo_dir,
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        check=True,
    )
    # Create a uv venv that inherits all packages pre-installed in the image.
    # --system-site-packages means no reinstall of torch/transformers/etc.
    subprocess.run(
        ["uv", "venv", ".venv", "--system-site-packages"],
        cwd=CFG.remote_repo_dir,
        check=True,
    )
    # Register the egomimic package itself inside the venv (editable, no dep reinstall).
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            CFG.venv_python,
            "-e",
            ".",
            "--no-deps",
            "-q",
        ],
        cwd=CFG.remote_repo_dir,
        check=True,
    )


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------


@app.function(
    gpu=CFG.gpu,
    cpu=CFG.cpu,
    memory=CFG.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes=_volumes,
)
def run_hydra_train(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> str:
    """Clone the repo at *git_commit* and run trainHydra.py with *hydra_args*.

    Returns the path relative to the output volume root where artifacts were
    written (e.g. 'my_run/test_2026-05-11_20-35-51'), or '' if no output volume.
    """
    import glob as _glob
    import json as _json

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)

    hydra_args = _resolve_volume_paths(hydra_args)
    cmd = _build_train_cmd(hydra_args)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env["MODAL_IS_REMOTE"] = "1"
    _container_app = modal.App._get_container_app()
    _resolved_app = (
        (
            _container_app.name
            if _container_app is not None and _container_app.name
            else ""
        )
        or os.environ.get("MODAL_APP_NAME", "").strip()
        or _MODAL_APP_DEFAULT
    )
    env["MODAL_APP_NAME"] = _resolved_app
    # Tell EpisodeResolver to read data from the mounted volume
    env["EGOVERSE_MODAL_DATA_DIR"] = CFG.zarr_volume_mount_path
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    # Expose context so ModalAutoRestartCallback can spawn a continuation job
    env["MODAL_TIMEOUT_SECONDS"] = str(CFG.timeout_seconds)
    env["MODAL_START_TIME"] = str(time.time())
    env["MODAL_HYDRA_ARGS"] = _json.dumps(list(hydra_args))
    env["MODAL_GIT_REMOTE"] = git_remote
    env["MODAL_GIT_COMMIT"] = git_commit

    print(f"Running: {shlex.join(cmd)}")
    process = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)

    zarr_volume.commit()

    all_run_dirs = sorted(
        _glob.glob(f"{CFG.output_mount_path}/*/*"),
        key=os.path.getmtime,
    )
    output_rel_path = (
        os.path.relpath(all_run_dirs[-1], CFG.output_mount_path) if all_run_dirs else ""
    )

    training_outputs_volume.commit()

    if process.returncode != 0:
        raise RuntimeError(
            f"Training failed (exit {process.returncode}): {shlex.join(cmd)}"
        )

    return output_rel_path


# ---------------------------------------------------------------------------
# Container health-check function
# ---------------------------------------------------------------------------


@app.function(
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes=_volumes,
    timeout=120,
)
def _health_check() -> dict:
    """Verify secrets and volume mounts from inside the container."""
    results: dict = {}

    results["WANDB_API_KEY"] = (
        "OK"
        if os.environ.get("WANDB_API_KEY")
        else "MISSING (optional — needed for W&B)"
    )

    for label, path in [
        ("egoverse-data", CFG.zarr_volume_mount_path),
        ("egoverse-training-outputs", CFG.output_mount_path),
    ]:
        probe = f"{path}/.modal_health_probe"
        try:
            os.makedirs(path, exist_ok=True)
            open(probe, "w").close()
            os.remove(probe)
            results[label] = f"OK — mounted at {path}"
        except Exception as e:
            results[label] = f"ERROR: {e}"

    return results


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def verify() -> None:
    """Boot the container and verify secrets and volumes are accessible.

    Example:
        modal run egomimic/modal/run.py::verify
    """
    print("Running container health check...")
    results = _health_check.remote()
    all_ok = True
    for k, v in results.items():
        symbol = "✓" if v.startswith("OK") else ("~" if "optional" in v else "✗")
        print(f"  {symbol}  {k}: {v}")
        if v.startswith("ERROR"):
            all_ok = False
    print()
    if all_ok:
        print("All checks passed — Modal setup is ready.")
    else:
        raise SystemExit("One or more checks failed.")


@app.local_entrypoint()
def submit(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a Modal job and return immediately.

    Example:
        modal run egomimic/modal/run.py::submit -- \\
            data=mecka_all_zarr trainer=ddp_modal model=hpt_bc_flow_mecka \\
            name=my_run description=test

    After completion, download artifacts manually:
        modal volume get --env main egoverse-training-outputs <run-path> ./modal-outputs/
    """
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Submitting commit {git_commit[:12]} from {git_remote}")
    handle = run_hydra_train.spawn(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    modal_environment = os.environ.get("MODAL_ENVIRONMENT", "main")
    print(f"Submitted Modal job: {handle.object_id}")
    _app_url_name = (
        os.environ.get("MODAL_APP_NAME", _MODAL_APP_DEFAULT).strip()
        or _MODAL_APP_DEFAULT
    )
    print(f"Monitor at: https://modal.com/apps/{_app_url_name}")
    print(
        "After completion, download artifacts:\n"
        f"  modal volume get --env {modal_environment} "
        "egoverse-training-outputs <run-path> ./modal-outputs/"
    )


@app.local_entrypoint()
def run(*hydra_args: str) -> None:
    """Blocking run: streams logs and downloads artifacts to ./modal-outputs/ on completion.

    Example:
        modal run egomimic/modal/run.py::run -- \\
            data=mecka_all_zarr trainer=ddp_modal model=hpt_bc_flow_mecka \\
            name=my_run description=test
    """
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Running commit {git_commit[:12]} from {git_remote}")
    output_rel_path = run_hydra_train.remote(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Remote run completed. Output path in volume: {output_rel_path}")
    if output_rel_path:
        _download_run_artifacts(output_rel_path)
