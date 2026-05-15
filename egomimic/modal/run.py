"""Modal training entrypoints for EgoVerse.

Usage
-----
Via trainHydra.py (recommended):
    python egomimic/trainHydra.py data=mecka_all_zarr trainer=ddp_modal \\
        logger=wandb model=hpt_bc_flow_mecka name=<run> description=<desc> \\
        [+modal_gpu=H100] [+modal_cpu=32] [+modal_memory_gb=128]

Direct (fire-and-forget):
    modal run --env robotics egomimic/modal/test_run.py::submit -- \\
        data=mecka_all_zarr trainer=ddp_modal logger=wandb model=hpt_bc_flow_mecka

Direct (blocking — streams logs, downloads artifacts when done):
    modal run --env robotics egomimic/modal/test_run.py::run -- \\
        data=mecka_all_zarr trainer=ddp_modal logger=wandb model=hpt_bc_flow_mecka

Volume note
-----------
The zarr dataset volume (egoverse-zarr-data) is mounted at /mnt/zarr-data.
Use LocalEpisodeResolver with folder_path=/mnt/zarr-data (see mecka_all_zarr.yaml).

Training outputs (checkpoints, logs, norm stats, videos) are written to
/root/EgoVerse/logs inside the container, which is backed by the Modal volume
egoverse-training-outputs. After a blocking `run`, artifacts are automatically
downloaded to ./modal-outputs/<name>/<desc>_<timestamp>/ on your local machine.
For fire-and-forget `submit` jobs, pull manually:
    modal volume get --env robotics egoverse-training-outputs <run-path> ./modal-outputs/

Notes
-----
- This file is intentionally self-contained (no egomimic imports at module level)
  because Modal mounts it as /root/test_run.py before the repo is cloned.
- Modal runs the *committed* git state. Commit before submitting a real run.
- Secrets must exist in the Modal dashboard (robotics env):
    egoverse-r2      → R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, R2_BUCKET
    egoverse-mongodb → MONGODB_URI
- WANDB_API_KEY is per-user: add it to ~/.egoverse_env on your local machine.
- GPU/CPU/memory are set via MODAL_GPU / MODAL_CPU / MODAL_MEMORY_GB env vars
  (trainHydra.py sets these from +modal_gpu= / +modal_cpu= / +modal_memory_gb= overrides).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Route CLI to robotics env by default
# ---------------------------------------------------------------------------
os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Inline config (duplicated from modal_config.py so this file is standalone)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class _Config:
    remote_repo_dir: str = "/root/EgoVerse"
    python_bin: str = "python3"

    @property
    def train_script(self) -> str:
        return f"{self.remote_repo_dir}/egomimic/trainHydra.py"

    zarr_volume_name: str = field(
        default_factory=lambda: os.environ.get("MODAL_ZARR_VOLUME", "mecka_data_v2")
    )
    volume_mount_path: str = "/mnt/zarr-data"
    # Training outputs (checkpoints, logs, norm stats, videos) are persisted here
    output_mount_path: str = "/root/EgoVerse/logs"

    # Overridable via env vars (set by trainHydra.py from +modal_gpu= etc.)
    gpu: str = field(default_factory=lambda: os.environ.get("MODAL_GPU", "A100"))
    cpu: float = field(default_factory=lambda: float(os.environ.get("MODAL_CPU", "12.0")))
    memory_mb: int = field(
        default_factory=lambda: (
            int(float(os.environ.get("MODAL_MEMORY_GB")) * 1024)
            if os.environ.get("MODAL_MEMORY_GB")
            else int(os.environ.get("MODAL_MEMORY_MB", "65536"))
        )
    )
    timeout_seconds: int = 86400  # 24 h (Modal max)

    secret_names: list[str] = field(
        default_factory=lambda: ["egoverse-r2", "egoverse-mongodb", "egoverse-db", "egoverse-sql"]
    )


CFG = _Config()

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    .apt_install("git")
    .pip_install(
        # Core training stack
        "lightning",
        "hydra-core",
        "omegaconf",
        "wandb",
        # Data / storage
        "boto3",
        "cloudpathlib",
        "zarr==3.1.5",
        "pyarrow",
        "simplejpeg",
        "h5py",
        "av==12.0.0",
        "mediapy",
        "datasets==4.0.0",
        # Model / ML
        "transformers==4.57.3",
        "timm",
        "einops",
        "positional-encodings[pytorch]",
        "pytorch-kinematics",
        "arm-pytorch-utilities",
        "geomloss",
        "tslearn",
        "scipy",
        # Hydra plugins
        "hydra-submitit-launcher==1.2.0",
        "submitit",
        # Vision
        "opencv-python-headless",
        "projectaria-tools",
        "pyquaternion",
        # Database / infra
        "sqlalchemy",
        "psycopg[binary]",
        "pandas",
        # Utilities
        "rich",
        "tabulate",
        "prettytable",
        "packaging",
        "overrides",
        "typing_extensions",
        "pyyaml",
        "matplotlib",
        "termcolor",
        "tqdm",
        "filelock",
        "imageio",
        "imageio-ffmpeg",
        "safetensors",
        "huggingface-hub",
        "scaleapi",
        "openai",
        "pyzmq",
        "datasets==4.0.0",
        "torchvision==0.21.0",
        "s5cmd",
    )
)

zarr_volume = modal.Volume.from_name(CFG.zarr_volume_name)
training_outputs_volume = modal.Volume.from_name(
    "egoverse-training-outputs", create_if_missing=True
)
app = modal.App("egomimic-training", image=image)

# ---------------------------------------------------------------------------
# Local helpers (execute on the submitting machine)
# ---------------------------------------------------------------------------


def _local_wandb_key() -> str:
    """Read WANDB_API_KEY from the local environment or ~/.egoverse_env."""
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
            "Warning: WANDB_API_KEY not set locally — W&B logging will be disabled. "
            "Add it to ~/.egoverse_env to enable."
        )
    return key


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def _resolve_git_state() -> tuple[str, str, bool]:
    """Return (remote_url, commit_sha, is_dirty).

    Raises SystemExit if HEAD has not been pushed to the remote — the container
    clones from GitHub so unpushed commits will cause a checkout failure there.
    """
    git_remote = _git_output(["git", "config", "--get", "remote.origin.url"])
    git_commit = _git_output(["git", "rev-parse", "HEAD"])
    is_dirty = bool(_git_output(["git", "status", "--porcelain"]))

    # Verify the commit exists on the remote before submitting
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin"],
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
        result = subprocess.run(
            ["git", "branch", "-r", "--contains", git_commit],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if not result.stdout.strip():
            raise SystemExit(
                f"ERROR: commit {git_commit[:12]} has not been pushed to {git_remote}.\n"
                "Push your branch first, then re-run.\n"
                "  git push --set-upstream origin <branch>"
            )
    except subprocess.CalledProcessError:
        pass  # fetch failed (no network / auth) — let the container surface the error

    return git_remote, git_commit, is_dirty


def _build_train_cmd(hydra_args: tuple[str, ...]) -> list[str]:
    return [CFG.python_bin, CFG.train_script, *hydra_args]


def _resolve_volume_paths(hydra_args: tuple[str, ...]) -> tuple[str, ...]:
    """Prepend the output volume mount path to relative values for path-bearing keys.

    Allows callers to pass  ckpt_path=mecka_modal/run/checkpoints/last.ckpt
    instead of the full container path  ckpt_path=/root/EgoVerse/logs/...
    """
    _PATH_KEYS = {"ckpt_path", "norm_stats.precomputed_norm_path"}
    fixed = []
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        if sep and key in _PATH_KEYS and val and val != "null" and not val.startswith("/"):
            val = f"{CFG.output_mount_path}/{val}"
            arg = f"{key}={val}"
        fixed.append(arg)
    return tuple(fixed)


def _download_run_artifacts(output_rel_path: str) -> None:
    """Download artifacts from the training outputs volume to ./modal-outputs/ locally."""
    local_dest = REPO_ROOT / "modal-outputs" / output_rel_path
    local_dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts to {local_dest} ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "modal", "volume", "get",
            "--env", "robotics",
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
            f"  modal volume get --env robotics egoverse-training-outputs "
            f'"{output_rel_path}" "{local_dest}"'
        )


# ---------------------------------------------------------------------------
# Container helpers (execute inside the Modal container)
# ---------------------------------------------------------------------------


def _ssh_to_https(url: str) -> str:
    """Convert git@github.com:org/repo.git → https://github.com/org/repo.git"""
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:"):]
        return f"https://github.com/{path}"
    return url


def _prepare_repo(git_remote: str, git_commit: str) -> None:
    """Clone (or update) the repo and check out the exact commit."""
    # The container has no SSH keys — always use HTTPS for cloning
    clone_url = _ssh_to_https(git_remote)
    repo_dir = Path(CFG.remote_repo_dir)

    if (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "--all", "--tags"],
            check=True,
        )
    elif repo_dir.exists():
        # Directory already exists but has no .git — happens when a volume is
        # mounted at a subdirectory (e.g. /root/EgoVerse/logs), which causes
        # Modal to create the parent before we clone. Use init+fetch instead.
        subprocess.run(["git", "init", CFG.remote_repo_dir], check=True)
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "remote", "add", "origin", clone_url],
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
        ["git", "-C", CFG.remote_repo_dir, "submodule", "update", "--init", "--recursive"],
        check=True,
    )

    # --no-deps: packages come from the image; just registers egomimic in sys.path
    subprocess.run(
        [CFG.python_bin, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
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
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_hydra_train(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> str:
    """Clone the repo at *git_commit* and run trainHydra.py with *hydra_args*.

    Returns the path relative to the output volume root where artifacts were written
    (e.g. 'mecka_modal/test_2026-05-11_20-35-51'), or empty string on failure.
    """
    import glob

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)

    # Reload the FUSE mount right before training starts — _prepare_repo can take
    # several minutes and the FUSE connection goes stale, causing EIO on listdir.
    zarr_volume.reload()

    hydra_args = _resolve_volume_paths(hydra_args)
    cmd = _build_train_cmd(hydra_args)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    # Expose container context so ModalAutoRestartCallback can spawn a continuation job
    import json as _json
    import time as _time
    env["MODAL_TIMEOUT_SECONDS"] = str(CFG.timeout_seconds)
    env["MODAL_START_TIME"] = str(_time.time())
    env["MODAL_HYDRA_ARGS"] = _json.dumps(list(hydra_args))
    env["MODAL_GIT_REMOTE"] = git_remote
    env["MODAL_GIT_COMMIT"] = git_commit

    print(f"Running: {shlex.join(cmd)}")

    process = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)

    zarr_volume.commit()

    # Find the most recently modified run directory and persist it to the volume
    all_run_dirs = sorted(
        glob.glob(f"{CFG.output_mount_path}/*/*"),
        key=os.path.getmtime,
    )
    output_rel_path = (
        os.path.relpath(all_run_dirs[-1], CFG.output_mount_path)
        if all_run_dirs
        else ""
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
    volumes={CFG.volume_mount_path: zarr_volume},
    timeout=120,
)
def _health_check() -> dict:
    """Verify secrets, DB, R2 credentials, and volume mount from inside the container."""
    results = {}

    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL"):
        results[key] = "OK" if os.environ.get(key) else "MISSING"

    results["MONGODB_URI"] = "OK" if os.environ.get("MONGODB_URI") else "MISSING"

    probe = f"{CFG.volume_mount_path}/.modal_health_probe"
    try:
        open(probe, "w").close()
        os.remove(probe)
        results["volume"] = f"OK — mounted at {CFG.volume_mount_path}"
    except Exception as e:
        results["volume"] = f"ERROR: {e}"

    r = subprocess.run(["s5cmd", "version"], capture_output=True, text=True)
    results["s5cmd"] = f"OK — {r.stdout.strip()}" if r.returncode == 0 else "MISSING"

    return results


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def verify() -> None:
    """Boot the container and verify all secrets, volume, and s5cmd."""
    print("Running container health check...")
    results = _health_check.remote()
    all_ok = True
    for k, v in results.items():
        symbol = "✓" if v.startswith("OK") else "✗"
        print(f"  {symbol}  {k}: {v}")
        if not v.startswith("OK"):
            all_ok = False
    print()
    if all_ok:
        print("All checks passed — Modal setup is ready.")
    else:
        raise SystemExit("One or more checks failed.")


@app.local_entrypoint()
def submit(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a Modal job and return immediately."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Submitting commit {git_commit[:12]} from {git_remote}")
    handle = run_hydra_train.spawn(tuple(hydra_args), git_remote, git_commit, _local_wandb_key())
    print(f"Submitted Modal job: {handle.object_id}")
    print("Monitor at: https://modal.com/apps/egomimic-training")
    print(
        "After completion, download artifacts:\n"
        "  modal volume get --env robotics egoverse-training-outputs <run-path> ./modal-outputs/"
    )


@app.local_entrypoint()
def run(*hydra_args: str) -> None:
    """Blocking run: streams logs and downloads artifacts to ./modal-outputs/ when complete."""
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
