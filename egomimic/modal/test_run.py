"""Modal training entrypoints for EgoVerse.

Usage
-----
Async (fire-and-forget):
    modal run --env robotics egomimic/modal/test_run.py -- \\
        --config-name=train_zarr_cartesian_pi \\
        trainer=ddp_modal \\
        "data.train_datasets.mecka_bimanual.resolver.folder_path=/mnt/zarr-data/processed_zarr" \\
        "data.valid_datasets.mecka_bimanual.resolver.folder_path=/mnt/zarr-data/processed_zarr"

Synchronous (blocks until done):
    modal run --env robotics egomimic/modal/test_run.py::run -- \\
        --config-name=train_zarr_cartesian_pi \\
        trainer=ddp_modal \\
        "data.train_datasets.mecka_bimanual.resolver.folder_path=/mnt/zarr-data/processed_zarr" \\
        "data.valid_datasets.mecka_bimanual.resolver.folder_path=/mnt/zarr-data/processed_zarr"

The MODAL_ENVIRONMENT variable is set to "robotics" automatically by
modal_config.py, so the --env flag is optional but shown for clarity.

Any extra arguments after `--` are forwarded verbatim to Hydra / trainHydra.py.

Volume note
-----------
The zarr dataset volume (egoverse-zarr-data) is mounted at /mnt/zarr-data.
Processed zarr stores live at /mnt/zarr-data/processed_zarr/<embodiment>.
Training uses LocalEpisodeResolver which reads directly from that path —
no SQL lookup or R2 download happens at training time.

Notes
-----
- Modal runs the *committed* git state.  Uncommitted changes are not sent to
  the container.  Commit (or stash) before submitting a real run.
- Secrets must exist in the Modal dashboard (robotics env) before first use:
    egoverse-r2      → R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, R2_BUCKET
    egoverse-mongodb → MONGODB_URI
    egoverse-wandb   → WANDB_API_KEY  (add when ready)
- To change GPU type or timeout, edit CFG in modal_config.py.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

from egomimic.modal.modal_config import CFG, REPO_ROOT, app, zarr_volume

# ---------------------------------------------------------------------------
# Local helpers (execute on the submitting machine)
# ---------------------------------------------------------------------------


def _local_wandb_key() -> str:
    """Read WANDB_API_KEY from the local environment or ~/.egoverse_env."""
    from egomimic.utils.aws.aws_data_utils import load_env
    load_env()
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
    """Return (remote_url, commit_sha, is_dirty)."""
    git_remote = _git_output(["git", "config", "--get", "remote.origin.url"])
    git_commit = _git_output(["git", "rev-parse", "HEAD"])
    is_dirty = bool(_git_output(["git", "status", "--porcelain"]))
    return git_remote, git_commit, is_dirty


def _build_train_cmd(hydra_args: tuple[str, ...]) -> list[str]:
    return [CFG.python_bin, CFG.train_script, *hydra_args]


# ---------------------------------------------------------------------------
# Container helpers (execute inside the Modal container)
# ---------------------------------------------------------------------------


def _prepare_repo(git_remote: str, git_commit: str) -> None:
    """Clone (or update) the repo and check out the exact commit."""
    repo_dir = Path(CFG.remote_repo_dir)

    if (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "--all", "--tags"],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "clone", git_remote, CFG.remote_repo_dir],
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

    # Install the project itself so `import egomimic` resolves correctly.
    # --no-deps avoids re-downloading packages already baked into the image.
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
    # zarr datasets persist across runs; episodes already on the volume are
    # skipped by S3EpisodeResolver so they are only downloaded once.
    volumes={CFG.volume_mount_path: zarr_volume},
)
def run_hydra_train(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> int:
    """Clone the repo at *git_commit* and run trainHydra.py with *hydra_args*."""
    _prepare_repo(git_remote=git_remote, git_commit=git_commit)

    cmd = _build_train_cmd(hydra_args)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    print(f"Running: {shlex.join(cmd)}")

    # Stream stdout/stderr directly so logs appear in the Modal dashboard in
    # real time (no capture — avoids buffering multi-hour training output).
    process = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)

    # Commit any new zarr episodes written to the volume so they are visible
    # to future runs even if this container is reused.
    zarr_volume.commit()

    if process.returncode != 0:
        raise RuntimeError(
            f"Training failed (exit {process.returncode}): {shlex.join(cmd)}"
        )

    return process.returncode


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
    import os
    import subprocess
    results = {}

    # R2 credentials
    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL"):
        results[key] = "OK" if os.environ.get(key) else "MISSING"

    # MongoDB URI present
    results["MONGODB_URI"] = "OK" if os.environ.get("MONGODB_URI") else "MISSING"

    # Volume mounted and writable
    probe = f"{CFG.volume_mount_path}/.modal_health_probe"
    try:
        open(probe, "w").close()
        os.remove(probe)
        results["volume"] = f"OK — mounted at {CFG.volume_mount_path}"
    except Exception as e:
        results["volume"] = f"ERROR: {e}"

    # s5cmd available
    r = subprocess.run(["s5cmd", "version"], capture_output=True, text=True)
    results["s5cmd"] = f"OK — {r.stdout.strip()}" if r.returncode == 0 else "MISSING"

    return results


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def verify() -> None:
    """Boot the container and verify all secrets, DB, volume, and s5cmd."""
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
    """Fire-and-forget: spawn a Modal job and return immediately.

    Example:
        modal run --env robotics egomimic/modal/test_run.py -- \\
            --config-name=train_zarr_cartesian_pi trainer=ddp_modal \\
            "data.train_datasets.mecka_bimanual.resolver.folder_path=/mnt/zarr-data/processed_zarr" \\
            "data.valid_datasets.mecka_bimanual.resolver.folder_path=/mnt/zarr-data/processed_zarr"
    """
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


@app.local_entrypoint()
def run(*hydra_args: str) -> None:
    """Synchronous run: block until the remote job completes.

    Example:
        modal run --env robotics egomimic/modal/test_run.py::run -- \\
            --config-name=train_zarr_cartesian_pi trainer=ddp_modal \\
            "data.train_datasets.mecka_bimanual.resolver.folder_path=/mnt/zarr-data/processed_zarr" \\
            "data.valid_datasets.mecka_bimanual.resolver.folder_path=/mnt/zarr-data/processed_zarr"
    """
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Running commit {git_commit[:12]} from {git_remote}")
    exit_code = run_hydra_train.remote(tuple(hydra_args), git_remote, git_commit, _local_wandb_key())
    print(f"Remote run completed with exit code: {exit_code}")
