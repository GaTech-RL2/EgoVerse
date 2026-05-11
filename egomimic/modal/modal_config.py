from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import modal

# Absolute path to the repository root on the LOCAL machine.
# Used by local helpers to read git state.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Route all modal CLI commands to the robotics environment by default.
# Override with MODAL_ENVIRONMENT=<env> or modal run --env <env>.
os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Starts from the official PyTorch image (CUDA 12.4 + cuDNN 9) so the GPU
# driver stack is already set up.  Only the packages not included in the base
# image need to be added here; the project itself is installed at runtime
# after cloning the repo (see _prepare_repo in test_run.py).
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    .apt_install("git")
    .pip_install(
        "lightning",
        "hydra-core",
        "omegaconf",
        "wandb",
        "boto3",
        "cloudpathlib",
        "zarr==3.1.5",
        "tabulate",
        "transformers==4.57.3",
        "timm",
        "einops",
        "pandas",
        "sqlalchemy",
        "psycopg[binary]",
        "pyarrow",
        "simplejpeg",
        "rich",
        "packaging",
        "h5py",
        "overrides",
        "typing_extensions",
        "pyyaml",
        "prettytable",
        "positional-encodings[pytorch]",
        # s5cmd: fast parallel S3/R2 downloader used by S3EpisodeResolver
        "s5cmd",
    )
)

# ---------------------------------------------------------------------------
# Persistent volume for zarr datasets
# ---------------------------------------------------------------------------
# Episodes downloaded from R2 are cached here across runs.
# S3EpisodeResolver skips episodes already present, so each episode is only
# downloaded once regardless of how many training runs reference it.
# Volume is named "egoverse-zarr-data" in the robotics environment.
# ---------------------------------------------------------------------------
zarr_volume = modal.Volume.from_name("egoverse-zarr-data")

app = modal.App("egomimic-training", image=image)


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
@dataclass
class _Config:
    # --- Paths inside the container ---
    remote_repo_dir: str = "/root/EgoVerse"
    python_bin: str = "python3"

    @property
    def train_script(self) -> str:
        return f"{self.remote_repo_dir}/egomimic/trainHydra.py"

    # --- Volume ---
    # Mount path for the zarr dataset volume inside the container.
    # Processed zarr stores live at /mnt/zarr-data/processed_zarr.
    # Pass this as folder_path when using LocalEpisodeResolver, e.g.:
    #   data.train_datasets.<name>.resolver.folder_path=/mnt/zarr-data/processed_zarr
    volume_mount_path: str = "/mnt/zarr-data"

    # --- Modal compute spec ---
    # Overridable via env vars at submission time (set by trainHydra.py):
    #   MODAL_GPU        e.g. "A100", "H100", "A10G", "A100:4" (4 GPUs)
    #   MODAL_CPU        e.g. "16"
    #   MODAL_MEMORY_MB  e.g. "131072"  (or set MODAL_MEMORY_GB for convenience)
    gpu: str = field(default_factory=lambda: os.environ.get("MODAL_GPU", "A100"))
    cpu: float = field(default_factory=lambda: float(os.environ.get("MODAL_CPU", "12.0")))
    memory_mb: int = field(
        default_factory=lambda: (
            int(float(os.environ.get("MODAL_MEMORY_GB")) * 1024)
            if os.environ.get("MODAL_MEMORY_GB")
            else int(os.environ.get("MODAL_MEMORY_MB", "65536"))
        )
    )
    timeout_seconds: int = 86400   # 24 h (Modal max)

    # --- Modal secrets (shared) ---
    # egoverse-r2      → R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL, R2_BUCKET
    # egoverse-mongodb → MONGODB_URI
    # WANDB_API_KEY is intentionally excluded — each user passes their own key
    # via ~/.egoverse_env at submission time so runs are attributed correctly.
    secret_names: list[str] = field(
        default_factory=lambda: ["egoverse-r2", "egoverse-mongodb"]
    )


CFG = _Config()
