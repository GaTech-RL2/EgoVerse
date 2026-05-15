"""Offline normalization-statistics computation on Modal CPUs.

Intentionally self-contained (no egomimic imports at module level) because
Modal mounts this file as /root/offline_norm_stats.py before the repo is cloned.

Computes norm stats for a given data config and writes:
  <training-outputs-volume>/precomputed_norm_stats/<data_config>/norm_stats.json

Usage:
    modal run --env robotics egomimic/modal/offline_norm_stats.py \\
        -- mecka_all_zarr [--cpu 32] [--memory_gb 128] [--num_workers 30] [--sample_frac 1.0] [--batch_size 512]

In training, point at the result with:
    norm_stats.precomputed_norm_path=precomputed_norm_stats/<data_config>
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Parse --cpu / --memory_gb from sys.argv before the decorator below runs
# ---------------------------------------------------------------------------


def _extract_arg(argv: list[str], flag: str, default: str) -> str:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return default


_argv = sys.argv[1:]
_CPU = float(_extract_arg(_argv, "--cpu", "16"))
_MEMORY_MB = int(float(_extract_arg(_argv, "--memory_gb", "64")) * 1024)

# ---------------------------------------------------------------------------
# Inline config (mirrors run.py — no egomimic import needed)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class _Config:
    remote_repo_dir: str = "/root/EgoVerse"
    python_bin: str = "python3"
    zarr_volume_name: str = field(
        default_factory=lambda: os.environ.get("MODAL_ZARR_VOLUME", "mecka_data_v2")
    )
    volume_mount_path: str = "/mnt/zarr-data"
    output_mount_path: str = "/root/EgoVerse/logs"
    secret_names: list[str] = field(
        default_factory=lambda: ["egoverse-r2", "egoverse-mongodb", "egoverse-db", "egoverse-sql"]
    )


CFG = _Config()

# ---------------------------------------------------------------------------
# Image and volumes (same as run.py)
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        add_python="3.10",
    )
    .apt_install("git", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .pip_install(
        "lightning",
        "hydra-core",
        "omegaconf",
        "wandb",
        "boto3",
        "cloudpathlib",
        "zarr==3.1.5",
        "pyarrow",
        "simplejpeg",
        "h5py",
        "av==12.0.0",
        "mediapy",
        "datasets==4.0.0",
        "transformers==4.57.3",
        "timm",
        "einops",
        "positional-encodings[pytorch]",
        "pytorch-kinematics",
        "arm-pytorch-utilities",
        "geomloss",
        "tslearn",
        "scipy",
        "hydra-submitit-launcher==1.2.0",
        "submitit",
        "opencv-python-headless",
        "projectaria-tools",
        "pyquaternion",
        "sqlalchemy",
        "psycopg[binary]",
        "pandas",
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
        "torchvision==0.21.0",
        "s5cmd",
    )
)

zarr_volume = modal.Volume.from_name(CFG.zarr_volume_name)
training_outputs_volume = modal.Volume.from_name(
    "egoverse-training-outputs", create_if_missing=True
)
app = modal.App("egomimic-norm-stats", image=image)

_TIMEOUT = 3 * 3600
_NORM_SUBDIR = "precomputed_norm_stats"

# ---------------------------------------------------------------------------
# Container helpers (inlined from run.py — no egomimic import)
# ---------------------------------------------------------------------------


def _ssh_to_https(url: str) -> str:
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:"):]
        return f"https://github.com/{path}"
    return url


def _prepare_repo(git_remote: str, git_commit: str) -> None:
    clone_url = _ssh_to_https(git_remote)
    repo_dir = Path(CFG.remote_repo_dir)

    if (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "--all", "--tags"], check=True
        )
    elif repo_dir.exists():
        subprocess.run(["git", "init", CFG.remote_repo_dir], check=True)
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "remote", "add", "origin", clone_url],
            check=True,
        )
        subprocess.run(
            ["git", "-C", CFG.remote_repo_dir, "fetch", "origin", "--tags"], check=True
        )
    else:
        subprocess.run(["git", "clone", clone_url, CFG.remote_repo_dir], check=True)

    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "checkout", git_commit], check=True
    )
    subprocess.run(
        ["git", "-C", CFG.remote_repo_dir, "submodule", "update", "--init", "--recursive"],
        check=True,
    )
    # Register egomimic as an editable install without touching deps
    # (all deps come from the Modal image). uv pip --system writes to the
    # system Python that the container process is already running.
    subprocess.run(
        ["uv", "pip", "install", "--system", "-e", ".", "--no-deps"],
        cwd=CFG.remote_repo_dir,
        check=True,
    )


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------


@app.function(
    cpu=_CPU,
    memory=_MEMORY_MB,
    timeout=_TIMEOUT,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_norm_stats(
    data_config: str,
    num_workers: int,
    sample_frac: float,
    batch_size: int,
    git_remote: str,
    git_commit: str,
) -> str:
    """Clone the repo and compute norm stats for *data_config*.

    Returns the container path of the written norm_stats.json.
    """
    import copy
    import json
    import sys

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)
    zarr_volume.reload()

    # uv pip --system writes to the system Python's site-packages, but the
    # current process won't pick up the new .pth until it restarts. Add the
    # repo root directly so egomimic is importable in-process.
    sys.path.insert(0, CFG.remote_repo_dir)

    import hydra
    import numpy as np
    from omegaconf import OmegaConf, open_dict

    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id
    from egomimic.rldb.zarr.utils import DataSchematic

    load_env()
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # Load just the data config YAML directly — no full Hydra compose needed,
    # which avoids pulling in trainer / model / callback defaults.
    data_cfg_path = (
        Path(CFG.remote_repo_dir) / "egomimic" / "hydra_configs" / "data" / f"{data_config}.yaml"
    )
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_cfg_path}")
    data_cfg = OmegaConf.load(str(data_cfg_path))

    # ---- Pass 1: build combined schematic_dict from all train datasets ----
    # We instantiate the key_map for each dataset, strip camera/annotation
    # keys, and collect the {key_name: {key_type, zarr_key}} entries needed
    # to construct DataSchematic.
    schematic_dict: dict = {}
    prepared_ds_cfgs: dict = {}

    for dataset_name, ds_cfg_raw in data_cfg.train_datasets.items():
        print(f"[NormStats] Building schematic for <{dataset_name}>")
        ds_cfg = copy.deepcopy(ds_cfg_raw)
        with open_dict(ds_cfg):
            if OmegaConf.select(ds_cfg, "resolver.debug", default=None) is not None:
                ds_cfg.resolver.debug = False

        # Instantiate key_map to get the actual {key_name: {...}} dict
        key_map = hydra.utils.instantiate(ds_cfg.resolver.key_map)
        norm_key_map = {
            k: v for k, v in key_map.items()
            if isinstance(v, dict) and v.get("key_type") not in ("camera_keys", "annotation_keys")
        }

        schematic_dict[dataset_name] = {
            key_name: {"key_type": key_info["key_type"], "zarr_key": key_info["zarr_key"]}
            for key_name, key_info in norm_key_map.items()
            if isinstance(key_info, dict) and "zarr_key" in key_info
        }
        prepared_ds_cfgs[dataset_name] = ds_cfg

    data_schematic = DataSchematic(schematic_dict=schematic_dict, norm_mode="quantile")

    # ---- Pass 2: infer shapes and norm stats per dataset ----
    for dataset_name, ds_cfg in prepared_ds_cfgs.items():
        print(f"[NormStats] Instantiating dataset <{dataset_name}>")
        sample_dataset = hydra.utils.instantiate(ds_cfg)
        data_schematic.infer_shapes_from_batch(sample_dataset[0])

        # Build the norm dataset: train split only, camera/annotation keys stripped
        norm_cfg = copy.deepcopy(ds_cfg)
        km = OmegaConf.to_container(norm_cfg.resolver.key_map, resolve=False)
        km["norm_mode"] = True
        norm_cfg.resolver.key_map = km
        with open_dict(norm_cfg):
            norm_cfg.mode = "train"
        norm_dataset = hydra.utils.instantiate(norm_cfg)

        emb_id = str(get_embodiment_id(dataset_name))
        ckpt_dir = out_path.parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        def _make_checkpoint_fn(ckpt_dir=ckpt_dir, emb_id=emb_id):
            def _fn(collected, pct):
                partial: dict = {}
                for k, arrays in collected.items():
                    if not arrays:
                        continue
                    X = np.concatenate(arrays, axis=0)
                    partial[k] = {
                        stat_name: np.asarray(arr).tolist()
                        for stat_name, arr in data_schematic._compute_stats_for_array(X).items()
                    }
                ckpt_path = ckpt_dir / f"norm_stats_checkpoint_{pct}pct.json"
                with open(ckpt_path, "w") as f:
                    json.dump({"stats": {emb_id: partial}, "checkpoint_pct": pct}, f, indent=4)
                training_outputs_volume.commit()
                print(f"[NormStats] Checkpoint {pct}% → {ckpt_path}")
            return _fn

        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=sample_frac,
            num_workers=num_workers,
            batch_size=batch_size,
            checkpoint_fn=_make_checkpoint_fn(),
        )

        emb_id = get_embodiment_id(dataset_name)
        produced = data_schematic.norm_stats.get(emb_id, {})
        if not produced:
            raise RuntimeError(f"[NormStats] No stats produced for dataset={dataset_name}")
        print(f"[NormStats] Validated {len(produced)} keys for <{dataset_name}>")

    out_path = Path(CFG.output_mount_path) / _NORM_SUBDIR / data_config / "norm_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats_out: dict = {}
    for emb, keys_dict in data_schematic.norm_stats.items():
        stats_out[str(emb)] = {
            key_name: {
                stat_name: np.asarray(arr).tolist()
                for stat_name, arr in stat_dict.items()
            }
            for key_name, stat_dict in keys_dict.items()
        }

    payload: dict = {
        "stats": stats_out,
        "loading_time": None,
        "computing_time": None,
        "frames": None,
    }
    if data_schematic._norm_run_metadata is not None:
        for k in ("loading_time", "computing_time", "frames"):
            if k in data_schematic._norm_run_metadata:
                payload[k] = data_schematic._norm_run_metadata[k]

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)

    training_outputs_volume.commit()
    print(f"[NormStats] Saved to {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


def _resolve_git_state() -> tuple[str, str, bool]:
    def _git(args):
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()

    git_remote = _git(["git", "config", "--get", "remote.origin.url"])
    git_commit = _git(["git", "rev-parse", "HEAD"])
    is_dirty = bool(_git(["git", "status", "--porcelain"]))

    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin"], cwd=REPO_ROOT, check=True, capture_output=True
        )
        result = subprocess.run(
            ["git", "branch", "-r", "--contains", git_commit],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if not result.stdout.strip():
            raise SystemExit(
                f"ERROR: commit {git_commit[:12]} has not been pushed.\n"
                "Push your branch first, then re-run."
            )
    except subprocess.CalledProcessError:
        pass

    return git_remote, git_commit, is_dirty


@app.local_entrypoint()
def main(*args: str) -> None:
    """Compute and cache norm stats for a data config on Modal CPUs."""
    import argparse

    parser = argparse.ArgumentParser(prog="offline_norm_stats")
    parser.add_argument("data_config", help="Data config name, e.g. mecka_all_zarr")
    parser.add_argument("--cpu", type=float, default=16.0, help="CPU cores (max 64)")
    parser.add_argument("--memory_gb", type=float, default=64.0, help="RAM in GB")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Fraction of episodes to sample (0.0–1.0)")
    parser.add_argument("--batch_size", type=int, default=512, help="DataLoader batch size for norm collection")
    parsed = parser.parse_args(list(args))

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes. Modal runs the last committed state.")

    print(
        f"Submitting norm-stats job: data={parsed.data_config!r} "
        f"cpu={parsed.cpu} memory={parsed.memory_gb}GB "
        f"workers={parsed.num_workers} sample_frac={parsed.sample_frac} "
        f"batch_size={parsed.batch_size}"
    )

    out_path = run_norm_stats.remote(
        data_config=parsed.data_config,
        num_workers=parsed.num_workers,
        sample_frac=parsed.sample_frac,
        batch_size=parsed.batch_size,
        git_remote=git_remote,
        git_commit=git_commit,
    )

    print(f"\nDone. Volume path: {out_path}")
    print(
        f"\nTo use in training:\n"
        f"  norm_stats.precomputed_norm_path={_NORM_SUBDIR}/{parsed.data_config}"
    )
