"""Auditable Hydra evaluation entry point.

This wrapper keeps the existing evaluation implementation in ``trainHydra`` but
records the provenance needed to reproduce every evaluation attempt.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_DIR = Path(__file__).resolve().parent.parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from egomimic.trainHydra import train
from egomimic.pl_utils.utils import extras


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _simulator_fingerprint() -> dict[str, Any]:
    return {
        "path": "Tsimulation",
        "tree_hash": _git("rev-parse", "HEAD:Tsimulation"),
        "commit": _git("rev-parse", "HEAD"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _manifest_path(cfg: DictConfig) -> Path:
    configured = OmegaConf.select(cfg, "eval_metadata.manifest_path", default=None)
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return Path(HydraConfig.get().runtime.output_dir) / "eval_manifest.json"


def _base_manifest(cfg: DictConfig) -> dict[str, Any]:
    checkpoint = cfg.get("ckpt_path", None) or os.environ.get("EVAL_CHECKPOINT")
    return {
        "schema_version": 1,
        "status": "running",
        "launched_at": _utc_now(),
        "completed_at": None,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "simulator": _simulator_fingerprint(),
        "checkpoint": checkpoint or cfg.get("checkpoint", None),
        "attempt": OmegaConf.select(cfg, "eval_metadata.attempt", default=1),
        "supersedes": OmegaConf.select(cfg, "eval_metadata.supersedes", default=None),
        "rerun_of": OmegaConf.select(cfg, "eval_metadata.rerun_of", default=None),
        # Keep interpolation expressions intact. Archived Hydra configs can
        # reference runtime-only keys such as `${paths.root_dir}` that are not
        # available until the training process composes its final config.
        "dataset": _jsonable(OmegaConf.to_container(cfg.get("data"), resolve=False)),
        "config": _jsonable(OmegaConf.to_container(cfg, resolve=False)),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@hydra.main(
    version_base="1.3",
    config_path="./hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> None:
    extras(cfg)
    with open_dict(cfg):
        cfg.mode = "eval"
        cfg.train = False
        cfg.eval = True
        if os.environ.get("EVAL_CHECKPOINT") and not cfg.get("ckpt_path"):
            cfg.ckpt_path = os.environ["EVAL_CHECKPOINT"]

    manifest_path = _manifest_path(cfg)
    manifest = _base_manifest(cfg)
    _write_manifest(manifest_path, manifest)
    try:
        metrics, _ = train(cfg)
    except BaseException as exc:
        manifest.update(
            status="failed",
            completed_at=_utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _write_manifest(manifest_path, manifest)
        raise
    manifest.update(status="success", completed_at=_utc_now(), metrics=_jsonable(metrics))
    _write_manifest(manifest_path, manifest)
    print(f"Evaluation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
