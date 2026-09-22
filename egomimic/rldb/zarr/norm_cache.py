"""Content-keyed norm-stat cache: same episodes + recipe share one file; any
change to either yields a new key so stale stats can never load silently.

The key covers each episode's store metadata (so a re-export under the same
hash, or an array added or rewritten in place, misses) and the source of every
egomimic module the recipe's code comes from (the episode reader and stats
methods, keymaps, transforms). Code reached some other way (trainHydra, lazy
imports) is not hashed: bump KEY_VERSION with changes there that alter the
sampled values."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from hydra.utils import get_class, get_object
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

KEY_VERSION = 2
# Resolver fields that say where the data lives, not what it is.
_LOCATION_FIELDS = (
    "folder_path",
    "bucket_name",
    "main_prefix",
    "debug",
    "norm_stats",
)
# Store-level metadata of a zarr v3 (zarr.json) or v2 group/array.
_V3_METADATA = ("zarr.json",)
_V2_METADATA = (".zgroup", ".zarray", ".zattrs")
# Holds the episode reader, resolvers and stats methods; hashed for every recipe.
_READER_MODULE = "egomimic.rldb.zarr.zarr_dataset_multi"


def episode_fingerprint(episode_path) -> str | None:
    """Digest of the name, mtime and size of the metadata files of the episode
    group and of each array directly under it (episodes are flat), or None if
    there are none. Re-exporting, or adding, deleting or rewriting an array in
    place (e.g. re-annotating), rewrites at least one of them. Only stats and a
    directory listing: chunk data is never read, so an in-place write of chunk
    bytes alone is not seen. A new mtime alone only costs a recompute."""
    if episode_path is None:
        return None
    root = Path(episode_path)
    names = _V3_METADATA if (root / "zarr.json").exists() else _V2_METADATA
    try:
        children = sorted(e.name for e in os.scandir(root) if e.is_dir())
    except OSError:
        return None
    digest = hashlib.sha256()
    found = False
    for rel in ["", *children]:
        for name in names:
            try:
                st = (root / rel / name).stat()
            except OSError:
                continue
            digest.update(f"{rel}/{name}\0{st.st_mtime_ns}\0{st.st_size}\0".encode())
            found = True
    return digest.hexdigest() if found else None


def _targets(node):
    if isinstance(node, Mapping):
        if isinstance(node.get("_target_"), str):
            yield node["_target_"]
        for v in node.values():
            yield from _targets(v)
    elif isinstance(node, list):
        for v in node:
            yield from _targets(v)


def _is_egomimic(name) -> bool:
    return isinstance(name, str) and name.startswith("egomimic.")


def code_modules(recipe: Mapping) -> list[str]:
    """The reader module, every egomimic module defining a ``_target_`` in the
    recipe, and the egomimic modules those reach through top-level imports."""
    targets = {get_object(t).__module__ for t in _targets(recipe)}
    todo = {_READER_MODULE, *filter(_is_egomimic, targets)}
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        seen.add(name)
        mod = importlib.import_module(name)
        if hasattr(mod, "__path__"):
            continue  # a package's globals include whatever submodules got imported
        for v in vars(mod).values():
            dep = v.__name__ if inspect.ismodule(v) else getattr(v, "__module__", None)
            if _is_egomimic(dep) and dep not in seen:
                todo.add(dep)
    return sorted(seen)


@functools.lru_cache(maxsize=None)
def _module_source_hash(name: str) -> str:
    file = getattr(importlib.import_module(name), "__file__", None)
    return hashlib.sha256(Path(file).read_bytes() if file else b"").hexdigest()


def code_hash(recipe: Mapping) -> str:
    """sha256 over (module name, source) of ``code_modules``: no paths or mtimes,
    so identical code hashes identically in any checkout."""
    digest = hashlib.sha256()
    for name in code_modules(recipe):
        digest.update(f"{name}\0{_module_source_hash(name)}\0".encode())
    return digest.hexdigest()


def _to_dict(dataset_cfg: DictConfig | dict) -> dict:
    if isinstance(dataset_cfg, DictConfig):
        return OmegaConf.to_container(dataset_cfg, resolve=True)
    return dict(dataset_cfg)


def recipe_inputs(dataset_cfg: DictConfig | dict) -> dict:
    """The dataset config with the resolver's location fields removed and its
    class replaced by the dataset class it loads: S3 and Local resolvers share
    entries, annotation-cutoff and plain ones do not."""
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

    cfg = _to_dict(dataset_cfg)
    resolver = dict(cfg.get("resolver") or {})
    for f in _LOCATION_FIELDS:
        resolver.pop(f, None)
    target = resolver.pop("_target_", None)
    if target is not None:
        cls = getattr(get_class(target), "_dataset_class", None) or ZarrDataset
        resolver["dataset_class"] = f"{cls.__module__}.{cls.__qualname__}"
    cfg["resolver"] = resolver
    return cfg


def cache_inputs(
    dataset_name: str,
    episodes: Mapping[str, str | None],
    dataset_cfg,
    sample_frac: float,
) -> dict:
    """Everything the key hashes. ``episodes`` maps episode hash to its
    ``episode_fingerprint``. norm_mode is not an input: every stat is computed
    regardless of it."""
    cfg = _to_dict(dataset_cfg)
    return {
        "v": KEY_VERSION,
        "dataset": dataset_name,
        "episodes": sorted([str(h), fp] for h, fp in episodes.items()),
        "recipe": recipe_inputs(cfg),
        "sample_frac": float(sample_frac),
        "code": code_hash(cfg),
    }


def norm_cache_key(inputs: dict) -> str:
    blob = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_path(cache_dir, dataset_name: str, key: str) -> Path:
    return Path(cache_dir) / dataset_name / f"{key}.json"


def find_cached(
    cache_dir, dataset_name: str, key: str, embodiment_id: int
) -> Path | None:
    p = cache_path(cache_dir, dataset_name, key)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text())
        if str(embodiment_id) not in payload["stats"]:
            logger.warning(
                "norm cache %s has no embodiment %s; recomputing", p, embodiment_id
            )
            return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("norm cache %s unreadable (%s); recomputing", p, e)
        return None
    return p


@functools.cache
def _git_sha() -> str | None:
    """HEAD of the checkout that contains this file, or None (installed wheel, no git)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


def write_cached(
    cache_dir,
    dataset_name: str,
    key: str,
    inputs: dict,
    embodiment_id: int,
    stats: dict,
    norm_run_metadata: dict | None,
) -> Path | None:
    """Write the cache entry and return its path, or None if the write failed --
    a cache write must never fail a training run."""
    p = cache_path(cache_dir, dataset_name, key)
    payload = {
        "stats": {
            str(embodiment_id): {
                k: {n: np.asarray(a).tolist() for n, a in d.items()}
                for k, d in stats.items()
            }
        },
        "norm_run_metadata": norm_run_metadata,
        "cache": {
            "key": key,
            "inputs": inputs,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
        },
    }
    # Every DDP rank runs train(), and two jobs (on any node) can share
    # EGOVERSE_CACHE_DIR, so each writer needs its own tmp file.
    tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, p)
    except OSError as e:
        logger.warning("norm cache write failed (%s): %s", p, e)
        with contextlib.suppress(OSError):
            tmp.unlink()
        return None
    logger.info("norm cache written: %s", p)
    return p
