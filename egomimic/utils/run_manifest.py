"""Record which data a run trained on, next to its checkpoints: ``split.json``
(train / valid dataset keys of each train dataset) and ``lance_versions.json``
(the Lance table versions each dataset read)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SPLIT_FILE = "split.json"
LANCE_FILE = "lance_versions.json"


def run_manifest(train_datasets: dict, valid_datasets: dict) -> dict[str, dict]:
    """{file name: contents} for ``write``; no Lance file without a Lance dataset."""
    split = {
        name: {
            "train": sorted(ds.train_collections),
            "valid": sorted(ds.valid_collections),
        }
        for name, ds in train_datasets.items()
        if hasattr(ds, "train_collections")
    }
    lance = {
        f"{head}/{name}": ds.source_manifest
        for head, datasets in (("train", train_datasets), ("valid", valid_datasets))
        for name, ds in datasets.items()
        if getattr(ds, "source_manifest", None)
    }
    return {SPLIT_FILE: split, **({LANCE_FILE: lance} if lance else {})}


def write(root: str | Path, files: dict[str, dict]) -> None:
    """Write each file into ``root``. A file already there with other contents (a
    requeue or a resume into the same run dir) is kept, and the new one goes to
    ``<stem>.<UTC time>.json``, so the launch that trained the earlier checkpoints
    keeps its record."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name, contents in files.items():
        path = root / name
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) == contents:
                continue
            path = root / f"{path.stem}.{stamp}{path.suffix}"
        path.write_text(json.dumps(contents, indent=1) + "\n", encoding="utf-8")
        log.info("run manifest: wrote %s", path)


def warn_on_split_drift(ckpt_path: str | None, split: dict) -> list[str]:
    """Warn about, and return, the keys the checkpoint's run validated on that this
    run trains on: dataset keys added since then reshuffle ``split_dataset_names``.
    Reads ``split.json`` of the run that wrote ``ckpt_path`` (``<run>/checkpoints/``)."""
    if not ckpt_path or not Path(ckpt_path).is_file():
        return []
    prev_file = Path(ckpt_path).resolve().parent.parent / SPLIT_FILE
    if not prev_file.is_file():
        return []
    prev = json.loads(prev_file.read_text(encoding="utf-8"))
    leaked = []
    for name, cur in split.items():
        moved = sorted(set(prev.get(name, {}).get("valid", [])) & set(cur["train"]))
        if moved:
            log.warning(
                "dataset %r: %d episode(s) the checkpoint's run validated on are in "
                "this run's train split (dataset keys changed since %s), e.g. %s; "
                "its val metrics no longer measure held-out data",
                name,
                len(moved),
                prev_file,
                moved[:5],
            )
            leaked += moved
    return leaked
