#!/usr/bin/env python3
"""Create a deterministic metadata inventory for an ImageFolder dataset."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def split_inventory(path: Path):
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for directory, dirnames, filenames in os.walk(path, followlinks=True):
        dirnames.sort()
        filenames.sort()
        base = Path(directory)
        for filename in filenames:
            item = base / filename
            stat = item.stat()
            relative = item.relative_to(path)
            record = f"{relative}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode()
            digest.update(record)
            count += 1
            total_bytes += stat.st_size
    return {"files": count, "bytes": total_bytes, "metadata_sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "data_root": str(args.data_root),
        "train_resolved": str((args.data_root / "train").resolve()),
        "val_resolved": str((args.data_root / "val").resolve()),
        "train": split_inventory(args.data_root / "train"),
        "val": split_inventory(args.data_root / "val"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".sha256").write_text(f"{digest}  {args.output.name}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
