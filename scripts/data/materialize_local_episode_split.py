#!/usr/bin/env python3
"""Materialize a deterministic local-Zarr episode split and its audit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from egomimic.rldb.zarr.zarr_dataset_multi import (  # noqa: E402
    LocalEpisodeResolver,
    episode_names_sha256,
    split_dataset_names,
)


def _parse_domain(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("domain must be NAME=/absolute/zarr/root")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"domain path must be absolute: {path}")
    return name, path


def _path_sha256(paths) -> str:
    values = sorted(str(path) for path in paths)
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("utf-8")
    ).hexdigest()


def _domain_evidence(
    name: str, root: Path, *, valid_ratio: float, seed: int
) -> tuple[dict, set[str], set[str]]:
    filtered = LocalEpisodeResolver._get_local_filtered_paths(root)
    path_by_id = {episode_id: Path(path).resolve() for path, episode_id in filtered}
    if len(path_by_id) != len(filtered):
        raise ValueError(f"{name} contains duplicate suffixless episode IDs")
    inventory = set(path_by_id)
    train, valid = split_dataset_names(inventory, valid_ratio=valid_ratio, seed=seed)
    train_paths = {path_by_id[episode_id] for episode_id in train}
    valid_paths = {path_by_id[episode_id] for episode_id in valid}
    id_overlap = train & valid
    path_overlap = train_paths & valid_paths
    union = train | valid
    if id_overlap or path_overlap or union != inventory:
        raise ValueError(
            f"{name} split audit failed: id_overlap={len(id_overlap)}, "
            f"path_overlap={len(path_overlap)}, union={len(union)}/{len(inventory)}"
        )
    return (
        {
            "folder_path": str(root.resolve()),
            "total_count": len(inventory),
            "inventory_names_sha256": episode_names_sha256(inventory),
            "train_count": len(train),
            "train_names_sha256": episode_names_sha256(train),
            "train_resolved_paths_sha256": _path_sha256(train_paths),
            "train_ids": sorted(train),
            "valid_count": len(valid),
            "valid_names_sha256": episode_names_sha256(valid),
            "valid_resolved_paths_sha256": _path_sha256(valid_paths),
            "valid_ids": sorted(valid),
            "id_overlap_count": len(id_overlap),
            "resolved_path_overlap_count": len(path_overlap),
            "union_count": len(union),
            "union_matches_inventory": union == inventory,
        },
        train_paths,
        valid_paths,
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        type=_parse_domain,
        metavar="NAME=/ABSOLUTE/PATH",
    )
    parser.add_argument("--valid-ratio", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    domain_specs = dict(args.domain)
    if len(domain_specs) != len(args.domain):
        raise ValueError("domain names must be unique")

    domains = {}
    all_train_paths: set[Path] = set()
    all_valid_paths: set[Path] = set()
    for name, root in sorted(domain_specs.items()):
        evidence, train_paths, valid_paths = _domain_evidence(
            name, root, valid_ratio=args.valid_ratio, seed=args.seed
        )
        domains[name] = evidence
        all_train_paths.update(train_paths)
        all_valid_paths.update(valid_paths)

    cross_split_path_overlap = all_train_paths & all_valid_paths
    if cross_split_path_overlap:
        raise ValueError(
            "physical train/validation overlap exists across logical domains: "
            f"{sorted(map(str, cross_split_path_overlap))}"
        )

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "split_algorithm": (
            "sorted suffixless resolver IDs; random.Random(seed).shuffle; "
            "first floor(valid_ratio * N), with a one-episode minimum when "
            "valid_ratio > 0"
        ),
        "split_seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "generator_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "cross_domain_train_valid_resolved_path_overlap_count": len(
            cross_split_path_overlap
        ),
        "domains": domains,
    }
    _atomic_write_json(args.output, payload)
    print(
        f"wrote {args.output} with {len(domains)} domains; "
        "train/validation path overlap=0"
    )


if __name__ == "__main__":
    main()
