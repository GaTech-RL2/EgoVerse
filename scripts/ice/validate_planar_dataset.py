#!/usr/bin/env python3
"""Validate a portable Planar dataset against its immutable split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DOMAIN = "pushshapes_sim_u_socket"
SPLIT_SEED = 42
VALID_RATIO = 0.01
SPLIT_ALGORITHM = (
    "sorted suffixless resolver IDs; random.Random(seed).shuffle; "
    "first floor(valid_ratio * N), with a one-episode minimum when "
    "valid_ratio > 0"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: Iterable[str]) -> str:
    payload = "".join(f"{name}\n" for name in sorted(map(str, names)))
    return hashlib.sha256(payload.encode()).hexdigest()


def paths_sha256(paths: Iterable[Path]) -> str:
    payload = "".join(f"{path}\n" for path in sorted(map(str, paths)))
    return hashlib.sha256(payload.encode()).hexdigest()


def split_names(names: Iterable[str]) -> tuple[set[str], set[str]]:
    ordered = sorted(names)
    random.Random(SPLIT_SEED).shuffle(ordered)
    count = max(1, int(len(ordered) * VALID_RATIO)) if ordered else 0
    return set(ordered[count:]), set(ordered[:count])


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _id_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item and "/" not in item for item in value
    ):
        raise RuntimeError(f"{label} must be a list of suffixless episode IDs")
    if value != sorted(value) or len(value) != len(set(value)):
        raise RuntimeError(f"{label} must be sorted and unique")
    return value


def validate_manifest_structure(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise RuntimeError("split manifest must be a schema-1 PASS artifact")
    if manifest.get("split_algorithm") != SPLIT_ALGORITHM:
        raise RuntimeError("split manifest algorithm is not the training algorithm")
    if manifest.get("split_seed") != SPLIT_SEED:
        raise RuntimeError("split manifest must use seed 42")
    if manifest.get("valid_ratio") != VALID_RATIO:
        raise RuntimeError("split manifest must use a 1% validation ratio")
    if manifest.get("cross_domain_train_valid_resolved_path_overlap_count") != 0:
        raise RuntimeError("split manifest records cross-domain path overlap")
    _digest(manifest.get("generator_sha256"), "generator_sha256")

    domains = manifest.get("domains")
    if not isinstance(domains, Mapping) or set(domains) != {DOMAIN}:
        raise RuntimeError(f"split manifest must contain only {DOMAIN}")
    domain = domains[DOMAIN]
    if not isinstance(domain, Mapping):
        raise RuntimeError("split manifest domain must be an object")
    source_root = domain.get("folder_path")
    if not isinstance(source_root, str) or not Path(source_root).is_absolute():
        raise RuntimeError("manifest source folder_path must be absolute")

    train_ids = _id_list(domain.get("train_ids"), "train_ids")
    valid_ids = _id_list(domain.get("valid_ids"), "valid_ids")
    train = set(train_ids)
    valid = set(valid_ids)
    inventory = train | valid
    if train & valid:
        raise RuntimeError("manifest train and validation IDs overlap")
    expected_train, expected_valid = split_names(inventory)
    if train != expected_train or valid != expected_valid:
        raise RuntimeError("manifest IDs do not reproduce the seed-42 1% split")

    expected_scalars = {
        "total_count": len(inventory),
        "train_count": len(train),
        "valid_count": len(valid),
        "id_overlap_count": 0,
        "resolved_path_overlap_count": 0,
        "union_count": len(inventory),
        "union_matches_inventory": True,
    }
    for key, expected in expected_scalars.items():
        if domain.get(key) != expected:
            raise RuntimeError(
                f"manifest {key} mismatch: {domain.get(key)!r} != {expected!r}"
            )
    expected_digests = {
        "inventory_names_sha256": names_sha256(inventory),
        "train_names_sha256": names_sha256(train),
        "valid_names_sha256": names_sha256(valid),
    }
    for key, expected in expected_digests.items():
        if _digest(domain.get(key), key) != expected:
            raise RuntimeError(f"manifest {key} does not match its episode IDs")
    _digest(domain.get("train_resolved_paths_sha256"), "train path SHA-256")
    _digest(domain.get("valid_resolved_paths_sha256"), "valid path SHA-256")
    return {
        "domain": DOMAIN,
        "source_dataset_root": source_root,
        "inventory": inventory,
        "train": train,
        "valid": valid,
        **expected_scalars,
        **expected_digests,
    }


def validate_physical_inventory(
    manifest: Mapping[str, Any],
    entries: Sequence[tuple[str | Path, str]],
    dataset_root: Path,
    *,
    excluded_deleted_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    expected = validate_manifest_structure(manifest)
    root = dataset_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"dataset root is not a directory: {root}")

    path_by_id: dict[str, Path] = {}
    raw_paths: set[Path] = set()
    for raw_path, episode_id in entries:
        path = Path(raw_path)
        if not path.is_absolute():
            raise RuntimeError(f"resolver returned a relative episode path: {path}")
        raw_paths.add(path)
        if episode_id in path_by_id:
            raise RuntimeError(f"duplicate suffixless episode ID: {episode_id}")
        path_by_id[episode_id] = path.resolve(strict=True)

    physical_directories = {
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    }
    excluded = {Path(path) for path in excluded_deleted_paths}
    if not excluded <= physical_directories or excluded & raw_paths:
        raise RuntimeError("invalid excluded-deleted directory accounting")
    skipped = physical_directories - raw_paths - excluded
    if skipped:
        raise RuntimeError(
            "dataset contains directories rejected by the training resolver: "
            + ", ".join(sorted(path.name for path in skipped))
        )
    resolved_paths = set(path_by_id.values())
    if len(resolved_paths) != len(path_by_id):
        raise RuntimeError("multiple episode IDs resolve to the same physical path")

    inventory = set(path_by_id)
    if inventory != expected["inventory"]:
        missing = sorted(expected["inventory"] - inventory)
        extra = sorted(inventory - expected["inventory"])
        raise RuntimeError(
            f"physical episode inventory mismatch: missing={missing}, extra={extra}"
        )
    if names_sha256(inventory) != expected["inventory_names_sha256"]:
        raise RuntimeError("physical inventory episode-name SHA-256 mismatch")

    train, valid = split_names(inventory)
    if train != expected["train"] or valid != expected["valid"]:
        raise RuntimeError("physical inventory does not reproduce the committed split")
    train_paths = {path_by_id[name] for name in train}
    valid_paths = {path_by_id[name] for name in valid}
    id_overlap = train & valid
    path_overlap = train_paths & valid_paths
    union = train | valid
    if id_overlap or path_overlap or union != inventory:
        raise RuntimeError(
            "physical split audit failed: "
            f"id_overlap={len(id_overlap)}, path_overlap={len(path_overlap)}, "
            f"union={len(union)}/{len(inventory)}"
        )
    return {
        "dataset_root": str(root),
        "total_count": len(inventory),
        "inventory_names_sha256": names_sha256(inventory),
        "train_count": len(train),
        "train_names_sha256": names_sha256(train),
        "train_resolved_paths_sha256": paths_sha256(train_paths),
        "valid_count": len(valid),
        "valid_names_sha256": names_sha256(valid),
        "valid_resolved_paths_sha256": paths_sha256(valid_paths),
        "id_overlap_count": len(id_overlap),
        "resolved_path_overlap_count": len(path_overlap),
        "union_count": len(union),
        "union_matches_inventory": union == inventory,
        "excluded_deleted_directory_count": len(excluded),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite dataset validation: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    expected_digest = _digest(
        args.expected_manifest_sha256, "expected manifest SHA-256"
    )
    actual_digest = sha256_file(manifest_path)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"split manifest SHA-256 mismatch: {actual_digest} != {expected_digest}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid split manifest JSON: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError("split manifest root must be an object")

    generator_path = repo / "scripts" / "data" / "materialize_local_episode_split.py"
    generator_digest = sha256_file(generator_path)
    if generator_digest != _digest(
        manifest.get("generator_sha256"), "generator_sha256"
    ):
        raise RuntimeError("split manifest generator does not match repository source")

    sys.path.insert(0, str(repo))
    import zarr  # noqa: PLC0415

    from egomimic.rldb.zarr.zarr_dataset_multi import (  # noqa: PLC0415
        LocalEpisodeResolver,
        episode_names_sha256,
        split_dataset_names,
    )

    root = args.dataset_root.expanduser().resolve(strict=True)
    entries = LocalEpisodeResolver._get_local_filtered_paths(root)
    accepted_paths = {Path(path) for path, _ in entries}
    excluded_deleted_paths = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith(".") or path in accepted_paths:
            continue
        try:
            metadata = dict(zarr.open_group(str(path), mode="r").attrs)
        except Exception as exc:
            raise RuntimeError(
                f"dataset contains an unreadable directory rejected by resolver: {path}"
            ) from exc
        if metadata.get("is_deleted") is not True:
            raise RuntimeError(
                f"dataset contains a non-deleted directory rejected by resolver: {path}"
            )
        excluded_deleted_paths.append(path)
    report = validate_physical_inventory(
        manifest,
        entries,
        root,
        excluded_deleted_paths=excluded_deleted_paths,
    )
    training_train, training_valid = split_dataset_names(
        report_ids := {episode_id for _, episode_id in entries},
        valid_ratio=VALID_RATIO,
        seed=SPLIT_SEED,
    )
    if training_train != split_names(report_ids)[0] or training_valid != split_names(
        report_ids
    )[1]:
        raise RuntimeError("portable validator split differs from training implementation")
    if episode_names_sha256(report_ids) != names_sha256(report_ids):
        raise RuntimeError("portable validator hash differs from training implementation")

    payload = {
        "schema_version": 1,
        "status": "DATASET_VALIDATED",
        "manifest": str(manifest_path),
        "manifest_sha256": actual_digest,
        "split_seed": SPLIT_SEED,
        "valid_ratio": VALID_RATIO,
        "split_algorithm": SPLIT_ALGORITHM,
        "generator_sha256": generator_digest,
        "domain": DOMAIN,
        "physical_inventory": report,
    }
    output = args.output.expanduser().resolve()
    _atomic_json(output, payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
