#!/usr/bin/env python3
"""Publish an immutable, level-balanced ChainGripper obstacle subset.

The publisher never modifies or copies episode payloads. It creates a new
audited directory tree containing relative symlinks to an already-audited
source corpus, then atomically renames the completed staging directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

LEVELS = tuple(range(1, 31))
EPISODE_RE = re.compile(r"_(\d{6})\.zarr$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _evenly_spaced_indices(first: int, last: int, count: int) -> list[int]:
    if count < 1 or last < first or count > last - first + 1:
        raise ValueError("invalid inclusive sampling interval")
    if count == 1:
        return [first]
    denominator = count - 1
    span = last - first
    indices = [
        first + (rank * span + denominator // 2) // denominator
        for rank in range(count)
    ]
    if len(indices) != len(set(indices)):
        raise AssertionError("evenly spaced selection produced duplicate indices")
    return indices


def _selected_indices(
    *, source_per_level: int, manual_per_level: int, target_per_level: int
) -> tuple[list[int], list[int], list[int]]:
    if not 0 <= manual_per_level <= target_per_level <= source_per_level:
        raise ValueError("require 0 <= manual <= target <= source per level")
    manual = list(range(manual_per_level))
    generated_count = target_per_level - manual_per_level
    generated = _evenly_spaced_indices(
        manual_per_level,
        source_per_level - 1,
        generated_count,
    )
    selected = manual + generated
    if len(selected) != target_per_level or len(selected) != len(set(selected)):
        raise AssertionError("selection cardinality mismatch")
    return manual, generated, selected


def _episode_index(path: Path) -> int:
    match = EPISODE_RE.search(path.name)
    if match is None:
        raise ValueError(f"unexpected episode name: {path.name}")
    return int(match.group(1))


def _episode_total_frames(path: Path) -> int:
    metadata = json.loads((path / "zarr.json").read_text())
    total_frames = int(metadata["attributes"]["total_frames"])
    if total_frames <= 0:
        raise ValueError(f"invalid total_frames for {path}: {total_frames}")
    return total_frames


def _build_publication(
    *,
    source_root: Path,
    source_audit_sha256: str,
    source_per_level: int,
    manual_per_level: int,
    target_per_level: int,
) -> tuple[dict, bytes, dict, bytes]:
    source_audit_path = source_root / "audit_report.json"
    if _sha256_file(source_audit_path) != source_audit_sha256:
        raise ValueError("source audit SHA-256 mismatch")
    source_audit = json.loads(source_audit_path.read_text())
    generated_source_per_level = source_per_level - manual_per_level
    if source_audit.get("status") != "PASS":
        raise ValueError("source audit did not pass")
    if source_audit.get("levels") != list(LEVELS):
        raise ValueError("source audit level set mismatch")
    if source_audit.get("target_total_per_level") != source_per_level:
        raise ValueError("source audit episode cardinality mismatch")
    if source_audit.get("manual_per_level") != manual_per_level:
        raise ValueError("source audit manual cardinality mismatch")
    if source_audit.get("generated_per_level") != generated_source_per_level:
        raise ValueError("source audit generated cardinality mismatch")

    manual, generated, selected = _selected_indices(
        source_per_level=source_per_level,
        manual_per_level=manual_per_level,
        target_per_level=target_per_level,
    )
    inventory_lines: list[str] = []
    episodes: list[dict] = []
    level_reports: list[dict] = []
    total_frames = 0
    for level in LEVELS:
        level_root = source_root / f"level_{level:02d}" / "chain_gripper" / "T"
        source_paths = sorted(level_root.glob("*.zarr"))
        indexed = {_episode_index(path): path for path in source_paths}
        if set(indexed) != set(range(source_per_level)):
            raise ValueError(f"level {level} source inventory is not exactly indexed")
        level_frames = 0
        for index in selected:
            source_path = indexed[index]
            frames = _episode_total_frames(source_path)
            relative_path = source_path.relative_to(source_root).as_posix()
            inventory_lines.append(f"level_{level:02d}/{source_path.name}")
            episodes.append(
                {
                    "episode_index": index,
                    "level": level,
                    "provenance": "manual" if index < manual_per_level else "generated",
                    "source_relative_path": relative_path,
                    "total_frames": frames,
                }
            )
            level_frames += frames
        total_frames += level_frames
        level_reports.append(
            {
                "generated": len(generated),
                "level": level,
                "manual": len(manual),
                "total": len(selected),
                "total_frames": level_frames,
            }
        )

    inventory_bytes = "".join(f"{line}\n" for line in inventory_lines).encode()
    publisher_sha256 = _sha256_file(Path(__file__).resolve())
    manifest = {
        "episodes": episodes,
        "generated_indices": generated,
        "levels": list(LEVELS),
        "manual_indices": manual,
        "publisher_sha256": publisher_sha256,
        "schema_version": 1,
        "selection": (
            "retain every manual episode, then choose generated indices evenly "
            "across their full inclusive source-index range"
        ),
        "source_audit_sha256": source_audit_sha256,
        "source_root": str(source_root),
        "target_per_level": target_per_level,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
    }
    manifest_bytes = _json_bytes(manifest)
    audit = {
        "generated_per_level": len(generated),
        "inventory_sha256": _sha256_bytes(inventory_bytes),
        "levels": list(LEVELS),
        "manual_per_level": len(manual),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "publisher_sha256": publisher_sha256,
        "reports": level_reports,
        "schema_version": 1,
        "source_audit_sha256": source_audit_sha256,
        "source_root": str(source_root),
        "status": "PASS",
        "target_total_per_level": target_per_level,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
    }
    return manifest, inventory_bytes, audit, _json_bytes(audit)


def _verify_tree(
    *,
    output_root: Path,
    source_root: Path,
    manifest: dict,
    inventory_bytes: bytes,
    audit_bytes: bytes,
) -> None:
    if (output_root / "subset_manifest.json").read_bytes() != _json_bytes(manifest):
        raise ValueError("published subset manifest mismatch")
    if (output_root / "inventory.txt").read_bytes() != inventory_bytes:
        raise ValueError("published subset inventory mismatch")
    if (output_root / "audit_report.json").read_bytes() != audit_bytes:
        raise ValueError("published subset audit mismatch")
    for episode in manifest["episodes"]:
        source_path = source_root / episode["source_relative_path"]
        destination = (
            output_root
            / f"level_{episode['level']:02d}"
            / "chain_gripper"
            / "T"
            / source_path.name
        )
        if not destination.is_symlink():
            raise ValueError(f"not a symlink: {destination}")
        if destination.resolve() != source_path.resolve():
            raise ValueError(f"symlink target mismatch: {destination}")


def _publish(
    *,
    output_root: Path,
    source_root: Path,
    manifest: dict,
    inventory_bytes: bytes,
    audit_bytes: bytes,
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.incomplete-", dir=output_root.parent)
    )
    try:
        for episode in manifest["episodes"]:
            source_path = source_root / episode["source_relative_path"]
            destination_dir = (
                staging
                / f"level_{episode['level']:02d}"
                / "chain_gripper"
                / "T"
            )
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source_path.name
            relative_target = os.path.relpath(source_path, start=destination_dir)
            destination.symlink_to(relative_target, target_is_directory=True)
        (staging / "subset_manifest.json").write_bytes(_json_bytes(manifest))
        (staging / "inventory.txt").write_bytes(inventory_bytes)
        (staging / "audit_report.json").write_bytes(audit_bytes)
        for path in (
            staging / "subset_manifest.json",
            staging / "inventory.txt",
            staging / "audit_report.json",
        ):
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        _verify_tree(
            output_root=staging,
            source_root=source_root,
            manifest=manifest,
            inventory_bytes=inventory_bytes,
            audit_bytes=audit_bytes,
        )
        for directory in sorted(
            (
                path
                for path in staging.rglob("*")
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        staging.chmod(0o555)
        os.replace(staging, output_root)
    except BaseException:
        if staging.exists():
            for directory in (staging, *staging.rglob("*")):
                if directory.is_dir() and not directory.is_symlink():
                    directory.chmod(0o755)
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-audit-sha256", required=True)
    parser.add_argument("--source-per-level", type=int, default=128)
    parser.add_argument("--manual-per-level", type=int, default=16)
    parser.add_argument("--target-per-level", type=int, default=100)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    output_root = args.output_root.absolute()
    manifest, inventory_bytes, audit, audit_bytes = _build_publication(
        source_root=source_root,
        source_audit_sha256=args.expected_source_audit_sha256,
        source_per_level=args.source_per_level,
        manual_per_level=args.manual_per_level,
        target_per_level=args.target_per_level,
    )
    if args.verify_existing:
        _verify_tree(
            output_root=output_root,
            source_root=source_root,
            manifest=manifest,
            inventory_bytes=inventory_bytes,
            audit_bytes=audit_bytes,
        )
    else:
        _publish(
            output_root=output_root,
            source_root=source_root,
            manifest=manifest,
            inventory_bytes=inventory_bytes,
            audit_bytes=audit_bytes,
        )
    print(f"output_root={output_root}")
    print(f"episodes={manifest['total_episodes']}")
    print(f"frames={manifest['total_frames']}")
    print(f"inventory_sha256={audit['inventory_sha256']}")
    print(f"audit_sha256={_sha256_bytes(audit_bytes)}")


if __name__ == "__main__":
    main()
