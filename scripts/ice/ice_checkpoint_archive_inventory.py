#!/usr/bin/env python3
"""Inventory checkpoint-bearing ICE runs and their archive status.

Completion is intentionally derived only from a valid run-local COMPLETE.json.
Scheduler state is not used: a Slurm COMPLETED record does not prove that the
terminal checkpoint was validated and durably published by the run owner.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import ice_checkpoint_mirror as core

PRUNE_DIRS = {
    ".git",
    ".venv",
    "data",
    "datasets",
    "env",
    "source",
    "validation_predictions",
    "wandb",
}


def discover_runs(search_root: Path, max_depth: int) -> dict[Path, list[Path]]:
    """Find run roots containing checkpoints without traversing bulky trees."""
    discovered: dict[Path, list[Path]] = {}
    root_depth = len(search_root.parts)
    for raw_dir, dirnames, filenames in os.walk(search_root):
        directory = Path(raw_dir)
        depth = len(directory.parts) - root_depth
        dirnames[:] = [name for name in dirnames if name not in PRUNE_DIRS]
        if depth >= max_depth:
            dirnames.clear()
        if "COMPLETE.json" in filenames:
            discovered.setdefault(directory.resolve(), [])
        checkpoints = [
            (directory / name).resolve()
            for name in filenames
            if name.endswith(".ckpt") and not name.endswith(core.INCOMPLETE_SUFFIXES)
        ]
        if not checkpoints:
            continue
        run_root = directory.parent.resolve() if directory.name == "checkpoints" else directory.resolve()
        discovered.setdefault(run_root, []).extend(checkpoints)
    return {root: sorted(set(paths), key=str) for root, paths in discovered.items()}


def load_state(state_dir: Path) -> dict[str, Any]:
    state_path = state_dir / "mirror-state.json"
    lock_path = state_dir / "state.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        if not state_path.exists():
            return {"schema_version": 2, "files": {}}
        state = json.loads(state_path.read_text())
    if state.get("schema_version") != 2 or not isinstance(state.get("files"), dict):
        raise SystemExit(f"invalid mirror state schema: {state_path}")
    return state


def build_inventory(
    search_root: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    max_depth: int,
) -> dict[str, Any]:
    discovered = discover_runs(search_root, max_depth)
    rows_by_root = {
        Path(row["local_root"]).expanduser().resolve(): row for row in manifest["runs"]
    }
    all_roots = sorted(set(discovered) | set(rows_by_root), key=str)
    runs: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    attention: list[str] = []

    for run_root in all_roots:
        row = rows_by_root.get(run_root)
        checkpoints = discovered.get(run_root, [])
        sentinel = Path(row["completion_sentinel"]) if row and row.get("completion_sentinel") else run_root / "COMPLETE.json"
        terminal = core.completion_sentinel_checkpoint(sentinel)
        complete = terminal is not None
        if row is None:
            status = "unregistered"
            verified = 0
        else:
            checkpoint_paths = {str(path) for path in checkpoints}
            verified = sum(
                1
                for path, entry in state["files"].items()
                if path in checkpoint_paths
                and isinstance(entry, dict)
                and entry.get("run_id") == row["id"]
                and entry.get("remote_verified") is True
            )
            terminal_verified = any(
                isinstance(entry, dict)
                and entry.get("run_id") == row["id"]
                and entry.get("remote_verified") is True
                and terminal is not None
                and entry.get("global_step") == terminal["global_step"]
                and entry.get("sha256") == terminal["sha256"]
                for entry in state["files"].values()
            )
            if not checkpoints:
                status = "complete_missing_checkpoints" if complete else "registered_no_checkpoints"
            elif complete and verified == len(checkpoints) and terminal_verified:
                status = "complete_archived"
            elif complete:
                status = "complete_needs_transfer"
            elif verified == len(checkpoints):
                status = "active_archived"
            else:
                status = "active_needs_transfer"

        counts[status] = counts.get(status, 0) + 1
        if status in {"unregistered", "complete_needs_transfer", "complete_missing_checkpoints"}:
            attention.append(str(run_root))
        runs.append(
            {
                "run_root": str(run_root),
                "run_id": row["id"] if row else None,
                "registered": row is not None,
                "complete": complete,
                "completion_evidence": str(sentinel) if complete else None,
                "checkpoint_count": len(checkpoints),
                "remote_verified_count": verified,
                "terminal_checkpoint_remote_verified": terminal_verified if row else False,
                "status": status,
            }
        )

    return {
        "schema_version": 1,
        "search_root": str(search_root),
        "completion_policy": "valid_run_local_COMPLETE.json_only",
        "generated_at_unix": time.time(),
        "counts": counts,
        "attention_required": attention,
        "runs": runs,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--search-root", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--state-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-depth", type=int, default=6)
    result.add_argument("--require-clear", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    search_root = core.specific_absolute(args.search_root, "search-root")
    state_dir = core.specific_absolute(args.state_dir, "state-dir")
    output = args.output.expanduser().resolve()
    if not search_root.is_dir():
        raise SystemExit(f"search-root is not a directory: {search_root}")
    if args.max_depth < 1:
        raise SystemExit("max-depth must be positive")
    core.contained(state_dir, search_root, "state-dir")
    core.contained(output, search_root, "output")
    manifest = core.load_manifest(args.manifest.expanduser().resolve())
    inventory = build_inventory(search_root, manifest, load_state(state_dir), args.max_depth)
    core.atomic_json(output, inventory)
    print(json.dumps(inventory, sort_keys=True))
    return 1 if args.require_clear and inventory["attention_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
