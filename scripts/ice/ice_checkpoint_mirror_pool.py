#!/usr/bin/env python3
"""Collaboratively mirror ICE checkpoints from multiple CPU nodes.

Each process is an identical worker over a shared state directory. Short global
locks protect JSON state and the event log; per-checkpoint locks ensure that
validation, hashing, transfer, and verification happen exactly once at a time.
Worker zero additionally owns storage-pressure reporting, guarded pruning, and
the terminal completion record.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import ice_checkpoint_archive_inventory as inventory
import ice_checkpoint_mirror as core

STOP = False


class LockedFile:
    """Advisory cross-process lock backed by a shared regular file."""

    def __init__(self, path: Path, *, blocking: bool = True) -> None:
        self.path = path
        self.blocking = blocking
        self.stream: Any = None

    def __enter__(self) -> "LockedFile | None":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        operation = fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(self.stream, operation)
        except BlockingIOError:
            self.stream.close()
            self.stream = None
            return None
        return self

    def __exit__(self, *_unused: object) -> None:
        if self.stream is not None:
            self.stream.close()


def task_key(path: Path) -> str:
    return hashlib.sha256(os.fsencode(str(path))).hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 2, "files": {}}
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid mirror state: {path}: {exc}") from exc
    if state.get("schema_version") != 2 or not isinstance(state.get("files"), dict):
        raise RuntimeError(f"invalid mirror state schema: {path}")
    return state


def read_entry(state_dir: Path, checkpoint: Path) -> dict[str, Any] | None:
    with LockedFile(state_dir / "state.lock"):
        entry = load_state(state_dir / "mirror-state.json")["files"].get(str(checkpoint))
    return dict(entry) if isinstance(entry, dict) else None


def update_entry(state_dir: Path, checkpoint: Path, entry: dict[str, Any]) -> None:
    with LockedFile(state_dir / "state.lock"):
        state_path = state_dir / "mirror-state.json"
        state = load_state(state_path)
        state["files"][str(checkpoint)] = entry
        core.atomic_json(state_path, state)


def log_event(state_dir: Path, payload: dict[str, Any]) -> None:
    with LockedFile(state_dir / "events.lock"):
        core.append_event(state_dir / "mirror-events.jsonl", payload)


def ordered_candidates(row: dict[str, Any], worker_index: int) -> list[Path]:
    paths = core.candidates(row)
    if not paths:
        return paths
    offset = worker_index % len(paths)
    return paths[offset:] + paths[:offset]


def claim(state_dir: Path, checkpoint: Path) -> LockedFile:
    return LockedFile(state_dir / "task-locks" / f"{task_key(checkpoint)}.lock", blocking=False)


def process_checkpoint(
    args: argparse.Namespace,
    state_dir: Path,
    row: dict[str, Any],
    checkpoint: Path,
) -> bool:
    """Process one claimed checkpoint; return whether it is remotely verified."""
    prior = read_entry(state_dir, checkpoint)
    info = core.stable_valid_info(
        checkpoint,
        Path(row["validator"]).expanduser().resolve(),
        cached=prior,
    )
    if info is None:
        log_event(
            state_dir,
            {"event": "checkpoint_not_stably_valid", "run_id": row["id"], "path": str(checkpoint)},
        )
        return False

    entry = core.state_entry(info, row["id"], prior)
    update_entry(state_dir, checkpoint, entry)
    remote_path = entry.get("remote_path")
    if (
        entry.get("remote_verified")
        and isinstance(remote_path, str)
        and core.remote_sha(args.ssh, args.remote_host, remote_path) == info.sha256
    ):
        return True

    entry["remote_verified"] = False
    try:
        remote_path = core.mirror_one(
            info.path,
            info.sha256,
            row["remote_dir"],
            args.remote_host,
            args.ssh,
            args.rsync,
            args.rsync_rsh,
            temporary_tag=task_key(info.path)[:24],
        )
    except Exception as exc:
        entry["last_error"] = str(exc)
        entry["last_error_at_unix"] = time.time()
        update_entry(state_dir, checkpoint, entry)
        log_event(
            state_dir,
            {"event": "mirror_error", "run_id": row["id"], "path": str(checkpoint), "error": str(exc)},
        )
        print(f"mirror error for {row['id']}: {exc}", file=sys.stderr, flush=True)
        return False

    entry.update(
        {
            "remote_path": remote_path,
            "remote_verified": True,
            "verified_at_unix": time.time(),
        }
    )
    entry.pop("last_error", None)
    entry.pop("last_error_at_unix", None)
    update_entry(state_dir, checkpoint, entry)
    log_event(
        state_dir,
        {
            "event": "mirrored",
            "worker_index": args.worker_index,
            "run_id": row["id"],
            "path": str(checkpoint),
            "global_step": info.global_step,
            "sha256": info.sha256,
            "remote_path": remote_path,
        },
    )
    return True


def prune_row(args: argparse.Namespace, state_dir: Path, row: dict[str, Any]) -> bool:
    """Prune old local files only after fresh local and remote verification."""
    ok = True
    validator = Path(row["validator"]).expanduser().resolve()
    infos: list[core.CheckpointInfo] = []
    for checkpoint in core.candidates(row):
        prior = read_entry(state_dir, checkpoint)
        info = core.stable_valid_info(checkpoint, validator, cached=prior)
        if info is None:
            ok = False
        else:
            infos.append(info)
    infos.sort(key=lambda item: (item.global_step, item.mtime_ns, str(item.path)), reverse=True)
    for info in infos[int(row.get("retain_local", 2)) :]:
        with claim(state_dir, info.path) as reservation:
            if reservation is None:
                ok = False
                continue
            entry = read_entry(state_dir, info.path)
            if not entry or not entry.get("remote_verified"):
                ok = False
                continue
            current = core.stable_valid_info(info.path, validator, force=True)
            remote_path = entry.get("remote_path")
            if (
                current is None
                or current.sha256 != entry.get("sha256")
                or not isinstance(remote_path, str)
                or core.remote_sha(args.ssh, args.remote_host, remote_path) != current.sha256
            ):
                ok = False
                continue
            try:
                final_stat = info.path.stat()
            except (FileNotFoundError, OSError):
                ok = False
                continue
            if core.stat_identity(final_stat) != (current.size, current.mtime_ns, current.inode):
                ok = False
                continue
            info.path.unlink()
            log_event(
                state_dir,
                {
                    "event": "pruned",
                    "run_id": row["id"],
                    "path": str(info.path),
                    "global_step": current.global_step,
                    "sha256": current.sha256,
                },
            )
    return ok


def maintain(args: argparse.Namespace, state_dir: Path, manifest: dict[str, Any]) -> bool:
    """Publish pressure and completion state; optionally perform guarded pruning."""
    ok = True
    if args.inventory_search_root is not None:
        try:
            report = inventory.build_inventory(
                args.inventory_search_root,
                manifest,
                inventory.load_state(state_dir),
                args.inventory_max_depth,
            )
            core.atomic_json(state_dir / "archive-inventory.json", report)
        except Exception as exc:
            ok = False
            log_event(state_dir, {"event": "inventory_error", "error": str(exc)})
    try:
        used = core.scratch_bytes(args.scratch_root, args.du_timeout_seconds)
        fraction = used / args.quota_bytes
        pressure = (
            "hard"
            if fraction >= args.hard_used_fraction
            else "soft"
            if fraction >= args.soft_used_fraction
            else "normal"
        )
        pressure_payload = {
            "scratch_root": str(args.scratch_root),
            "used_bytes": used,
            "quota_bytes": args.quota_bytes,
            "used_fraction": fraction,
            "pressure": pressure,
            "worker_count": args.worker_count,
            "updated_at_unix": time.time(),
        }
    except Exception as exc:
        ok = False
        pressure_payload = {
            "scratch_root": str(args.scratch_root),
            "quota_bytes": args.quota_bytes,
            "pressure": "unknown",
            "error": str(exc),
            "worker_count": args.worker_count,
            "updated_at_unix": time.time(),
        }
        log_event(state_dir, {"event": "scratch_usage_error", "error": str(exc)})
    core.atomic_json(state_dir / "storage-pressure.json", pressure_payload)

    if args.prune_after_verify:
        for row in manifest["runs"]:
            ok = prune_row(args, state_dir, row) and ok

    with LockedFile(state_dir / "state.lock"):
        state = load_state(state_dir / "mirror-state.json")
    archive_counts: dict[str, int] = {}
    for row in manifest["runs"]:
        sentinel = row.get("completion_sentinel")
        if sentinel is None:
            return False
        terminal = core.completion_sentinel_checkpoint(Path(sentinel).expanduser().resolve())
        if terminal is None:
            return False
        entries = [
            entry
            for entry in state["files"].values()
            if isinstance(entry, dict) and entry.get("run_id") == row["id"]
        ]
        if not entries:
            return False
        terminal_archived = False
        archive_counts[row["id"]] = len(entries)
        for entry in entries:
            if not entry.get("remote_verified"):
                return False
            remote_path = entry.get("remote_path")
            if not isinstance(remote_path, str) or core.remote_sha(
                args.ssh, args.remote_host, remote_path
            ) != entry.get("sha256"):
                return False
            if (
                entry.get("global_step") == terminal["global_step"]
                and entry.get("sha256") == terminal["sha256"]
            ):
                terminal_archived = True
        if not terminal_archived:
            return False
    if ok:
        core.atomic_json(
            state_dir / "mirror-complete.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "manifest": str(args.manifest),
                "worker_count": args.worker_count,
                "verified_archive_counts": archive_counts,
                "completed_at_unix": time.time(),
            },
        )
    return ok


def parser() -> argparse.ArgumentParser:
    result = core.parser()
    result.description = __doc__
    result.add_argument("--worker-index", type=int, required=True)
    result.add_argument("--worker-count", type=int, required=True)
    result.add_argument("--inventory-search-root", type=Path)
    result.add_argument("--inventory-max-depth", type=int, default=6)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    global STOP
    STOP = False
    args = parser().parse_args(argv)
    if args.print_example_manifest:
        print(json.dumps(core.example_manifest(), indent=2))
        return 0
    required = (args.manifest, args.state_dir, args.scratch_root, args.quota_bytes, args.remote_host)
    if any(value is None for value in required):
        raise SystemExit("manifest, state-dir, scratch-root, quota-bytes, and remote-host are required")
    if not core.HOST_RE.fullmatch(args.remote_host):
        raise SystemExit("remote-host contains unsupported characters")
    if args.worker_count < 2 or not 0 <= args.worker_index < args.worker_count:
        raise SystemExit("worker-count must be >=2 and worker-index must be in range")
    if args.quota_bytes <= 0 or args.poll_seconds <= 0 or args.du_timeout_seconds <= 0:
        raise SystemExit("quota, poll interval, and du timeout must be positive")
    if args.inventory_max_depth < 1:
        raise SystemExit("inventory-max-depth must be positive")
    if not 0 < args.soft_used_fraction < args.hard_used_fraction < 1:
        raise SystemExit("require 0 < soft-used-fraction < hard-used-fraction < 1")

    args.state_dir = core.specific_absolute(args.state_dir, "state-dir")
    args.scratch_root = core.specific_absolute(args.scratch_root, "scratch-root")
    if args.inventory_search_root is not None:
        args.inventory_search_root = core.specific_absolute(
            args.inventory_search_root, "inventory-search-root"
        )
        core.contained(args.inventory_search_root, args.scratch_root, "inventory-search-root")
    if not args.scratch_root.is_dir():
        raise SystemExit(f"scratch-root is not a directory: {args.scratch_root}")
    core.contained(args.state_dir, args.scratch_root, "state-dir")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.manifest = args.manifest.expanduser().resolve()
    manifest = core.load_manifest(args.manifest)
    for row in manifest["runs"]:
        core.contained(
            Path(row["local_root"]).expanduser().resolve(),
            args.scratch_root,
            f"local_root[{row['id']}]",
        )

    def stop(_received: int, _frame: object) -> None:
        global STOP
        STOP = True

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not STOP:
        errors = 0
        for row in manifest["runs"]:
            for checkpoint in ordered_candidates(row, args.worker_index):
                if STOP:
                    break
                with claim(args.state_dir, checkpoint) as reservation:
                    if reservation is None:
                        continue
                    if not process_checkpoint(args, args.state_dir, row, checkpoint):
                        errors += 1
        if args.worker_index == 0:
            with LockedFile(args.state_dir / "maintenance.lock", blocking=False) as reservation:
                if reservation is not None and maintain(args, args.state_dir, manifest):
                    return 0
        if args.once:
            return 0 if errors == 0 else 1
        deadline = time.monotonic() + args.poll_seconds
        while not STOP and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
