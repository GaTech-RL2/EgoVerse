#!/usr/bin/env python3
"""Mirror validated ICE checkpoints to Skynet with verified, guarded pruning.

Validators must exit zero and emit a JSON object on their final non-empty
stdout line. The object must contain a non-negative integer ``global_step``.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

INCOMPLETE_SUFFIXES = (".tmp", ".part", ".partial", ".incomplete")
HOST_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
STOP = False


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    sha256: str
    global_step: int
    size: int
    mtime_ns: int
    inode: int
    metadata: dict[str, Any]

    def as_state(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "global_step": self.global_step,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "inode": self.inode,
            "metadata": self.metadata,
        }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_identity(stat_result: os.stat_result) -> tuple[int, int, int]:
    return (stat_result.st_size, stat_result.st_mtime_ns, stat_result.st_ino)


def specific_absolute(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise SystemExit(f"{label} must be a specific absolute path: {path}")
    resolved = expanded.resolve()
    if resolved == Path("/") or len(resolved.parts) < 4:
        raise SystemExit(f"{label} must be a specific absolute path: {resolved}")
    return resolved


def contained(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside scratch-root {root}: {path}") from exc


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or not isinstance(data.get("runs"), list):
        raise SystemExit("manifest must have schema_version=1 and a runs list")
    if not data["runs"]:
        raise SystemExit("manifest must contain at least one run")
    ids: set[str] = set()
    for row in data["runs"]:
        required = {"id", "local_root", "checkpoint_glob", "validator", "remote_dir"}
        missing = required - set(row)
        if missing:
            raise SystemExit(f"manifest row missing {sorted(missing)}")
        if row["id"] in ids or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(row["id"])):
            raise SystemExit(f"invalid or duplicate run id: {row['id']!r}")
        ids.add(row["id"])
        root = specific_absolute(Path(row["local_root"]), "local_root")
        pattern = Path(row["checkpoint_glob"])
        if pattern.is_absolute():
            try:
                pattern.parent.resolve().relative_to(root)
            except ValueError as exc:
                raise SystemExit(f"checkpoint_glob escapes local_root for {row['id']}") from exc
        validator = Path(row["validator"]).expanduser().resolve()
        if not validator.is_file() or not os.access(validator, os.X_OK):
            raise SystemExit(f"validator is not executable for {row['id']}: {validator}")
        remote = Path(row["remote_dir"])
        if not remote.is_absolute() or remote == Path("/") or len(remote.parts) < 3:
            raise SystemExit(f"remote_dir must be a specific absolute path for {row['id']}")
        if int(row.get("retain_local", 2)) < 2:
            raise SystemExit(f"retain_local must be at least 2 for {row['id']}")
        if "completion_sentinel" in row:
            sentinel = specific_absolute(Path(row["completion_sentinel"]), "completion_sentinel")
            try:
                sentinel.relative_to(root)
            except ValueError as exc:
                raise SystemExit(
                    f"completion_sentinel must be inside local_root for {row['id']}"
                ) from exc
    return data


def candidates(row: dict[str, Any]) -> list[Path]:
    root = Path(row["local_root"]).expanduser().resolve()
    pattern = row["checkpoint_glob"]
    resolved_pattern = pattern if os.path.isabs(pattern) else str(root / pattern)
    result: set[Path] = set()
    for raw in glob.glob(resolved_pattern, recursive=True):
        path = Path(raw)
        if path.name.endswith(INCOMPLETE_SUFFIXES):
            continue
        try:
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            result.add(resolved)
        except (FileNotFoundError, OSError, ValueError):
            continue
    return sorted(result, key=str)


def parse_validator_metadata(stdout: str, path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"validator emitted no metadata for {path}")
    try:
        metadata = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"validator's last non-empty line is not JSON for {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"validator metadata must be a JSON object for {path}")
    step = metadata.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"validator metadata requires non-negative integer global_step for {path}")
    if metadata.get("valid") is False:
        raise ValueError(f"validator metadata marked checkpoint invalid: {path}")
    return metadata


def info_from_cache(path: Path, prior: dict[str, Any] | None) -> CheckpointInfo | None:
    if not prior:
        return None
    try:
        current = path.stat()
    except (FileNotFoundError, OSError):
        return None
    required = ("sha256", "global_step", "size", "mtime_ns", "inode", "metadata")
    if any(key not in prior for key in required):
        return None
    if not SHA_RE.fullmatch(str(prior["sha256"])):
        return None
    if (
        current.st_size != prior["size"]
        or current.st_mtime_ns != prior["mtime_ns"]
        or current.st_ino != prior["inode"]
    ):
        return None
    step = prior["global_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    if not isinstance(prior["metadata"], dict):
        return None
    return CheckpointInfo(
        path=path,
        sha256=str(prior["sha256"]),
        global_step=step,
        size=current.st_size,
        mtime_ns=current.st_mtime_ns,
        inode=current.st_ino,
        metadata=dict(prior["metadata"]),
    )


def stable_valid_info(
    path: Path,
    validator: Path,
    *,
    cached: dict[str, Any] | None = None,
    force: bool = False,
) -> CheckpointInfo | None:
    if not force:
        cached_info = info_from_cache(path, cached)
        if cached_info is not None:
            return cached_info
    try:
        before = path.stat()
        valid = subprocess.run(
            [str(validator), str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if valid.returncode != 0:
        return None
    try:
        metadata = parse_validator_metadata(valid.stdout, path)
        digest = sha256_file(path)
        after = path.stat()
    except (ValueError, FileNotFoundError, OSError):
        return None
    if stat_identity(before) != stat_identity(after):
        return None
    expected_digest = metadata.get("sha256")
    if expected_digest is not None and str(expected_digest).lower() != digest:
        return None
    expected_path = metadata.get("checkpoint_path")
    if expected_path is not None:
        try:
            if Path(str(expected_path)).expanduser().resolve(strict=True) != path:
                return None
        except (FileNotFoundError, OSError):
            return None
    enriched = dict(metadata)
    enriched["checkpoint_path"] = str(path)
    enriched["sha256"] = digest
    return CheckpointInfo(
        path=path,
        sha256=digest,
        global_step=int(enriched["global_step"]),
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        inode=after.st_ino,
        metadata=enriched,
    )


def run_checked(
    command: list[str],
    capture: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=True,
        timeout=timeout,
    )


def remote_command(ssh: str, host: str, argv: list[str], capture: bool = False) -> str:
    result = run_checked([ssh, host, shlex.join(argv)], capture=capture)
    return (result.stdout or "").strip()


def remote_sha(ssh: str, host: str, path: str) -> str | None:
    try:
        output = remote_command(ssh, host, ["sha256sum", "--", path], capture=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    fields = output.split()
    return fields[0].lower() if fields and SHA_RE.fullmatch(fields[0].lower()) else None


def mirror_one(
    path: Path,
    digest: str,
    remote_dir: str,
    host: str,
    ssh: str,
    rsync: str,
    rsync_rsh: str | None,
    temporary_tag: str | None = None,
) -> str:
    suffix = "".join(path.suffixes) or ".checkpoint"
    base = path.name[: -len(suffix)] if suffix else path.name
    final = f"{remote_dir.rstrip('/')}/{base}.{digest}{suffix}"
    if temporary_tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", temporary_tag):
        raise ValueError("temporary_tag contains unsupported characters")
    temporary = f"{final}.partial" + (f".{temporary_tag}" if temporary_tag else "")
    existing = remote_sha(ssh, host, final)
    if existing == digest:
        try:
            remote_command(ssh, host, ["rm", "-f", "--", temporary])
        except (subprocess.CalledProcessError, OSError):
            pass
        return final
    if existing is not None:
        raise RuntimeError(f"remote immutable-path collision: {final}")
    remote_command(ssh, host, ["mkdir", "-p", "--", remote_dir])
    command = [rsync, "-a", "--partial", "--append-verify", "--protect-args"]
    if rsync_rsh:
        command.extend(["-e", rsync_rsh])
    command.extend(["--", str(path), f"{host}:{temporary}"])
    run_checked(command)
    if remote_sha(ssh, host, temporary) != digest:
        raise RuntimeError(f"remote SHA-256 mismatch for resumable copy: {temporary}")
    remote_command(ssh, host, ["mv", "-n", "--", temporary, final])
    if remote_sha(ssh, host, final) != digest:
        raise RuntimeError(f"remote SHA-256 mismatch after atomic rename: {final}")
    try:
        remote_command(ssh, host, ["rm", "-f", "--", temporary])
    except (subprocess.CalledProcessError, OSError):
        pass
    return final


def scratch_bytes(root: Path, timeout_seconds: float = 120.0) -> int:
    result = run_checked(
        ["du", "-sx", "--block-size=1", "--", str(root)],
        capture=True,
        timeout=timeout_seconds,
    )
    return int(result.stdout.split()[0])


def append_event(path: Path, payload: dict[str, Any]) -> None:
    payload = {**payload, "at_unix": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path)
    result.add_argument("--state-dir", type=Path)
    result.add_argument("--scratch-root", type=Path)
    result.add_argument("--quota-bytes", type=int)
    result.add_argument("--remote-host")
    result.add_argument("--ssh", default="ssh")
    result.add_argument("--rsync", default="rsync")
    result.add_argument(
        "--rsync-rsh",
        help="remote-shell command passed to rsync -e; use the same identity as --ssh",
    )
    result.add_argument("--poll-seconds", type=float, default=300)
    result.add_argument("--du-timeout-seconds", type=float, default=120)
    result.add_argument("--soft-used-fraction", type=float, default=0.80)
    result.add_argument("--hard-used-fraction", type=float, default=0.90)
    result.add_argument("--prune-after-verify", action="store_true")
    result.add_argument("--once", action="store_true")
    result.add_argument("--print-example-manifest", action="store_true")
    return result


def example_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runs": [
            {
                "id": "run-id",
                "local_root": "/absolute/ice/scratch/run",
                "checkpoint_glob": "checkpoints/*.ckpt",
                "validator": "/absolute/path/validate_checkpoint",
                "remote_dir": "/absolute/skynet/archive/run/checkpoints",
                "retain_local": 2,
                "completion_sentinel": "/absolute/ice/scratch/run/COMPLETE.json",
            }
        ],
    }


def state_entry(info: CheckpointInfo, row_id: str, prior: dict[str, Any] | None) -> dict[str, Any]:
    result = {**info.as_state(), "run_id": row_id}
    if prior and prior.get("sha256") == info.sha256:
        for key in ("remote_path", "remote_verified", "verified_at_unix"):
            if key in prior:
                result[key] = prior[key]
    return result


def completion_sentinel_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != 1 or payload.get("status") != "COMPLETE":
        return None
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    step = checkpoint.get("global_step")
    digest = checkpoint.get("sha256")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or not isinstance(digest, str)
        or SHA_RE.fullmatch(digest) is None
    ):
        return None
    return checkpoint


def completion_sentinel_is_valid(path: Path) -> bool:
    return completion_sentinel_checkpoint(path) is not None


def main(argv: Sequence[str] | None = None) -> int:
    global STOP
    STOP = False
    args = parser().parse_args(argv)
    if args.print_example_manifest:
        print(json.dumps(example_manifest(), indent=2))
        return 0
    required = (args.manifest, args.state_dir, args.scratch_root, args.quota_bytes, args.remote_host)
    if any(value is None for value in required):
        raise SystemExit("manifest, state-dir, scratch-root, quota-bytes, and remote-host are required")
    if not HOST_RE.fullmatch(args.remote_host):
        raise SystemExit("remote-host contains unsupported characters")
    if args.quota_bytes <= 0 or args.poll_seconds <= 0 or args.du_timeout_seconds <= 0:
        raise SystemExit("quota, poll interval, and du timeout must be positive")
    if not 0 < args.soft_used_fraction < args.hard_used_fraction < 1:
        raise SystemExit("require 0 < soft-used-fraction < hard-used-fraction < 1")
    state_dir = specific_absolute(args.state_dir, "state-dir")
    scratch_root = specific_absolute(args.scratch_root, "scratch-root")
    if not scratch_root.is_dir():
        raise SystemExit(f"scratch-root is not a directory: {scratch_root}")
    contained(state_dir, scratch_root, "state-dir")
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest.expanduser().resolve())
    for row in manifest["runs"]:
        contained(Path(row["local_root"]).expanduser().resolve(), scratch_root, f"local_root[{row['id']}]")
    state_path = state_dir / "mirror-state.json"
    events_path = state_dir / "mirror-events.jsonl"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid mirror state: {state_path}: {exc}") from exc
    else:
        state = {"schema_version": 2, "files": {}}
    if not isinstance(state.get("files"), dict):
        raise SystemExit(f"invalid mirror state files map: {state_path}")
    state["schema_version"] = 2

    lock_stream = (state_dir / "mirror.lock").open("a+")
    try:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another checkpoint mirror owns this state directory") from exc

        def stop(_received: int, _frame: object) -> None:
            global STOP
            STOP = True

        signal.signal(signal.SIGUSR1, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

        while not STOP:
            cycle_errors = 0
            unresolved = False
            all_rows_declare_completion = True
            all_rows_complete = True
            for row in manifest["runs"]:
                row_candidates = candidates(row)
                validator = Path(row["validator"]).expanduser().resolve()
                infos: list[CheckpointInfo] = []
                for checkpoint in row_candidates:
                    prior = state["files"].get(str(checkpoint))
                    info = stable_valid_info(checkpoint, validator, cached=prior)
                    if info is None:
                        unresolved = True
                        append_event(
                            events_path,
                            {
                                "event": "checkpoint_not_stably_valid",
                                "run_id": row["id"],
                                "path": str(checkpoint),
                            },
                        )
                        continue
                    infos.append(info)
                    state["files"][str(checkpoint)] = state_entry(info, row["id"], prior)
                infos.sort(
                    key=lambda item: (item.global_step, item.mtime_ns, str(item.path)),
                    reverse=True,
                )
                atomic_json(state_path, state)

                for info in infos:
                    prior = state["files"][str(info.path)]
                    if prior.get("remote_verified") and prior.get("sha256") == info.sha256:
                        remote_path = prior.get("remote_path")
                        if remote_path and remote_sha(args.ssh, args.remote_host, remote_path) == info.sha256:
                            continue
                        prior["remote_verified"] = False
                        prior["remote_recheck_failed_at_unix"] = time.time()
                        atomic_json(state_path, state)
                    try:
                        remote_path = mirror_one(
                            info.path,
                            info.sha256,
                            row["remote_dir"],
                            args.remote_host,
                            args.ssh,
                            args.rsync,
                            args.rsync_rsh,
                        )
                        prior.update(
                            {
                                "remote_path": remote_path,
                                "remote_verified": True,
                                "verified_at_unix": time.time(),
                            }
                        )
                        atomic_json(state_path, state)
                        append_event(
                            events_path,
                            {
                                "event": "mirrored",
                                "run_id": row["id"],
                                "path": str(info.path),
                                "global_step": info.global_step,
                                "sha256": info.sha256,
                                "remote_path": remote_path,
                            },
                        )
                    except Exception as exc:
                        cycle_errors += 1
                        unresolved = True
                        append_event(
                            events_path,
                            {
                                "event": "mirror_error",
                                "run_id": row["id"],
                                "path": str(info.path),
                                "error": str(exc),
                            },
                        )
                        print(f"mirror error for {row['id']}: {exc}", file=sys.stderr, flush=True)

                if args.prune_after_verify:
                    retain = int(row.get("retain_local", 2))
                    for info in infos[retain:]:
                        prior = state["files"].get(str(info.path))
                        if not prior or not prior.get("remote_verified"):
                            continue
                        current = stable_valid_info(info.path, validator, force=True)
                        if current is None or current.sha256 != prior.get("sha256"):
                            unresolved = True
                            continue
                        remote_path = prior.get("remote_path")
                        remote_digest = (
                            remote_sha(args.ssh, args.remote_host, remote_path)
                            if remote_path
                            else None
                        )
                        if remote_digest != current.sha256:
                            prior["remote_verified"] = False
                            prior["remote_recheck_failed_at_unix"] = time.time()
                            atomic_json(state_path, state)
                            cycle_errors += 1
                            unresolved = True
                            append_event(
                                events_path,
                                {
                                    "event": "prune_refused_remote_sha",
                                    "run_id": row["id"],
                                    "path": str(info.path),
                                    "expected_sha256": current.sha256,
                                    "observed_remote_sha256": remote_digest,
                                },
                            )
                            continue
                        try:
                            final_stat = info.path.stat()
                        except (FileNotFoundError, OSError):
                            unresolved = True
                            continue
                        if stat_identity(final_stat) != (current.size, current.mtime_ns, current.inode):
                            unresolved = True
                            continue
                        info.path.unlink()
                        append_event(
                            events_path,
                            {
                                "event": "pruned",
                                "run_id": row["id"],
                                "path": str(info.path),
                                "global_step": current.global_step,
                                "sha256": current.sha256,
                            },
                        )

                sentinel_raw = row.get("completion_sentinel")
                if sentinel_raw is None:
                    all_rows_declare_completion = False
                elif not completion_sentinel_is_valid(Path(sentinel_raw).expanduser().resolve()):
                    all_rows_complete = False

            try:
                used = scratch_bytes(scratch_root, args.du_timeout_seconds)
                fraction = used / args.quota_bytes
                pressure = (
                    "hard"
                    if fraction >= args.hard_used_fraction
                    else "soft"
                    if fraction >= args.soft_used_fraction
                    else "normal"
                )
                pressure_payload: dict[str, Any] = {
                    "scratch_root": str(scratch_root),
                    "used_bytes": used,
                    "quota_bytes": args.quota_bytes,
                    "used_fraction": fraction,
                    "pressure": pressure,
                    "cycle_errors": cycle_errors,
                    "updated_at_unix": time.time(),
                }
            except Exception as exc:
                cycle_errors += 1
                unresolved = True
                pressure = "unknown"
                pressure_payload = {
                    "scratch_root": str(scratch_root),
                    "quota_bytes": args.quota_bytes,
                    "pressure": "unknown",
                    "error": str(exc),
                    "cycle_errors": cycle_errors,
                    "updated_at_unix": time.time(),
                }
                append_event(events_path, {"event": "scratch_usage_error", "error": str(exc)})
            atomic_json(state_dir / "storage-pressure.json", pressure_payload)
            if pressure != "normal":
                print(f"ICE scratch pressure={pressure}: {pressure_payload}", file=sys.stderr, flush=True)

            archive_counts: dict[str, int] = {}
            if all_rows_declare_completion and all_rows_complete and not unresolved and cycle_errors == 0:
                for row in manifest["runs"]:
                    entries = [
                        entry
                        for entry in state["files"].values()
                        if entry.get("run_id") == row["id"]
                    ]
                    archive_counts[row["id"]] = len(entries)
                    if not entries:
                        unresolved = True
                        append_event(
                            events_path,
                            {"event": "completion_refused_no_archived_checkpoint", "run_id": row["id"]},
                        )
                        continue
                    for entry in entries:
                        expected = entry.get("sha256")
                        remote_path = entry.get("remote_path")
                        observed = (
                            remote_sha(args.ssh, args.remote_host, remote_path)
                            if remote_path and isinstance(expected, str)
                            else None
                        )
                        if observed != expected:
                            entry["remote_verified"] = False
                            entry["remote_recheck_failed_at_unix"] = time.time()
                            cycle_errors += 1
                            unresolved = True
                            append_event(
                                events_path,
                                {
                                    "event": "completion_refused_remote_sha",
                                    "run_id": row["id"],
                                    "remote_path": remote_path,
                                    "expected_sha256": expected,
                                    "observed_remote_sha256": observed,
                                },
                            )
                atomic_json(state_path, state)
            if all_rows_declare_completion and all_rows_complete and not unresolved and cycle_errors == 0:
                atomic_json(
                    state_dir / "mirror-complete.json",
                    {
                        "schema_version": 1,
                        "status": "COMPLETE",
                        "manifest": str(args.manifest.expanduser().resolve()),
                        "verified_archive_counts": archive_counts,
                        "completed_at_unix": time.time(),
                    },
                )
                return 0
            if args.once:
                return 0 if cycle_errors == 0 and not unresolved else 1
            deadline = time.monotonic() + args.poll_seconds
            while not STOP and time.monotonic() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        return 0
    finally:
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
