#!/usr/bin/env python3
"""Checkpoint-aware Slurm requeue runner for PACE ICE.

The checkpoint validator must exit zero and write a JSON object on its last
non-empty stdout line. The object must contain a non-negative integer
``global_step``. The runner computes SHA-256 itself and enriches the metadata.

The child receives ICE_RESUME_CHECKPOINT, ICE_RESUME_CHECKPOINT_SHA256,
ICE_RESUME_GLOBAL_STEP, ICE_RESUME_CHECKPOINT_METADATA_JSON,
ICE_RESTART_COUNT, ICE_ATTEMPT_DIR, and ICE_REQUEUE_OWNER.
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
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

INCOMPLETE_SUFFIXES = (".tmp", ".part", ".partial", ".incomplete")
STEP_RE = re.compile(r"(?:^|[^A-Za-z])(?:global[_-]?step|step)[=_-]?(\d+)", re.IGNORECASE)
IDENTITY_KEYS = (
    "run_id",
    "wandb_run_id",
    "wandb_entity",
    "wandb_project",
    "config_sha256",
    "source_sha256",
    "source_commit",
)
OBSERVATION_INDEX_NAME = "checkpoint-observations.json"
OBSERVATION_INDEX_SCHEMA_VERSION = 1
LAUNCH_GATE_CODE = """\
import os
import sys

fd = int(os.environ.pop("ICE_INTERNAL_LAUNCH_GATE_FD"))
try:
    token = os.read(fd, 1)
finally:
    os.close(fd)
if token != b"G":
    raise SystemExit(125)
os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
"""


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    sha256: str
    global_step: int
    size: int
    mtime_ns: int
    inode: int
    metadata: dict[str, Any]
    device: int = 0
    ctime_ns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
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


def atomic_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Publish a JSON proof exactly once without an overwrite window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SystemExit(f"immutable proof already exists: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def checkpoint_candidates(patterns: str | Sequence[str], state_dir: Path) -> list[Path]:
    if isinstance(patterns, str):
        patterns = [patterns]
    unique: set[Path] = set()
    for pattern in patterns:
        resolved_pattern = pattern if os.path.isabs(pattern) else str(state_dir / pattern)
        for raw in glob.glob(resolved_pattern, recursive=True):
            path = Path(raw)
            if path.name.endswith(INCOMPLETE_SUFFIXES):
                continue
            try:
                if path.is_file():
                    unique.add(path.resolve(strict=True))
            except (FileNotFoundError, OSError):
                continue
    return sorted(unique, key=str)


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


def unvalidated_metadata(path: Path) -> dict[str, Any] | None:
    matches = list(STEP_RE.finditer(path.name))
    if not matches:
        return None
    return {
        "global_step": int(matches[-1].group(1)),
        "validation": "UNVALIDATED_EXPLICIT_OPT_IN",
    }


def stable_checkpoint_info(
    path: Path,
    validator: Path | None,
    *,
    allow_unvalidated: bool = False,
    cache: dict[tuple[str, int, int, int, int, int], CheckpointInfo] | None = None,
) -> CheckpointInfo | None:
    try:
        before = path.stat()
    except (FileNotFoundError, OSError):
        return None
    cache_key = (str(path), *stat_identity(before))
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if validator is None:
        if not allow_unvalidated:
            return None
        metadata = unvalidated_metadata(path)
        if metadata is None:
            return None
    else:
        try:
            result = subprocess.run(
                [str(validator), str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        try:
            metadata = parse_validator_metadata(result.stdout, path)
        except ValueError:
            return None
    try:
        digest = sha256_file(path)
        after = path.stat()
    except (FileNotFoundError, OSError):
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
    info = CheckpointInfo(
        path=path,
        sha256=digest,
        global_step=int(enriched["global_step"]),
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        inode=after.st_ino,
        metadata=enriched,
        device=after.st_dev,
        ctime_ns=after.st_ctime_ns,
    )
    if cache is not None:
        cache[cache_key] = info
    return info


def select_checkpoint(
    patterns: str | Sequence[str],
    state_dir: Path,
    validator: Path | None,
    *,
    allow_unvalidated: bool = False,
    cache: dict[tuple[str, int, int, int, int, int], CheckpointInfo] | None = None,
) -> CheckpointInfo | None:
    validated = [
        info
        for candidate in checkpoint_candidates(patterns, state_dir)
        if (
            info := stable_checkpoint_info(
                candidate,
                validator,
                allow_unvalidated=allow_unvalidated,
                cache=cache,
            )
        )
        is not None
    ]
    if not validated:
        return None
    return newest_checkpoint(*validated)


def stable_sha256_file(path: Path) -> str:
    """Hash one small authority/state file while rejecting concurrent change."""
    try:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(f"could not hash stable file {path}: {exc}") from exc
    if stat_identity(before) != stat_identity(after):
        raise SystemExit(f"file changed while being hashed: {path}")
    return digest


def stable_file_bytes(path: Path) -> bytes:
    """Read and authenticate the exact bytes later parsed by the caller."""
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(f"could not read stable file {path}: {exc}") from exc
    if stat_identity(before) != stat_identity(after):
        raise SystemExit(f"file changed while being read: {path}")
    return payload


def validation_authority(
    validator: Path | None,
    *,
    allow_unvalidated: bool,
) -> dict[str, Any]:
    if validator is None:
        if not allow_unvalidated:
            raise SystemExit("checkpoint validator authority is unavailable")
        return {"mode": "UNVALIDATED_EXPLICIT_OPT_IN"}
    return {
        "mode": "validator",
        "path": str(validator),
        "sha256": stable_sha256_file(validator),
    }


def checkpoint_observation(info: CheckpointInfo) -> dict[str, Any]:
    return {
        "path": str(info.path),
        "device": info.device,
        "inode": info.inode,
        "size": info.size,
        "mtime_ns": info.mtime_ns,
        "ctime_ns": info.ctime_ns,
        "global_step": info.global_step,
        "sha256": info.sha256,
        "metadata_identity": {
            key: info.metadata[key]
            for key in IDENTITY_KEYS
            if key in info.metadata
        },
    }


def _validated_observation(raw: Any) -> dict[str, Any]:
    required_keys = {
        "path",
        "device",
        "inode",
        "size",
        "mtime_ns",
        "ctime_ns",
        "global_step",
        "sha256",
        "metadata_identity",
    }
    if not isinstance(raw, dict) or set(raw) != required_keys:
        raise SystemExit("checkpoint observation has non-canonical keys")
    path = raw["path"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise SystemExit("checkpoint observation path is not absolute")
    for key in ("device", "inode", "size", "mtime_ns", "ctime_ns", "global_step"):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SystemExit(f"checkpoint observation has invalid {key}")
    digest = raw["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SystemExit("checkpoint observation has invalid sha256")
    identity = raw["metadata_identity"]
    if not isinstance(identity, dict) or any(key not in IDENTITY_KEYS for key in identity):
        raise SystemExit("checkpoint observation has invalid metadata identity")
    return raw


def observation_matches_path(observation: dict[str, Any], path: Path) -> bool:
    try:
        current = path.stat()
    except (FileNotFoundError, OSError):
        return False
    expected = (
        observation["device"],
        observation["inode"],
        observation["size"],
        observation["mtime_ns"],
        observation["ctime_ns"],
    )
    return stat_identity(current) == expected


def load_checkpoint_observations(
    path: Path,
    high_water: dict[str, Any] | None,
    authority: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load only an index content-bound by the authenticated high-water record."""
    if high_water is None:
        return {}
    expected_digest = high_water.get("observation_index_sha256")
    if not isinstance(expected_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_digest
    ) is None:
        return {}
    if not path.is_file():
        return {}
    raw_payload = stable_file_bytes(path)
    if hashlib.sha256(raw_payload).hexdigest() != expected_digest:
        # An interrupted index-before-high-water commit is recoverable by a full
        # validator-authoritative rescan; never trust the unbound bytes.
        return {}
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid checkpoint observation index: {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "validation_authority",
        "entries",
    }:
        raise SystemExit(f"non-canonical checkpoint observation index: {path}")
    if payload["schema_version"] != OBSERVATION_INDEX_SCHEMA_VERSION:
        raise SystemExit(f"unsupported checkpoint observation index schema: {path}")
    if payload["validation_authority"] != authority:
        return {}
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise SystemExit(f"checkpoint observation entries are not a list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for raw in entries:
        entry = _validated_observation(raw)
        entry_path = entry["path"]
        if entry_path in result:
            raise SystemExit(f"duplicate checkpoint observation path: {entry_path}")
        result[entry_path] = entry
    return result


def write_checkpoint_observations(
    path: Path,
    authority: dict[str, Any],
    observations: dict[str, dict[str, Any]],
) -> str:
    atomic_json(
        path,
        {
            "schema_version": OBSERVATION_INDEX_SCHEMA_VERSION,
            "validation_authority": authority,
            "entries": [observations[key] for key in sorted(observations)],
        },
    )
    return stable_sha256_file(path)


def authenticate_high_water(
    high_water: dict[str, Any] | None,
    validator: Path | None,
    *,
    allow_unvalidated: bool,
    cache: dict[tuple[str, int, int, int, int, int], CheckpointInfo],
) -> CheckpointInfo | None:
    if high_water is None:
        return None
    path = Path(high_water["path"])
    if not path.exists():
        return None
    info = stable_checkpoint_info(
        path,
        validator,
        allow_unvalidated=allow_unvalidated,
        cache=cache,
    )
    if info is None:
        raise SystemExit("recorded high-water checkpoint failed application validation")
    enforce_high_water(info, high_water)
    return info


def discover_checkpoint_incrementally(
    patterns: str | Sequence[str],
    state_dir: Path,
    validator: Path | None,
    *,
    allow_unvalidated: bool,
    cache: dict[tuple[str, int, int, int, int, int], CheckpointInfo],
    high_water: dict[str, Any] | None,
    observations: dict[str, dict[str, Any]],
    prevalidated: Sequence[CheckpointInfo | None] = (),
) -> tuple[CheckpointInfo | None, dict[str, dict[str, Any]]]:
    """Validate only new/changed files while preserving the semantic maximum.

    The durable observation index is itself SHA-bound by high-water. Unchanged
    entries at or below high-water were already validator-authorized and cannot
    become the semantic maximum. The exact high-water is freshly authenticated
    separately; every unknown or changed path is sent through the validator.
    """
    available = [info for info in prevalidated if info is not None]
    prevalidated_by_path = {str(info.path): info for info in available}
    candidate_paths = checkpoint_candidates(patterns, state_dir)
    current_paths = {str(path) for path in candidate_paths} | set(prevalidated_by_path)
    next_observations = {
        key: value for key, value in observations.items() if key in current_paths
    }
    for info in available:
        next_observations[str(info.path)] = checkpoint_observation(info)

    authenticated_floor = any(
        high_water is not None
        and info.global_step == high_water["global_step"]
        and info.sha256 == high_water["sha256"]
        for info in available
    )
    for path in candidate_paths:
        path_text = str(path)
        if path_text in prevalidated_by_path:
            continue
        observed = observations.get(path_text)
        if observed is not None and observation_matches_path(observed, path):
            if high_water is not None:
                if observed["global_step"] > high_water["global_step"]:
                    raise SystemExit(
                        "checkpoint observation exceeds its SHA-bound high-water"
                    )
                if observed["global_step"] == high_water["global_step"]:
                    if observed["sha256"] != high_water["sha256"]:
                        raise SystemExit(
                            "checkpoint identity fork refused: same-step observation "
                            "differs from SHA-recorded high-water"
                        )
                    synthetic = {"metadata": observed["metadata_identity"]}
                    mismatches = identity_mismatches(high_water, synthetic)
                    if mismatches:
                        raise SystemExit(
                            "checkpoint identity mismatch for observed high-water keys: "
                            + ", ".join(mismatches)
                        )
                    if not authenticated_floor:
                        # The recorded path may have been offloaded. Authenticate
                        # one byte-identical surviving copy before resuming.
                        observed = None
            if observed is not None:
                continue

        # A changed positive observation has lost its authority. Do not retain
        # stale state when its replacement currently fails validation; retry it
        # on later scans because an atomic proof sidecar may still appear.
        next_observations.pop(path_text, None)
        info = stable_checkpoint_info(
            path,
            validator,
            allow_unvalidated=allow_unvalidated,
            cache=cache,
        )
        if info is None:
            continue
        next_observations[path_text] = checkpoint_observation(info)
        if high_water is not None:
            if info.global_step == high_water["global_step"] and (
                info.sha256 != high_water["sha256"]
            ):
                raise SystemExit(
                    "checkpoint identity fork refused: same-step checkpoint differs "
                    "from SHA-recorded high-water"
                )
            if info.global_step >= high_water["global_step"]:
                enforce_high_water(info, high_water)
            if (
                info.global_step == high_water["global_step"]
                and info.sha256 == high_water["sha256"]
            ):
                authenticated_floor = True
        available.append(info)

    return newest_checkpoint(*available), next_observations


def newest_checkpoint(*checkpoints: CheckpointInfo | None) -> CheckpointInfo | None:
    """Return the semantic maximum without treating path/mtime as progress."""
    available = [checkpoint for checkpoint in checkpoints if checkpoint is not None]
    if not available:
        return None
    maximum_step = max(item.global_step for item in available)
    maximum_digests = {
        item.sha256 for item in available if item.global_step == maximum_step
    }
    if len(maximum_digests) != 1:
        raise SystemExit(
            "checkpoint identity fork refused: semantic-maximum checkpoints at "
            f"step {maximum_step} have different SHA-256 identities"
        )
    return max(
        available,
        key=lambda item: (item.global_step, item.mtime_ns, str(item.path)),
    )


def parse_signal(value: str) -> signal.Signals:
    name = value if value.startswith("SIG") else f"SIG{value}"
    try:
        return signal.Signals(getattr(signal, name).value)
    except (AttributeError, ValueError, KeyError) as exc:
        raise argparse.ArgumentTypeError(f"unknown signal: {value}") from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--state-dir", type=Path, required=True)
    result.add_argument(
        "--initial-checkpoint",
        type=Path,
        help=(
            "exact staged checkpoint used to seed a first migration attempt; "
            "never inferred from the live checkpoint glob"
        ),
    )
    result.add_argument(
        "--checkpoint-glob",
        action="append",
        required=True,
        help="live checkpoint glob used for boundary saves and restart discovery; repeat as needed",
    )
    result.add_argument("--checkpoint-validator", type=Path)
    result.add_argument("--allow-unvalidated-checkpoints", action="store_true")
    result.add_argument("--require-initial-checkpoint", action="store_true")
    result.add_argument(
        "--checkpoint-signal",
        type=parse_signal,
        required=True,
        help="save-only signal forwarded to Slurm job steps at the Slurm USR1 boundary",
    )
    result.add_argument(
        "--checkpoint-forwarding",
        choices=("slurm-steps",),
        required=True,
        help="explicit checkpoint signal transport; slurm-steps excludes the batch shell",
    )
    result.add_argument("--checkpoint-grace-seconds", type=int, default=240)
    result.add_argument("--poll-seconds", type=float, default=2.0)
    result.add_argument("--max-restarts", type=int, default=32)
    result.add_argument(
        "--requeue-owner",
        choices=("runner", "child"),
        required=True,
        help="exactly one owner may invoke scontrol requeue",
    )
    result.add_argument(
        "--confirm-child-requeue-disabled",
        action="store_true",
        help="required for runner ownership; confirms framework auto-requeue is disabled",
    )
    result.add_argument("--scontrol", default="scontrol")
    result.add_argument("--scancel", default="scancel")
    result.add_argument("--completion-sentinel", type=Path)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def absolute_specific(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise SystemExit(f"{label} must be an absolute persistent path")
    resolved = expanded.resolve()
    if resolved == Path("/") or len(resolved.parts) < 4:
        raise SystemExit(f"{label} must be a specific absolute persistent path")
    return resolved


def load_high_water(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid checkpoint high-water record: {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise SystemExit(f"unsupported checkpoint high-water schema: {path}")
    step = payload.get("global_step")
    digest = payload.get("sha256")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise SystemExit(f"invalid global_step in checkpoint high-water record: {path}")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit(f"invalid sha256 in checkpoint high-water record: {path}")
    checkpoint_path = payload.get("path")
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
        raise SystemExit(f"invalid path in checkpoint high-water record: {path}")
    observation_digest = payload.get("observation_index_sha256")
    if observation_digest is not None and (
        not isinstance(observation_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", observation_digest) is None
    ):
        raise SystemExit(
            f"invalid observation index SHA-256 in checkpoint high-water record: {path}"
        )
    return payload


def identity_mismatches(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    old_metadata = old.get("metadata", {})
    new_metadata = new.get("metadata", {})
    return [
        key
        for key in IDENTITY_KEYS
        if key in old_metadata
        and (key not in new_metadata or old_metadata[key] != new_metadata[key])
    ]


def enforce_high_water(info: CheckpointInfo | None, high_water: dict[str, Any] | None) -> None:
    if high_water is None:
        return
    if info is None:
        raise SystemExit("no valid checkpoint remains, but a checkpoint high-water record exists")
    prior_step = int(high_water["global_step"])
    if info.global_step < prior_step:
        raise SystemExit(
            f"checkpoint rollback refused: selected step {info.global_step} < high-water {prior_step}"
        )
    if info.global_step == prior_step and info.sha256 != high_water["sha256"]:
        raise SystemExit(
            "checkpoint identity fork refused: same-step checkpoint differs from SHA-recorded high-water"
        )
    mismatches = identity_mismatches(high_water, info.as_dict())
    if mismatches:
        raise SystemExit(f"checkpoint identity mismatch for high-water keys: {', '.join(mismatches)}")


def write_high_water(
    path: Path,
    info: CheckpointInfo,
    *,
    observation_index_sha256: str | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        **info.as_dict(),
        "updated_at_unix": time.time(),
    }
    if observation_index_sha256 is not None:
        if re.fullmatch(r"[0-9a-f]{64}", observation_index_sha256) is None:
            raise ValueError("invalid checkpoint observation index SHA-256")
        payload["observation_index_sha256"] = observation_index_sha256
    atomic_json(path, payload)


def commit_checkpoint_state(
    *,
    high_water_path: Path,
    observation_index_path: Path,
    info: CheckpointInfo,
    authority: dict[str, Any],
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if authority.get("mode") == "validator":
        current_authority = validation_authority(
            Path(authority["path"]),
            allow_unvalidated=False,
        )
        if current_authority != authority:
            raise SystemExit(
                "checkpoint validator changed after validation; refusing state commit"
            )
    # Write the index first and bind its exact bytes from high-water second. A
    # crash between these atomic writes only causes a safe full rescan.
    observation_digest = write_checkpoint_observations(
        observation_index_path,
        authority,
        observations,
    )
    write_high_water(
        high_water_path,
        info,
        observation_index_sha256=observation_digest,
    )
    reloaded = load_high_water(high_water_path)
    if reloaded is None:
        raise SystemExit("checkpoint high-water commit disappeared")
    return reloaded


def is_fresh(candidate: CheckpointInfo | None, baseline: CheckpointInfo | None) -> bool:
    if candidate is None:
        return False
    if baseline is None:
        return True
    return candidate.global_step > baseline.global_step


def forward_checkpoint_to_slurm_steps(
    scancel: str,
    job_id: str,
    checkpoint_signal: signal.Signals,
) -> subprocess.CompletedProcess[str]:
    """Signal Slurm job steps, deliberately excluding the batch shell/runner."""
    if checkpoint_signal != signal.SIGUSR2:
        raise ValueError("slurm-step checkpoint forwarding currently requires SIGUSR2")
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SCANCEL_") and key != "SLURM_CLUSTERS"
    }
    return subprocess.run(
        [scancel, "--signal=USR2", job_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=clean_env,
    )


def resolve_executable(value: str) -> str:
    if os.path.sep in value:
        resolved = str(Path(value).expanduser().resolve())
        if not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
            raise SystemExit(f"executable is unavailable: {resolved}")
        return resolved
    resolved = shutil.which(value)
    if resolved is None:
        raise SystemExit(f"executable is unavailable on PATH: {value}")
    return resolved


def validated_completion_sentinel(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid completion sentinel: {path}: {exc}") from exc
    if payload.get("schema_version") != 1 or payload.get("status") != "COMPLETE":
        raise SystemExit(f"completion sentinel is not a schema-1 COMPLETE record: {path}")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"completion sentinel lacks checkpoint identity: {path}")
    step = checkpoint.get("global_step")
    digest = checkpoint.get("sha256")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise SystemExit(f"completion sentinel has invalid global_step: {path}")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit(f"completion sentinel has invalid checkpoint SHA-256: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    # Slurm may deliver the pre-timeout signal while strict checkpoint
    # discovery is still running. Install a minimal in-memory latch before any
    # filesystem discovery or validator execution. A prelaunch boundary uses
    # the authenticated no-work requeue path below; TERM/INT take priority.
    boundary_requested_at: float | None = None
    boundary_processed = False
    termination_signal: signal.Signals | None = None
    handled_signals = {signal.SIGUSR1, signal.SIGTERM, signal.SIGINT}
    launch_gate_write_fd: int | None = None
    workload_launch_committed = False

    def close_launch_gate() -> None:
        nonlocal launch_gate_write_fd
        descriptor = launch_gate_write_fd
        launch_gate_write_fd = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def handler(received: int, _frame: object) -> None:
        nonlocal boundary_requested_at, boundary_processed, termination_signal
        if not workload_launch_committed:
            # Closing the one-byte launch gate makes a signal racing with
            # Popen fail before the wrapper can exec the real workload.
            close_launch_gate()
        received_signal = signal.Signals(received)
        if received_signal == signal.SIGUSR1:
            if boundary_requested_at is None and not boundary_processed:
                boundary_requested_at = time.time()
        else:
            termination_signal = received_signal

    if not args.dry_run:
        signal.signal(signal.SIGUSR1, handler)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        # A parent shell must not be able to leave the runner deaf to Slurm's
        # boundary/termination contract through an inherited signal mask.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, handled_signals)

    if not args.command:
        raise SystemExit("a child command is required after --")
    if args.checkpoint_grace_seconds < 1 or args.poll_seconds <= 0:
        raise SystemExit("grace and poll intervals must be positive")
    if args.max_restarts < 0:
        raise SystemExit("max restarts must be nonnegative")
    if args.requeue_owner == "runner" and not args.confirm_child_requeue_disabled:
        raise SystemExit(
            "runner ownership requires --confirm-child-requeue-disabled; disable framework auto-requeue"
        )
    if args.checkpoint_forwarding == "slurm-steps" and args.checkpoint_signal != signal.SIGUSR2:
        raise SystemExit("slurm-step checkpoint forwarding requires --checkpoint-signal USR2")
    if args.require_initial_checkpoint and args.initial_checkpoint is None:
        raise SystemExit("--require-initial-checkpoint requires an exact --initial-checkpoint")

    state_dir = absolute_specific(args.state_dir, "state-dir")
    state_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(state_dir, os.W_OK):
        raise SystemExit(f"state-dir is not writable: {state_dir}")
    completion_sentinel = (
        absolute_specific(args.completion_sentinel, "completion-sentinel")
        if args.completion_sentinel is not None
        else None
    )
    initial_checkpoint_path = (
        absolute_specific(args.initial_checkpoint, "initial-checkpoint")
        if args.initial_checkpoint is not None
        else None
    )

    validator = args.checkpoint_validator
    if validator is None and not args.allow_unvalidated_checkpoints:
        raise SystemExit("checkpoint-validator is required for safe requeue")
    if validator is not None:
        validator = validator.expanduser().resolve()
        if not validator.is_file() or not os.access(validator, os.X_OK):
            raise SystemExit(f"validator must be executable: {validator}")

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id and not args.dry_run:
        raise SystemExit("SLURM_JOB_ID is required outside --dry-run")
    job_id = job_id or "dry-run"
    try:
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    except ValueError as exc:
        raise SystemExit("SLURM_RESTART_COUNT must be an integer") from exc
    if restart_count < 0 or restart_count > args.max_restarts:
        raise SystemExit(
            f"restart count {restart_count} is outside allowed range 0..{args.max_restarts}"
        )
    if validator is None and (args.require_initial_checkpoint or restart_count > 0):
        raise SystemExit("validated checkpoint metadata is mandatory for migration and restart")

    if completion_sentinel is not None and completion_sentinel.exists():
        validated_completion_sentinel(completion_sentinel)
        print(json.dumps({"status": "ALREADY_COMPLETE", "sentinel": str(completion_sentinel)}))
        return 0

    lock_stream = (state_dir / "runner.lock").open("a+")
    try:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another requeue runner owns this state directory") from exc

        scancel = None
        if not args.dry_run:
            scancel = resolve_executable(args.scancel)
        scontrol = None
        if args.requeue_owner == "runner" and not args.dry_run:
            scontrol = resolve_executable(args.scontrol)

        high_water_path = state_dir / "checkpoint-high-water.json"
        observation_index_path = state_dir / OBSERVATION_INDEX_NAME
        high_water = load_high_water(high_water_path)
        if high_water is not None and validator is None:
            raise SystemExit("validated checkpoint metadata is mandatory when high-water state exists")
        authority = validation_authority(
            validator,
            allow_unvalidated=args.allow_unvalidated_checkpoints,
        )
        observations = load_checkpoint_observations(
            observation_index_path,
            high_water,
            authority,
        )
        validation_cache: dict[
            tuple[str, int, int, int, int, int], CheckpointInfo
        ] = {}
        authenticated_high_water = authenticate_high_water(
            high_water,
            validator,
            allow_unvalidated=args.allow_unvalidated_checkpoints,
            cache=validation_cache,
        )
        initial_checkpoint = None
        if initial_checkpoint_path is not None and initial_checkpoint_path.exists():
            if initial_checkpoint_path.name.endswith(INCOMPLETE_SUFFIXES):
                raise SystemExit(f"initial checkpoint is incomplete: {initial_checkpoint_path}")
            if not initial_checkpoint_path.is_file():
                raise SystemExit(f"initial checkpoint is not a regular file: {initial_checkpoint_path}")
            initial_checkpoint = stable_checkpoint_info(
                initial_checkpoint_path.resolve(strict=True),
                validator,
                allow_unvalidated=args.allow_unvalidated_checkpoints,
                cache=validation_cache,
            )
            if initial_checkpoint is None:
                raise SystemExit(
                    f"initial checkpoint failed stable application validation: {initial_checkpoint_path}"
                )
        elif initial_checkpoint_path is not None and high_water is None:
            raise SystemExit(f"initial checkpoint is unavailable: {initial_checkpoint_path}")

        # On a new migration state, pin startup to the exact staged checkpoint.
        # Live checkpoint discovery is enabled only after that seed has been
        # authenticated into the high-water record. Established states may
        # resume from a newer live checkpoint even if the staged seed is gone.
        if high_water is None and initial_checkpoint is not None:
            checkpoint = initial_checkpoint
            observations = {
                str(initial_checkpoint.path): checkpoint_observation(initial_checkpoint)
            }
        else:
            live_checkpoint, observations = discover_checkpoint_incrementally(
                args.checkpoint_glob,
                state_dir,
                validator,
                allow_unvalidated=args.allow_unvalidated_checkpoints,
                cache=validation_cache,
                high_water=high_water,
                observations=observations,
                prevalidated=(authenticated_high_water, initial_checkpoint),
            )
            checkpoint = newest_checkpoint(
                authenticated_high_water,
                initial_checkpoint,
                live_checkpoint,
            )
        enforce_high_water(checkpoint, high_water)
        if checkpoint is None and (
            args.require_initial_checkpoint or restart_count > 0 or high_water is not None
        ):
            raise SystemExit(
                "required resume checkpoint is unavailable; refusing to start or fork tracking identity"
            )
        if checkpoint is not None:
            high_water = commit_checkpoint_state(
                high_water_path=high_water_path,
                observation_index_path=observation_index_path,
                info=checkpoint,
                authority=authority,
                observations=observations,
            )

        attempt_dir = state_dir / "requeue" / f"job-{job_id}-restart-{restart_count:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "schema_version": 2,
            "job_id": job_id,
            "restart_count": restart_count,
            "checkpoint": checkpoint.as_dict() if checkpoint else None,
            "initial_checkpoint": (
                initial_checkpoint.as_dict()
                if initial_checkpoint is not None
                else (str(initial_checkpoint_path) if initial_checkpoint_path is not None else None)
            ),
            "checkpoint_globs": args.checkpoint_glob,
            "validator": str(validator) if validator else "UNVALIDATED_EXPLICIT_OPT_IN",
            "command": args.command,
            "requeue_owner": args.requeue_owner,
            "checkpoint_signal": args.checkpoint_signal.name,
            "checkpoint_forwarding": args.checkpoint_forwarding,
            "scancel": scancel,
            "started_at_unix": time.time(),
            "status": "DRY_RUN" if args.dry_run else "STARTING",
        }
        atomic_json(attempt_dir / "attempt.json", record)

        metadata_json = (
            json.dumps(checkpoint.metadata, sort_keys=True, separators=(",", ":"))
            if checkpoint
            else "{}"
        )
        child_env = os.environ.copy()
        child_env.update(
            {
                "ICE_RESUME_CHECKPOINT": str(checkpoint.path) if checkpoint else "",
                "ICE_RESUME_CHECKPOINT_SHA256": checkpoint.sha256 if checkpoint else "",
                "ICE_RESUME_GLOBAL_STEP": str(checkpoint.global_step) if checkpoint else "",
                "ICE_RESUME_CHECKPOINT_METADATA_JSON": metadata_json,
                "ICE_RESTART_COUNT": str(restart_count),
                "ICE_ATTEMPT_DIR": str(attempt_dir),
                "ICE_REQUEUE_OWNER": args.requeue_owner,
                "ICE_CHILD_REQUEUE_DISABLED": "1" if args.requeue_owner == "runner" else "0",
            }
        )
        if completion_sentinel is not None:
            child_env["ICE_COMPLETION_SENTINEL"] = str(completion_sentinel)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "checkpoint": checkpoint.as_dict() if checkpoint else None,
                        "command": shlex.join(args.command),
                        "attempt_dir": str(attempt_dir),
                        "requeue_owner": args.requeue_owner,
                    },
                    sort_keys=True,
                )
            )
            return 0

        event_path = attempt_dir / "events.jsonl"
        child = None
        launch_gate_error: str | None = None
        if termination_signal is None and boundary_requested_at is None:
            gate_read_fd, gate_write_fd = os.pipe()
            launch_gate_write_fd = gate_write_fd
            gate_env = child_env.copy()
            gate_env["ICE_INTERNAL_LAUNCH_GATE_FD"] = str(gate_read_fd)
            try:
                child = subprocess.Popen(
                    [sys.executable, "-c", LAUNCH_GATE_CODE, *args.command],
                    env=gate_env,
                    start_new_session=True,
                    pass_fds=(gate_read_fd,),
                )
            except BaseException:
                close_launch_gate()
                raise
            finally:
                os.close(gate_read_fd)

            if termination_signal is None and boundary_requested_at is None:
                try:
                    written = os.write(gate_write_fd, b"G")
                    if written != 1:
                        launch_gate_error = "launch gate accepted a short write"
                except OSError as exc:
                    launch_gate_error = str(exc)
                else:
                    workload_launch_committed = True
            close_launch_gate()

            if not workload_launch_committed:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                child = None

        if child is None:
            if termination_signal is not None:
                record.update(
                    {
                        "status": f"RECEIVED_{termination_signal.name}_BEFORE_CHILD_SPAWN",
                        "finished_at_unix": time.time(),
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                return 128 + termination_signal.value
            if boundary_requested_at is None:
                record.update(
                    {
                        "status": "CHILD_LAUNCH_GATE_FAILED",
                        "launch_gate_error": launch_gate_error or "unknown launch gate failure",
                        "finished_at_unix": time.time(),
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                return 77

            boundary_processed = True
            signal_at = boundary_requested_at
            boundary_requested_at = None
            if (
                checkpoint is None
                or high_water is None
                or str(checkpoint.path) != high_water["path"]
                or checkpoint.global_step != high_water["global_step"]
                or checkpoint.sha256 != high_water["sha256"]
            ):
                record.update(
                    {
                        "status": "EARLY_BOUNDARY_REFUSED_NO_AUTHENTICATED_HIGH_WATER",
                        "finished_at_unix": time.time(),
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                return 76

            proof_path = attempt_dir / "early-boundary-proof.json"
            proof = {
                "schema_version": 1,
                "status": "EARLY_BOUNDARY_NO_WORK_CHECKPOINT_AUTHENTICATED",
                "job_id": job_id,
                "restart_count": restart_count,
                "signal": "SIGUSR1",
                "signal_received_at_unix": signal_at,
                "proof_created_at_unix": time.time(),
                "child_spawned": False,
                "optimizer_work_performed": False,
                "checkpoint_signal_forwarded": False,
                "fresh_checkpoint_required": False,
                "requeue_owner": args.requeue_owner,
                "validation_authority": authority,
                "checkpoint": checkpoint.as_dict(),
                "high_water_observation_index_sha256": high_water.get(
                    "observation_index_sha256"
                ),
            }
            atomic_json_once(proof_path, proof)
            proof_sha256 = stable_sha256_file(proof_path)
            append_event(
                event_path,
                {
                    "event": "early_boundary_no_work_checkpoint_authenticated",
                    "proof": str(proof_path),
                    "proof_sha256": proof_sha256,
                    "checkpoint": checkpoint.as_dict(),
                    "at_unix": time.time(),
                },
            )
            if args.requeue_owner != "runner":
                record.update(
                    {
                        "status": "EARLY_BOUNDARY_REFUSED_CHILD_REQUEUE_OWNER",
                        "early_boundary_proof": str(proof_path),
                        "early_boundary_proof_sha256": proof_sha256,
                        "finished_at_unix": time.time(),
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                return 76
            if restart_count >= args.max_restarts:
                record.update(
                    {
                        "status": "EARLY_BOUNDARY_REFUSED_RESTART_LIMIT",
                        "early_boundary_proof": str(proof_path),
                        "early_boundary_proof_sha256": proof_sha256,
                        "finished_at_unix": time.time(),
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                return 71
            if termination_signal is not None:
                record.update(
                    {
                        "status": f"RECEIVED_{termination_signal.name}_BEFORE_EARLY_REQUEUE",
                        "early_boundary_proof": str(proof_path),
                        "early_boundary_proof_sha256": proof_sha256,
                        "finished_at_unix": time.time(),
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                return 128 + termination_signal.value

            assert scontrol is not None
            record.update(
                {
                    "status": "EARLY_BOUNDARY_REQUEUE_REQUESTED",
                    "checkpoint": checkpoint.as_dict(),
                    "early_boundary_proof": str(proof_path),
                    "early_boundary_proof_sha256": proof_sha256,
                    "fresh_checkpoint_after_signal": None,
                }
            )
            atomic_json(attempt_dir / "attempt.json", record)
            completed = subprocess.run(
                [scontrol, "requeue", job_id],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                record.update(
                    {
                        "status": "EARLY_BOUNDARY_REQUEUE_FAILED",
                        "scontrol_returncode": completed.returncode,
                        "scontrol_stdout": completed.stdout,
                        "scontrol_stderr": completed.stderr,
                        "finished_at_unix": time.time(),
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                return completed.returncode or 72
            record.update(
                {
                    "status": "EARLY_BOUNDARY_REQUEUE_ACCEPTED",
                    "finished_at_unix": time.time(),
                }
            )
            atomic_json(attempt_dir / "attempt.json", record)
            return 0

        record.update({"status": "RUNNING", "child_pid": child.pid})
        atomic_json(attempt_dir / "attempt.json", record)

        def forward_termination(received: signal.Signals) -> None:
            try:
                os.killpg(child.pid, received)
            except ProcessLookupError:
                pass

        while True:
            if termination_signal is not None:
                append_event(
                    event_path,
                    {"event": "signal", "signal": termination_signal.name, "at_unix": time.time()},
                )
                forward_termination(termination_signal)
                record.update({"status": f"FORWARDED_{termination_signal.name}"})
                atomic_json(attempt_dir / "attempt.json", record)
                return 128 + termination_signal.value

            if boundary_requested_at is not None:
                # A queued startup boundary is meaningful only while the child
                # is alive. Never signal Slurm steps for an already-completed
                # command merely because discovery took a long time.
                if child.poll() is not None:
                    boundary_requested_at = None
                    continue
                boundary_processed = True
                signal_at = boundary_requested_at
                boundary_requested_at = None
                live_baseline, observations = discover_checkpoint_incrementally(
                    args.checkpoint_glob,
                    state_dir,
                    validator,
                    allow_unvalidated=args.allow_unvalidated_checkpoints,
                    cache=validation_cache,
                    high_water=high_water,
                    observations=observations,
                    prevalidated=(checkpoint,),
                )
                baseline = newest_checkpoint(checkpoint, live_baseline)
                enforce_high_water(baseline, high_water)
                if baseline is not None:
                    high_water = commit_checkpoint_state(
                        high_water_path=high_water_path,
                        observation_index_path=observation_index_path,
                        info=baseline,
                        authority=authority,
                        observations=observations,
                    )
                append_event(
                    event_path,
                    {
                        "event": "signal",
                        "signal": "SIGUSR1",
                        "checkpoint_signal": args.checkpoint_signal.name,
                        "baseline": baseline.as_dict() if baseline else None,
                        "at_unix": signal_at,
                    },
                )
                if termination_signal is not None or child.poll() is not None:
                    continue
                assert scancel is not None
                try:
                    signal_result = forward_checkpoint_to_slurm_steps(
                        scancel,
                        job_id,
                        args.checkpoint_signal,
                    )
                except OSError as exc:
                    record.update(
                        {
                            "status": "CHECKPOINT_SIGNAL_FORWARD_FAILED",
                            "checkpoint_signal_argv": [scancel, "--signal=USR2", job_id],
                            "checkpoint_signal_error": str(exc),
                        }
                    )
                    atomic_json(attempt_dir / "attempt.json", record)
                    return 75
                if signal_result.returncode != 0:
                    record.update(
                        {
                            "status": "CHECKPOINT_SIGNAL_FORWARD_FAILED",
                            "checkpoint_signal_argv": [scancel, "--signal=USR2", job_id],
                            "checkpoint_signal_returncode": signal_result.returncode,
                            "checkpoint_signal_stdout": signal_result.stdout,
                            "checkpoint_signal_stderr": signal_result.stderr,
                        }
                    )
                    atomic_json(attempt_dir / "attempt.json", record)
                    return signal_result.returncode or 75
                append_event(
                    event_path,
                    {
                        "event": "checkpoint_signal_forwarded",
                        "argv": [scancel, "--signal=USR2", job_id],
                        "at_unix": time.time(),
                    },
                )
                # Give an early queued boundary a complete save window after
                # forwarding; validator startup time must not consume grace.
                deadline = time.time() + args.checkpoint_grace_seconds
                latest = baseline
                fresh = False
                while time.time() < deadline:
                    if termination_signal is not None:
                        break
                    candidate, observations = discover_checkpoint_incrementally(
                        args.checkpoint_glob,
                        state_dir,
                        validator,
                        allow_unvalidated=args.allow_unvalidated_checkpoints,
                        cache=validation_cache,
                        high_water=high_water,
                        observations=observations,
                        prevalidated=(baseline,),
                    )
                    latest = newest_checkpoint(latest, candidate)
                    if is_fresh(candidate, baseline):
                        latest = candidate
                        fresh = True
                        break
                    time.sleep(args.poll_seconds)
                if termination_signal is not None:
                    continue
                if latest is None:
                    record.update({"status": "REQUEUE_REFUSED_NO_VALID_CHECKPOINT"})
                    atomic_json(attempt_dir / "attempt.json", record)
                    return 70
                if args.requeue_owner == "runner" and not fresh:
                    record.update(
                        {
                            "status": "REQUEUE_REFUSED_NO_FRESH_CHECKPOINT",
                            "checkpoint": latest.as_dict(),
                            "fresh_checkpoint_after_signal": False,
                        }
                    )
                    atomic_json(attempt_dir / "attempt.json", record)
                    return 74
                enforce_high_water(latest, high_water)
                high_water = commit_checkpoint_state(
                    high_water_path=high_water_path,
                    observation_index_path=observation_index_path,
                    info=latest,
                    authority=authority,
                    observations=observations,
                )
                checkpoint = latest
                if restart_count >= args.max_restarts:
                    record.update(
                        {
                            "status": "REQUEUE_REFUSED_RESTART_LIMIT",
                            "checkpoint": latest.as_dict(),
                        }
                    )
                    atomic_json(attempt_dir / "attempt.json", record)
                    return 71
                record.update(
                    {
                        "status": (
                            "REQUEUE_REQUESTED"
                            if args.requeue_owner == "runner"
                            else "CHILD_REQUEUE_OWNER_CHECKPOINT_READY"
                        ),
                        "checkpoint": latest.as_dict(),
                        "fresh_checkpoint_after_signal": fresh,
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                append_event(
                    event_path,
                    {
                        "event": "checkpoint_ready_for_requeue",
                        "checkpoint": latest.as_dict(),
                        "fresh_checkpoint_after_signal": fresh,
                        "requeue_owner": args.requeue_owner,
                        "at_unix": time.time(),
                    },
                )
                if args.requeue_owner == "runner":
                    if termination_signal is not None:
                        continue
                    assert scontrol is not None
                    completed = subprocess.run(
                        [scontrol, "requeue", job_id],
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    if completed.returncode != 0:
                        record.update(
                            {
                                "status": "REQUEUE_FAILED",
                                "scontrol_returncode": completed.returncode,
                                "scontrol_stdout": completed.stdout,
                                "scontrol_stderr": completed.stderr,
                            }
                        )
                        atomic_json(attempt_dir / "attempt.json", record)
                        return completed.returncode or 72
                    return 0

            code = child.poll()
            if code is not None:
                if code == 0 and completion_sentinel is not None:
                    live_terminal, observations = discover_checkpoint_incrementally(
                        args.checkpoint_glob,
                        state_dir,
                        validator,
                        allow_unvalidated=args.allow_unvalidated_checkpoints,
                        cache=validation_cache,
                        high_water=high_water,
                        observations=observations,
                        prevalidated=(checkpoint,),
                    )
                    terminal = newest_checkpoint(checkpoint, live_terminal)
                    enforce_high_water(terminal, high_water)
                    if terminal is None:
                        record.update({"status": "COMPLETE_REFUSED_NO_TERMINAL_CHECKPOINT"})
                        atomic_json(attempt_dir / "attempt.json", record)
                        return 73
                    high_water = commit_checkpoint_state(
                        high_water_path=high_water_path,
                        observation_index_path=observation_index_path,
                        info=terminal,
                        authority=authority,
                        observations=observations,
                    )
                    atomic_json(
                        completion_sentinel,
                        {
                            "schema_version": 1,
                            "status": "COMPLETE",
                            "job_id": job_id,
                            "restart_count": restart_count,
                            "checkpoint": terminal.as_dict(),
                            "completed_at_unix": time.time(),
                        },
                    )
                record.update(
                    {
                        "status": "COMPLETE" if code == 0 else "CHILD_FAILED_NO_REQUEUE",
                        "child_exit_code": code,
                        "finished_at_unix": time.time(),
                    }
                )
                atomic_json(attempt_dir / "attempt.json", record)
                return code
            time.sleep(args.poll_seconds)
    finally:
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
