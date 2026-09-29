"""Bounded, offline verification of recipe result archives.

Streams all compressed and regular-member bytes. Retains structured records and
small learned checkpoint files, never rollout arrays or a whole archive. This
verifies file integrity and completion seals, not numerical or physics replay.
"""

import argparse
import contextlib
import gzip
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from .records import file_sha256

SCHEMA_VERSION = "recipe-archive-audit-1.0"
BLOCK = 1024 * 1024
DEFAULT_RETAINED_LIMIT = 256 * BLOCK
DEFAULT_RESERVE = 512 * BLOCK


class AuditError(ValueError):
    """An allowlisted error code; never carries a URL or response body."""


class PendingArchive(AuditError):
    """The final archive has not been published yet."""


def _require(condition, code):
    if not condition:
        raise AuditError(code)


def _json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, "duplicate_json_key")
            value[key] = item
        return value

    try:
        return json.loads(
            Path(path).read_bytes(),
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                AuditError("nonfinite_json")
            ),
        )
    except (UnicodeError, json.JSONDecodeError):
        raise AuditError("invalid_json") from None


def _relative(name):
    _require(isinstance(name, str) and bool(name), "invalid_member_path")
    parts = PurePosixPath(name).parts
    _require(
        not name.startswith("/")
        and "\\" not in name
        and not any(ord(char) < 32 or ord(char) == 127 for char in name)
        and all(part not in ("", ".", "..") for part in name.split("/"))
        and ":" not in name
        and len(name) <= 1024
        and parts,
        "unsafe_member_path",
    )
    return name


def _record(value):
    _require(
        isinstance(value, dict)
        and set(value) == {"bytes", "sha256"}
        and type(value["bytes"]) is int
        and value["bytes"] >= 0
        and isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "invalid_file_record",
    )
    return value


class _HashReader:
    def __init__(self, source):
        self.source = source
        self.sha = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        value = self.source.read(size)
        self.sha.update(value)
        self.bytes += len(value)
        return value

    def record(self):
        return {"sha256": self.sha.hexdigest(), "bytes": self.bytes}


def _hash_member(stream, destination=None):
    reader = _HashReader(stream)
    output = (
        destination.open("xb") if destination is not None else contextlib.nullcontext()
    )
    with output as retained:
        for block in iter(lambda: reader.read(BLOCK), b""):
            if retained is not None:
                retained.write(block)
    return reader.record()


def _tar_gzip(source, root_name, visit):
    """Drain tar padding and gzip EOF: a tar end marker alone is insufficient."""
    reader = _HashReader(source)
    names, files = set(), {}
    with gzip.GzipFile(fileobj=reader, mode="rb") as expanded:
        with tarfile.open(fileobj=expanded, mode="r|", bufsize=BLOCK) as archive:
            for member in archive:
                name = _relative(
                    member.name.rstrip("/") if member.isdir() else member.name
                )
                _require(name not in names, "duplicate_archive_member")
                names.add(name)
                _require(
                    name == root_name or name.startswith(root_name + "/"),
                    "unexpected_archive_root",
                )
                if member.isdir():
                    continue
                _require(
                    member.isfile() and not member.sparse, "nonregular_archive_member"
                )
                relative = name[len(root_name) + 1 :]
                _relative(relative)
                _require(member.size >= 0, "negative_member_size")
                stream = archive.extractfile(member)
                _require(stream is not None, "unreadable_archive_member")
                with stream:
                    record = visit(relative, stream, member.size)
                _require(record["bytes"] == member.size, "member_size_mismatch")
                files[relative] = record
            # _Stream may have read ahead beyond the first tar end block.
            for block in iter(lambda: archive.fileobj.read(BLOCK), b""):
                _require(not block.strip(b"\0"), "nonzero_tar_trailer")
        for block in iter(lambda: expanded.read(BLOCK), b""):
            _require(not block.strip(b"\0"), "nonzero_tar_trailer")
    _require(reader.read(1) == b"", "unconsumed_compressed_bytes")
    return reader.record(), files


def _remote_open(url, *, probe=False):
    headers = {"Range": "bytes=0-0"} if probe else {}
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=120
        )
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise PendingArchive("archive_not_published") from None
        raise AuditError(f"archive_http_{error.code}") from None
    except Exception as error:
        raise AuditError("archive_transport_" + type(error).__name__) from None
    return response


def _remote_identity(response, *, probe=False):
    _require(
        response.status in ((200, 206) if probe else (200,)), "archive_http_status"
    )
    try:
        length = int(response.headers.get("Content-Length", "-1"))
        if response.status == 206:
            match = re.fullmatch(
                r"bytes 0-0/(\d+)", response.headers.get("Content-Range", "")
            )
            _require(
                probe and match is not None and length == 1, "archive_range_identity"
            )
            length = int(match[1])
    except (TypeError, ValueError):
        raise AuditError("archive_http_length") from None
    etag = response.headers.get("ETag", "")
    _require(
        length > 0 and re.fullmatch(r'"[A-Za-z0-9_-]{1,128}"', etag),
        "archive_missing_length_or_etag",
    )
    return {"bytes": length, "etag": etag}


@contextlib.contextmanager
def _open_source(source):
    if isinstance(source, str) and source.startswith("https://"):
        with _remote_open(source) as stream:
            identity = _remote_identity(stream)
            yield stream, identity
        with _remote_open(source, probe=True) as response:
            _require(
                _remote_identity(response, probe=True) == identity,
                "archive_changed_during_read",
            )
    else:
        path = Path(source)
        _require(path.is_file(), "missing_local_archive")
        before = path.stat()
        with path.open("rb") as stream:
            yield stream, {"bytes": before.st_size, "etag": None}
        after = path.stat()
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "archive_changed_during_read",
        )


def _seal_matches(seal, files, *, exclude=()):
    _require(isinstance(seal, dict) and bool(seal), "invalid_completion_seal")
    for name, record in seal.items():
        _relative(name)
        _record(record)
    expected = {name: record for name, record in files.items() if name not in exclude}
    _require(seal == expected, "completion_seal_mismatch")


def _verify_contract(root, files, nested, phase, worker, workflow):
    def read(name):
        _require(
            name in files and (root / name).is_file(),
            "missing_required_structured_file",
        )
        return _json(root / name)

    runtime, progress, protocol = (
        read("runtime.json"),
        read("progress.json"),
        read("protocol.json"),
    )
    _require(
        runtime.get("phase") == phase
        and runtime.get("worker") == worker
        and runtime.get("workflow") == workflow
        and runtime.get("provider_credential_attached") is False
        and runtime.get("tf32") is False,
        "runtime_identity_mismatch",
    )
    _require(
        progress.get("status") == "complete" and progress.get("phase") == phase,
        "run_not_complete",
    )
    _require("failure.json" not in files, "failure_artifact_present")
    training = read("training_receipt.json")
    _require(
        training.get("schema_version") == "recipe-training-1.0"
        and training.get("status") == "complete"
        and training.get("base_weights_unchanged") is True
        and training.get("provider_calls") == 0
        and training.get("provider_tokens") == 0,
        "training_receipt_incomplete",
    )
    checks = {
        "completion_seal_verified": False,
        "training_bundle_seal_verified": False,
        "embedded_bundle_archive_verified": False,
        "array_descriptors_verified": False,
    }
    if phase == "evaluation":
        completion = read("completion_receipt.json")
        _require(
            completion.get("schema_version") == "recipe-evaluation-worker-1.0"
            and completion.get("status") == "complete"
            and completion.get("worker") == worker
            and completion.get("provider_calls") == 0
            and completion.get("provider_tokens") == 0,
            "evaluation_receipt_incomplete",
        )
        _seal_matches(
            completion.get("files"), files, exclude=("completion_receipt.json",)
        )
        checks["completion_seal_verified"] = True
    else:
        _require(training.get("workflow") == workflow, "training_workflow_mismatch")
        bundle = {
            name[len("bundle/") :]: record
            for name, record in files.items()
            if name.startswith("bundle/")
        }
        _seal_matches(read("bundle/seal.json"), bundle, exclude=("seal.json",))
        _require(
            read("bundle/training_receipt.json") == training,
            "bundle_training_receipt_mismatch",
        )
        _require(read("bundle/protocol.json") == protocol, "bundle_protocol_mismatch")
        learned = read("learned_bundle.json")
        _require(
            learned.get("schema_version") == "recipe-bundle-1.0",
            "bundle_metadata_schema",
        )
        _require(
            files.get("learned_bundle.tar.gz")
            == {key: learned.get(key) for key in ("bytes", "sha256")}
            and nested == bundle,
            "embedded_bundle_mismatch",
        )
        checks.update(
            training_bundle_seal_verified=True, embedded_bundle_archive_verified=True
        )
    return checks


def audit_archive(
    source,
    destination,
    *,
    phase,
    worker,
    workflow,
    expected_archive=None,
    expected_files=None,
    retain_videos=(),
    maximum_retained_bytes=DEFAULT_RETAINED_LIMIT,
    minimum_free_bytes=DEFAULT_RESERVE,
):
    """Write an immutable receipt and retained ``results/`` on complete success.

    ``source`` is a local Path or a private HTTPS URL, never included in output.
    Expected file paths are relative to results/. Array bytes are hashed only.
    The staging directory is removed on any error; existing destinations fail.
    """
    _require(phase in ("training", "evaluation"), "invalid_phase")
    _require(
        type(worker) is int and 0 <= worker < (1 if phase == "training" else 5),
        "invalid_worker",
    )
    _require(
        isinstance(workflow, str) and re.fullmatch(r"[A-Za-z0-9_-]+", workflow),
        "invalid_workflow",
    )
    for value in (maximum_retained_bytes, minimum_free_bytes):
        _require(type(value) is int and value >= 0, "invalid_disk_limit")
    source_sha = file_sha256(__file__)
    destination = Path(destination)
    _require(not destination.exists(), "audit_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(
        shutil.disk_usage(destination.parent).free
        >= minimum_free_bytes + maximum_retained_bytes,
        "insufficient_disk_reserve",
    )
    videos = {_relative(name) for name in retain_videos}
    _require(all(name.endswith(".mp4") for name in videos), "invalid_selected_video")
    expected_files = expected_files or {}
    for name, record in expected_files.items():
        _relative(name)
        _record(record)
    if expected_archive is not None:
        _record(expected_archive)
    temporary = Path(tempfile.mkdtemp(prefix=".recipe-audit-", dir=destination.parent))
    root = temporary / "results"
    root.mkdir()
    kept, nested, retained_bytes = {}, None, 0
    try:

        def visit(name, stream, size):
            nonlocal nested, retained_bytes
            retain = (
                name.endswith((".json", ".jsonl", ".log", ".yaml", ".yml"))
                or name.startswith("bundle/")
                or name in videos
            )
            target = None
            if retain:
                _require(
                    retained_bytes + size <= maximum_retained_bytes,
                    "retained_byte_limit_exceeded",
                )
                retained_bytes += size
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
            if name == "learned_bundle.tar.gz":
                record, nested = _tar_gzip(
                    stream, "bundle", lambda _, data, size: _hash_member(data)
                )
            else:
                record = _hash_member(stream, target)
            if retain:
                kept[name] = record
            return record

        with _open_source(source) as (stream, identity):
            compressed, files = _tar_gzip(stream, "results", visit)
            _require(
                compressed["bytes"] == identity["bytes"], "compressed_size_mismatch"
            )
        if expected_archive is not None:
            _require(compressed == expected_archive, "expected_archive_mismatch")
        _require(
            all(files.get(name) == record for name, record in expected_files.items()),
            "published_file_mismatch",
        )
        _require(videos <= set(kept), "selected_video_missing")
        checks = _verify_contract(root, files, nested, phase, worker, workflow)
        inventory_bytes = _json_bytes(files)
        _require(
            retained_bytes + len(inventory_bytes) <= maximum_retained_bytes,
            "retained_byte_limit_exceeded",
        )
        (temporary / "member_inventory.json").write_bytes(inventory_bytes)
        _require(file_sha256(__file__) == source_sha, "auditor_source_changed")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "complete": True,
            "phase": phase,
            "worker": worker,
            "workflow": workflow,
            "source_sha256": source_sha,
            "verified_unix": time.time(),
            "archive": {
                **compressed,
                "etag": identity["etag"],
                "gzip_crc_verified": True,
                "unchanged_before_after": True,
                "retained_local": False,
                "sha256_origin": "independently computed over all compressed bytes",
                "expected_compressed_hash_verified": expected_archive is not None,
            },
            "files": kept,
            "member_inventory": {
                "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                "members": len(files),
                "bytes": sum(record["bytes"] for record in files.values()),
            },
            "retained_file_inventory_sha256": hashlib.sha256(
                _json_bytes(kept)
            ).hexdigest(),
            "retained_files": len(kept),
            "retained_bytes": retained_bytes,
            "checks": {
                **checks,
                "full_archive_stream_verified": True,
                "all_member_bytes_verified": True,
                "published_file_bindings_verified": True,
            },
            "limitations": [
                "Regular-member bytes and seals are verified; array descriptors, hidden states, optimizer updates and physics are not replayed.",
                "Training seals only the learned bundle; its other files are bound by the independently computed whole-archive hash.",
                "ETag is an object-version consistency check, not a substitute for the computed SHA256.",
            ],
        }
        (temporary / "audit.json").write_bytes(_json_bytes(receipt))
        temporary.rename(destination)
        return receipt
    except AuditError:
        raise
    except Exception as error:
        raise AuditError("archive_read_" + type(error).__name__) from None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def audit_catalog(catalog_path, destination, *, phase, worker, **kwargs):
    """Resolve only this worker's archive from an existing private catalog."""
    catalog = _json(catalog_path)
    value = catalog["objects"][f"worker_{worker}/artifacts.tar.gz"]
    url = value["url"] if isinstance(value, dict) else value
    _require(
        isinstance(url, str) and url.startswith("https://"), "invalid_catalog_archive"
    )
    return audit_archive(
        url,
        destination,
        phase=phase,
        worker=worker,
        workflow=catalog["workflow"],
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--phase", choices=("training", "evaluation"), required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = audit_catalog(
            args.catalog, args.output, phase=args.phase, worker=args.worker
        )
    except AuditError as error:
        print(
            json.dumps(
                {
                    "status": "pending"
                    if isinstance(error, PendingArchive)
                    else "failed",
                    "code": str(error),
                }
            )
        )
        raise SystemExit(3 if isinstance(error, PendingArchive) else 1) from None
    print(
        json.dumps(
            {
                "status": "passed",
                "phase": receipt["phase"],
                "worker": receipt["worker"],
                "archive_sha256": receipt["archive"]["sha256"],
                "archive_bytes": receipt["archive"]["bytes"],
                "audit_sha256": file_sha256(args.output / "audit.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
