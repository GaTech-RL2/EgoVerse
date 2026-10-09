"""Strict JSON, immutable receipts and append-only evaluator-owned events."""

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate_key")
            result[k] = v
        return result

    def invalid(_):
        raise ValueError("nonfinite_json")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def exact(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("invalid_fields")


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


class Events:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.stream = (self.directory / "events.jsonl").open("x", buffering=1)
        self.identity = identity
        self.started = time.monotonic()
        self.sequence = 0
        self.previous = "0" * 64

    def emit(self, kind, **fields):
        def loggable(value):
            if isinstance(value, float) and not math.isfinite(value):
                return {"rejected_nonfinite_number": str(value)}
            if isinstance(value, dict):
                return {k: loggable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [loggable(v) for v in value]
            return value

        row = {
            **self.identity,
            "sequence": self.sequence,
            "event": kind,
            "utc": utc(),
            "monotonic_offset": time.monotonic() - self.started,
            "previous_sha256": self.previous,
            **loggable(fields),
        }
        row["sha256"] = digest(row)
        self.stream.write(encoded(row).decode() + "\n")
        self.previous = row["sha256"]
        self.sequence += 1
        return row

    def close(self):
        self.stream.close()
