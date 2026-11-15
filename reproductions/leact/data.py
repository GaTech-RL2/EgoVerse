"""Versioned oracle records and local artifact I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .core import distribution


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid JSON (possibly interrupted write)") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(row)
    return rows


def load_oracle(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    if metadata["schema_version"] != 1 or metadata["sha256"] != file_sha256(path):
        raise ValueError("Oracle version or checksum mismatch")
    rows = read_jsonl(path)
    if not rows or len(rows) != metadata["num_infosets"]:
        raise ValueError("Oracle coverage does not match manifest")
    if len({r["state_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate information-state IDs")
    for row in rows:
        if row["state_id"] != fingerprint([metadata["game_string"], row["info_state"]]):
            raise ValueError("Information-state ID does not match its contents")
        row["expert_policy"] = distribution(row["expert_policy"], row["legal_actions"])
        if set(row["action_ids"]) != set(row["legal_actions"]):
            raise ValueError("Missing OpenSpiel action IDs")
        ids = list(row["action_ids"].values())
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid or duplicate OpenSpiel action IDs")
    return rows, metadata
