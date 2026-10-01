"""Fetch exactly the pinned archive of this study's previous interrupted run."""

import json
import re
import tarfile
from pathlib import Path

from astra_reversal.demo_skill_program import ARMS
from astra_reversal.records import file_sha256

SCHEMA = "demo-skill-recovery-1"
OWNED_WORKFLOW = r"astra-pi05-input-skill-library-20260930-(pilot|full)-[0-9]+"


def validate_manifest(value):
    if value.get("schema_version") != SCHEMA or not re.fullmatch(
        OWNED_WORKFLOW, value.get("source_workflow", "")
    ):
        raise ValueError("Recovery must name one owned demo-skill workflow")
    workers = value.get("workers", [])
    if len(workers) != 2 or [w["worker"] for w in workers] != [0, 1]:
        raise ValueError("Recovery requires both separate-arm worker receipts")
    for row in workers:
        prefix = f"experiments/astra-reversal-20260924/{value['source_workflow']}/worker_{row['worker']}"
        if (
            row["arm"] != ARMS[row["worker"]]
            or row["key"] != prefix + "/artifacts.tar.gz"
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
        ):
            raise ValueError("Recovery archive identity is not pinned to its worker")
    return value


def unpack_archive(path, destination, receipt):
    path, destination = Path(path), Path(destination)
    if (
        path.stat().st_size != receipt["bytes"]
        or file_sha256(path) != receipt["sha256"]
    ):
        raise ValueError("Recovery archive bytes differ from the pinned receipt")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(path, "r:gz") as archive:
        names = set()
        for member in archive.getmembers():
            name = Path(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "results"
                or not (member.isfile() or member.isdir())
                or member.name in names
            ):
                raise ValueError("Unsafe or duplicate recovery archive member")
            names.add(member.name)
        archive.extractall(destination, filter="data")
    return destination / "results"


def fetch_archive(client, manifest, worker, destination):
    manifest = validate_manifest(manifest)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    receipt = manifest["workers"][worker]
    path = destination / "artifacts.tar.gz"
    client.download_file("rldb", receipt["key"], str(path))
    results = unpack_archive(path, destination / "unpacked", receipt)
    runtime = json.loads((results / "runtime.json").read_text())
    if (
        runtime["workflow"] != manifest["source_workflow"]
        or runtime["worker"] != worker
        or runtime["arm"] != receipt["arm"]
    ):
        raise ValueError("Recovered archive runtime differs from its provenance")
    return results
