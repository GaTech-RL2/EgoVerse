"""Ingest a SharpaDex release chunk by chunk, within a disk budget.

A full SharpaDex release needs more disk than a shared volume has free, so it
is converted in chunks, each going through::

    plan -> fetch -> convert -> verify -> upload (gated) -> clean

``plan`` packs the manifest's kept episodes into chunks whose inputs (exact
sizes from a ``path,size`` listing of the Hub repository) plus estimated
outputs fit ``--budget-gb``. Every other subcommand works on one chunk,
appends to ``<state-dir>/ledger.jsonl`` and can simply be rerun after a
timeout or preemption: ``fetch`` downloads only missing files, ``convert``
skips episodes whose ``<hash>.conversion.json`` exists, ``upload`` skips
episodes with a matching receipt, and ``clean`` deletes only files that
``fetch`` downloaded and outputs that a receipt shows uploaded. One chunk::

    python -m egomimic.scripts.sharpa_process.sharpadex_ingest plan \\
        --state-dir state/ --manifest manifest.jsonl --sizes hf_files.csv \\
        --budget-gb 400 --out-bytes-per-frame 180000
    python -m egomimic.scripts.sharpa_process.sharpadex_ingest fetch \\
        --state-dir state/ --chunk chunk_0000 \\
        --dataset-root /data/SharpaDex-v1.0 --work-root /data/sharpadex_work
    # then convert and verify with the same flags; upload (a dry run unless
    # --execute --licence-approved), clean, and status at any time.

``fetch`` and ``convert`` refuse a manifest whose episode hashes have not been
checked against the SQL episode table and S3 (header
``collision_check_complete``) unless ``--allow-provisional-hashes`` is given;
``upload`` always refuses one.
The manifest is read with ``sharpadex_records.read_manifest``, so a manifest
whose ids break the canonical-id contract is refused at every stage.
``verify`` passes an episode only if its sidecar, store and report name it
(canonical and source id), ``sharpa_to_zarr.read_report`` finds no problem
(which requires ``data_status: complete``), the store's ``lab`` is
``sharpa_dex`` and its MP4 exists; ``upload`` writes only to ``DESTINATION``,
``s3://rldb/processed_v3/sharpa_dex/``, the folder of that lab.
``sharpadex_chunk.sbatch`` runs fetch, convert and verify for one chunk on
Slurm.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import fnmatch
import functools
import hashlib
import json
import logging
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from egomimic.scripts.sharpa_process import sharpadex_records
from egomimic.scripts.sharpa_process.existing_hashes_s3 import (
    BUCKET,
    S3_ROOT,
    listing_covers_root,
)

logger = logging.getLogger(__name__)

TOOL = "egomimic.scripts.sharpa_process.sharpadex_ingest"
#: The episodes' lab; it must also be the S3 folder they are uploaded to.
LAB = "sharpa_dex"
#: The one place episodes are uploaded to. ``stage_processed_folder --folder
#: sharpa_dex`` lists it and the merge takes the lab from that folder.
DESTINATION = f"s3://{BUCKET}/{S3_ROOT}{LAB}/"
CONVERTER = "egomimic.scripts.sharpa_process.sharpa_to_zarr"
#: The checkout this file is in; workers import the converter from it.
REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ID = "Sharpa-Robotics/SharpaDex-v1.0"
#: The release revision that the manifest and the size listing describe.
REVISION = "6b48181a41d8836692fb1204cc60ade64e107d20"
#: Video streams fetched by default; ``tactile_raw`` is never stored or fetched.
STREAMS = ("head_left", "head_right", "wrist_left", "wrist_right", "tactile_deform")
#: Spellings of a season's LeRobot v2.1 folder; ``lerobot_v3.0`` is never fetched.
V21_DIRS = ("lerobot_v2.1", "lerobotv2.1", "lerobot_v2_1")
STAGES = ("fetch", "convert", "verify", "upload", "clean")
#: Exit status of ``fetch`` when the disk guard refuses.
EXIT_DISK = 3
#: Converter exit codes after which every episode is written (2 means
#: validation errors, which ``verify`` reports).
CONVERTED = (0, 2)
#: Arrays grouped for the bytes-per-frame report; the rest is numeric.
ARRAY_GROUPS = {
    "rgb_front_1": ("images.front_1",),
    "rgb_wrists": ("images.left_wrist", "images.right_wrist"),
    "rgb_head_right": ("images.head_right",),
    "tactile_images": ("left.obs_tactile_image", "right.obs_tactile_image"),
}
#: Environment variables a child process keeps; everything else, R2, AWS and
#: Hugging Face credentials among it, is dropped.
CHILD_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "SLURM_CONF",
)
_EPISODE_FILE = re.compile(r"episode_(\d+)\.(parquet|mp4)")
_SHARD_LOCK = re.compile(r"\.shard-(\d+)-of-(\d+)\.lock$")


class IngestError(RuntimeError):
    """A stage cannot run; ``exit_code`` is the process exit status."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


class _Interrupted(Exception):
    """SIGTERM (Slurm time limit, preemption or scancel) reached a stage."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f}"


def child_env(**extra: str) -> dict[str, str]:
    """The ``CHILD_ENV`` variables of this process, plus ``extra``."""
    env = {k: os.environ[k] for k in CHILD_ENV if k in os.environ}
    env.update(extra)
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.cache
def git_sha() -> str | None:
    """Return this checkout's commit, ``-dirty`` if modified, if git knows it.

    ``GIT_OPTIONAL_LOCKS=0`` keeps ``git status`` from rewriting the index.
    """
    env = child_env(GIT_OPTIONAL_LOCKS="0")
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return None
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    dirty = status.returncode == 0 and bool(status.stdout.strip())
    return head.stdout.strip() + ("-dirty" if dirty else "")


def id_key(recording_id: str) -> tuple[str, str, int]:
    """Sort key of a ``task/season/index`` recording id."""
    task, season, index = recording_id.split("/")
    return task, season, int(index)


def read_lines(path: Path) -> list[str]:
    """Read one item per line; blank and ``#`` lines are ignored."""
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def read_manifest(path: Path) -> tuple[dict, dict[str, dict]]:
    """Return ``(header, {recording_id: record})`` from a manifest file.

    ``sharpadex_records.read_manifest`` reads and id-checks it (stdlib only, so
    the driver does not import the converter stack); records are plain dicts.
    """
    try:
        header, records = sharpadex_records.read_manifest(path)
    except sharpadex_records.ManifestError as exc:
        raise IngestError(str(exc)) from exc
    return header, {rid: asdict(record) for rid, record in records.items()}


def du(path: Path) -> int:
    """Return the bytes of every file under ``path`` (0 if it is missing)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    total, stack = 0, [path]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except FileNotFoundError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                stack.append(entry.path)
            else:
                total += entry.stat(follow_symlinks=False).st_size
    return total


def load_sample() -> dict:
    """Return the node's load averages, CPU count and load1 per CPU."""
    load1, load5, load15 = os.getloadavg()
    cpus = os.cpu_count() or 1
    return {
        "load1": round(load1, 2),
        "load5": round(load5, 2),
        "load15": round(load15, 2),
        "cpu_count": cpus,
        "ratio": round(load1 / cpus, 3),
    }


@contextlib.contextmanager
def sigterm_raises():
    """Turn SIGTERM into ``_Interrupted`` so a stage can record why it stopped."""

    def handler(signum, frame):
        raise _Interrupted(signal.Signals(signum).name)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class Ledger:
    """The append-only record of stage runs, ``<state-dir>/ledger.jsonl``.

    Each line is ``{ts, chunk, stage, state, host, pid, slurm_job_id, git_sha,
    manifest_sha256, details}``; ``state`` is ``start``, ``ok`` or ``fail``,
    or ``progress`` for the samples ``convert`` takes while workers run.
    """

    def __init__(self, state_dir: Path, manifest_sha256: str | None = None):
        self.path = Path(state_dir) / "ledger.jsonl"
        self.manifest_sha256 = manifest_sha256

    def write(self, chunk: str | None, stage: str, state: str, **details) -> dict:
        entry = {
            "ts": utc_now(),
            "chunk": chunk,
            "stage": stage,
            "state": state,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_sha": git_sha(),
            "manifest_sha256": self.manifest_sha256,
            "details": details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return entry


def read_ledger(state_dir: Path) -> list[dict]:
    """Return the ledger's entries, skipping unreadable (e.g. truncated) lines."""
    return read_jsonl(Path(state_dir) / "ledger.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    """Return a JSON-lines file's entries, skipping unreadable lines."""
    path = Path(path)
    if not path.is_file():
        return []
    entries = []
    text = path.read_bytes().decode("utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("%s:%d: skipping an unreadable line", path, number)
    return entries


def last_entry(entries: list[dict], chunk: str, stage: str) -> dict | None:
    """Return the latest start/ok/fail entry of ``stage`` for ``chunk``."""
    for entry in reversed(entries):
        if (
            entry.get("chunk") == chunk
            and entry.get("stage") == stage
            and entry.get("state") in ("start", "ok", "fail")
        ):
            return entry
    return None


def upload_confirmed(entries: list[dict], chunk: str) -> bool:
    entry = last_entry(entries, chunk, "upload")
    return bool(
        entry
        and entry["state"] == "ok"
        and (entry.get("details") or {}).get("confirmed")
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def index_sizes(path: Path) -> dict[tuple[str, str], dict[int, dict]]:
    """Index a ``path,size`` listing's v2.1 episode files by season.

    Returns ``{(task, season): {episode_index: {"parquet": (path, size) or
    None, "videos": {stream: (path, size)}}}}``. A season listed under more
    than one v2.1 spelling uses the first spelling in ``V21_DIRS``.
    """
    found: dict = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"path", "size"} <= set(reader.fieldnames or ()):
            raise IngestError(f"{path}: expected a 'path,size' header")
        for row in reader:
            parts = row["path"].split("/")
            if len(parts) not in (6, 7) or parts[2] not in V21_DIRS:
                continue
            match = _EPISODE_FILE.fullmatch(parts[-1])
            if match is None:
                continue
            kind = None
            if len(parts) == 6 and parts[3] == "data" and match[2] == "parquet":
                kind = "parquet"
            elif (
                len(parts) == 7
                and parts[3] == "videos"
                and parts[5].startswith("observation.images.")
                and match[2] == "mp4"
            ):
                kind = parts[5].removeprefix("observation.images.")
            if kind is None:
                continue
            by_dir = found.setdefault((parts[0], parts[1]), {})
            episodes = by_dir.setdefault(parts[2], {})
            entry = episodes.setdefault(int(match[1]), {"parquet": None, "videos": {}})
            item = (row["path"], int(row["size"]))
            if kind == "parquet":
                entry["parquet"] = item
            else:
                entry["videos"][kind] = item
    return {
        season: by_dir[next(d for d in V21_DIRS if d in by_dir)]
        for season, by_dir in found.items()
    }


def est_out(record: dict, header: dict) -> int:
    """Estimated output bytes of one episode: Zarr per frame plus its mp4."""
    return round(
        int(record["length"]) * header["out_bytes_per_frame"]
        + header["mp4_bytes_per_episode"]
    )


def pack_chunks(seasons: list[list[dict]], budget: int) -> list[list[dict]]:
    """Pack episodes into chunks of at most ``budget`` bytes (inputs + outputs).

    Whole seasons are added while they fit; a season that does not fit in
    the current chunk starts a new one. A season larger than the budget
    starts a new chunk and is split by episode; its last part stays open for
    the following seasons.
    """
    chunks: list[list[dict]] = []
    current: list[dict] = []
    used = 0

    def close():
        nonlocal current, used
        if current:
            chunks.append(current)
        current, used = [], 0

    for episodes in seasons:
        total = sum(e["in"] + e["out"] for e in episodes)
        if total <= budget:
            if used + total > budget:
                close()
            current.extend(episodes)
            used += total
            continue
        close()
        for episode in episodes:
            cost = episode["in"] + episode["out"]
            if cost > budget:
                raise IngestError(
                    f"{episode['id']} alone needs {gb(cost)} GB, more than "
                    f"--budget-gb; raise the budget"
                )
            if used + cost > budget:
                close()
            current.append(episode)
            used += cost
    close()
    return chunks


def plan_file(args) -> Path:
    return args.plan or Path(args.state_dir) / "plan.json"


def cmd_plan(args) -> int:
    path = plan_file(args)
    if path.exists() and not args.force:
        raise IngestError(f"{path} exists; pass --force to replace it")
    if args.budget_gb <= 0:
        raise IngestError("--budget-gb must be positive")
    if args.out_bytes_per_frame < 0 or args.mp4_bytes_per_episode < 0:
        raise IngestError("byte estimates must not be negative")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise IngestError("--max-episodes must be positive")
    streams = list(dict.fromkeys(args.streams))
    header, records = read_manifest(args.manifest)
    source = header.get("source_revision")
    if source and source != args.revision:
        raise IngestError(
            f"{args.manifest} describes revision {source}, not --revision "
            f"{args.revision}"
        )
    kept = [r for r in records.values() if r.get("decision") == "keep"]
    if args.only_ids is not None:
        wanted = set(read_lines(args.only_ids))
        bad = sorted(i for i in wanted if records.get(i, {}).get("decision") != "keep")
        if bad:
            raise IngestError(
                f"{args.only_ids}: {len(bad)} id(s) the manifest does not keep, "
                f"e.g. {bad[:10]}"
            )
        kept = [r for r in kept if r["recording_id"] in wanted]
    kept = [r for r in kept if fnmatch.fnmatchcase(r["task"], args.tasks)]
    kept.sort(key=lambda r: id_key(r["recording_id"]))
    if args.max_episodes is not None:
        kept = kept[: args.max_episodes]
    if not kept:
        raise IngestError("no kept episode matches the filters")
    unnamed = [r["recording_id"] for r in kept if not r.get("episode_hash")]
    if unnamed:
        raise IngestError(f"kept episodes without an episode hash: {unnamed[:10]}")

    estimate = {
        "out_bytes_per_frame": args.out_bytes_per_frame,
        "mp4_bytes_per_episode": args.mp4_bytes_per_episode,
    }
    sizes = index_sizes(args.sizes)
    seasons: dict[tuple[str, str], list[dict]] = defaultdict(list)
    no_parquet = []
    for record in kept:
        files = sizes.get((record["task"], record["season"]), {}).get(
            int(record["episode_index"])
        )
        if files is None or files["parquet"] is None:
            no_parquet.append(record["recording_id"])
            continue
        listed = [files["parquet"]]
        missing = [s for s in streams if s not in files["videos"]]
        listed += [files["videos"][s] for s in streams if s in files["videos"]]
        seasons[(record["task"], record["season"])].append(
            {
                "id": record["recording_id"],
                "frames": int(record["length"]),
                "files": [[p, s] for p, s in listed],
                "in": sum(s for _, s in listed),
                "out": est_out(record, estimate),
                "missing": missing,
            }
        )
    if no_parquet:
        raise IngestError(
            f"{args.sizes}: no parquet listed for {len(no_parquet)} episode(s), "
            f"e.g. {no_parquet[:10]}"
        )
    budget = int(round(args.budget_gb * 1e9))
    chunks = []
    for n, episodes in enumerate(pack_chunks(list(seasons.values()), budget)):
        missing = [f"{e['id']}: {s}" for e in episodes for s in e["missing"]]
        for line in missing:
            logger.warning("no video in the listing for %s", line)
        chunks.append(
            {
                "chunk": f"chunk_{n:04d}",
                "episodes": len(episodes),
                "frames": sum(e["frames"] for e in episodes),
                "est_in": sum(e["in"] for e in episodes),
                "est_out": sum(e["out"] for e in episodes),
                "seasons": list(
                    dict.fromkeys(e["id"].rsplit("/", 1)[0] for e in episodes)
                ),
                "missing_videos": missing,
                "ids": [e["id"] for e in episodes],
                "files": [f for e in episodes for f in e["files"]],
            }
        )
    plan = {
        "header": {
            "tool": TOOL,
            "created_at": utc_now(),
            "git_sha": git_sha(),
            "repo_id": args.repo_id,
            "revision": args.revision,
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "sizes": str(Path(args.sizes).resolve()),
            "sizes_sha256": sha256_file(args.sizes),
            "budget_gb": args.budget_gb,
            "budget_bytes": budget,
            **estimate,
            "streams": streams,
            "filters": {
                "only_ids": None if args.only_ids is None else str(args.only_ids),
                "tasks": args.tasks,
                "max_episodes": args.max_episodes,
            },
            "totals": {
                "chunks": len(chunks),
                "episodes": sum(c["episodes"] for c in chunks),
                "frames": sum(c["frames"] for c in chunks),
                "est_in": sum(c["est_in"] for c in chunks),
                "est_out": sum(c["est_out"] for c in chunks),
                "missing_videos": sum(len(c["missing_videos"]) for c in chunks),
            },
        },
        "chunks": chunks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n")
    os.replace(tmp, path)
    totals = plan["header"]["totals"]
    print(
        f"Planned {totals['episodes']} episode(s), {totals['frames']} frames in "
        f"{totals['chunks']} chunk(s): {gb(totals['est_in'])} GB in, "
        f"~{gb(totals['est_out'])} GB out; {path}"
    )
    return 0


# ---------------------------------------------------------------------------
# Per-chunk stages: shared set-up
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    """What every per-chunk stage reads: the plan, its chunk, the manifest."""

    state_dir: Path
    plan: dict
    chunk: dict
    manifest: Path
    header: dict
    records: dict[str, dict]
    ledger: Ledger
    plan_sha256: str

    @property
    def name(self) -> str:
        return self.chunk["chunk"]


def open_stage(args) -> Stage:
    """Load the plan and the chunk; check the manifest against the plan."""
    path = plan_file(args)
    if not path.is_file():
        raise IngestError(f"no plan at {path}; run the plan subcommand first")
    raw = path.read_bytes()
    plan_sha256 = hashlib.sha256(raw).hexdigest()
    plan = json.loads(raw)
    chunks = {c["chunk"]: c for c in plan["chunks"]}
    if args.chunk not in chunks:
        names = list(chunks)
        raise IngestError(
            f"{args.chunk!r} is not in {path} ({names[0]} .. {names[-1]})"
        )
    chunk = chunks[args.chunk]
    manifest_path = Path(args.manifest or plan["header"]["manifest"])
    digest = sha256_file(manifest_path)
    planned = plan["header"]["manifest_sha256"]
    if digest != planned:
        raise IngestError(
            f"{manifest_path} is not the planned manifest (sha256 {digest[:12]}, "
            f"plan {planned[:12]}); pass that manifest or re-plan with --force"
        )
    header, records = read_manifest(manifest_path)
    return Stage(
        Path(args.state_dir),
        plan,
        chunk,
        manifest_path,
        header,
        records,
        Ledger(args.state_dir, digest),
        plan_sha256,
    )


def hash_gate(header: dict, allow: bool | None) -> bool:
    """Return True for a provisional-hash run; refuse one unless ``allow``.

    The hashes are checked only if the header says so and records both the
    SQL check and an S3 listing of every lab's folder. ``allow=None``
    (upload) has no override.
    """
    if (
        header.get("collision_check_complete") is True
        and header.get("sql_checked") is True
        and listing_covers_root(header.get("s3_listing"))
    ):
        return False
    if not allow:
        raise IngestError(
            "the manifest's episode hashes are provisional (its header does not "
            "show collision_check_complete with both the SQL check and an S3 "
            f"listing of s3://{BUCKET}/{S3_ROOT}); "
            + (
                "upload has no override"
                if allow is None
                else "pass --allow-provisional-hashes for a trial run"
            )
        )
    logger.warning("using provisional episode hashes (--allow-provisional-hashes)")
    return True


def output_dir(work_root: Path, chunk: str) -> Path:
    return Path(work_root) / "out" / chunk


def _named(path: Path) -> tuple | None:
    """``(vendor_recording_id, source_episode_id)`` a sidecar names, or None."""
    try:
        side = json.loads(Path(path).read_text())
        return side["vendor_recording_id"], side.get("source_episode_id")
    except (OSError, ValueError, TypeError, KeyError):
        return None


def episode_done(out_dir: Path, record: dict) -> bool:
    """Whether ``record``'s outputs are a finished, passing conversion.

    The sidecar must name the record's canonical and source ids, the store
    and MP4 must exist and the report must pass ``read_report``; the converter
    applies the same rule to ``--skip-done``.
    """
    from egomimic.scripts.sharpa_process.sharpa_to_zarr import read_report

    rid, name = record["recording_id"], record["episode_hash"]
    out_dir = Path(out_dir)
    ids = (sharpadex_records.canonical_id(rid), rid)
    if _named(out_dir / f"{name}.conversion.json") != ids:
        return False
    if not (out_dir / f"{name}.zarr" / "zarr.json").is_file():
        return False
    if not (out_dir / f"{name}.mp4").is_file():
        return False
    return not read_report(out_dir / f"{name}.validation.json", *ids)[1]


def file_id(rel: str) -> str:
    """Return the recording id of a planned episode file path."""
    parts = rel.split("/")
    match = _EPISODE_FILE.fullmatch(parts[-1])
    return f"{parts[0]}/{parts[1]}/{int(match[1])}"


def store_problems(store: Path) -> list[str]:
    """Problems of a store as it is now: main's episode validator, as the
    converter's ``--skip-done`` runs it (every array it lists, at full length)."""
    from egomimic.rldb.zarr.validate import validate_episode

    return validate_episode(store, allow_missing=True).errors


def artifact_inventory(out_dir: Path, name: str) -> dict:
    """An episode's artifacts as ``verify`` passed them and ``upload`` sends them.

    ``files`` maps every file of the store, the MP4 and the report (paths
    relative to ``out_dir``) to its size; ``sha256`` holds the digests of the
    store's root ``zarr.json`` and of the report. Bulk data is bound by size.
    """
    out_dir = Path(out_dir)
    files = {}
    for top, dirs, names in os.walk(out_dir / f"{name}.zarr"):
        dirs.sort()
        for file in sorted(names):
            path = Path(top) / file
            files[path.relative_to(out_dir).as_posix()] = path.stat().st_size
    for rel in (f"{name}.mp4", f"{name}.validation.json"):
        if (out_dir / rel).is_file():
            files[rel] = (out_dir / rel).stat().st_size
    digests = {
        rel: sha256_file(out_dir / rel)
        for rel in (f"{name}.zarr/zarr.json", f"{name}.validation.json")
        if rel in files
    }
    return {"files": files, "sha256": digests}


@contextlib.contextmanager
def chunk_io_lock(state_dir: Path, chunk: str):
    """Hold ``<chunk>.io.lock`` (``fcntl.flock``): one of verify, upload and
    clean at a time per chunk."""
    path = Path(state_dir) / f"{chunk}.io.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise IngestError(
                f"{path} is held: another verify, upload or clean of {chunk} is "
                "running"
            ) from exc
        yield


def receipt(stage: Stage, rid: str, inventory: dict) -> dict:
    """What an upload receipt of ``rid`` binds: plan, manifest, artifacts and
    exact destination."""
    name = stage.records[rid]["episode_hash"]
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode())
    return {
        "episode_hash": name,
        "recording_id": rid,
        "plan_sha256": stage.plan_sha256,
        "manifest_sha256": stage.ledger.manifest_sha256,
        "inventory_sha256": digest.hexdigest(),
        "destination": f"{DESTINATION}{name}.zarr",
    }


def matching_receipts(
    stage: Stage, out_dir: Path, inventories: dict | None = None
) -> dict[str, dict]:
    """Return ``{episode_hash: receipt}`` for the chunk's uploaded episodes.

    A receipt in ``<state-dir>/<chunk>.receipts.jsonl`` counts only if it
    names this plan and manifest, the episode's exact destination and the
    inventory its outputs have now (``inventories``, else computed).
    """
    inventories = dict(inventories or {})
    found = {}
    for entry in read_jsonl(stage.state_dir / f"{stage.name}.receipts.jsonl"):
        rid = entry.get("recording_id")
        name = entry.get("episode_hash")
        if rid not in stage.chunk["ids"] or name in found:
            continue
        if name not in inventories:
            inventories[name] = artifact_inventory(out_dir, name)
        want = receipt(stage, rid, inventories[name])
        if all(entry.get(key) == value for key, value in want.items()):
            found[name] = entry
    return found


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def download_file(repo_id: str, filename: str, revision: str, local_dir: str) -> None:
    """Download one release file into ``local_dir`` at ``revision``."""
    from huggingface_hub import hf_hub_download

    # Only missing or wrong-sized files get here; a local copy of the wrong
    # size can carry download metadata that would make the hub skip it.
    hf_hub_download(
        repo_id,
        filename,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        force_download=True,
    )


def retry_reason(exc: BaseException) -> str | None:
    """Return why a failed download is worth retrying (HTTP 429/5xx, network)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return f"HTTP {status}"
    name = type(exc).__name__
    if isinstance(exc, (ConnectionError, TimeoutError)) or any(
        word in name for word in ("Timeout", "ConnectionError")
    ):
        return name
    return None


def same_filesystem(a: Path, b: Path) -> bool:
    """Return whether ``b`` (or its nearest existing parent) is on ``a``'s device."""
    b = Path(b).absolute()
    while not b.exists() and b != b.parent:
        b = b.parent
    return os.stat(a).st_dev == os.stat(b).st_dev


def source_of(args, header: dict) -> tuple[str, str]:
    """Return the planned ``(repo_id, revision)``; refuse conflicting flags."""
    for flag, key in (("repo_id", "repo_id"), ("revision", "revision")):
        value = getattr(args, flag)
        if value is not None and value != header[key]:
            raise IngestError(
                f"--{flag.replace('_', '-')} {value} differs from the plan's "
                f"{header[key]}; the size listing describes the planned one"
            )
    return header["repo_id"], header["revision"]


def cmd_fetch(args) -> int:
    stage = open_stage(args)
    provisional = hash_gate(stage.header, args.allow_provisional_hashes)
    repo_id, revision = source_of(args, stage.plan["header"])
    root = Path(args.dataset_root).resolve()
    if not root.is_dir():
        raise IngestError(f"--dataset-root {root} is not a directory")
    if args.threads < 1 or args.retries < 0:
        raise IngestError("--threads must be positive and --retries not negative")

    todo, present, present_bytes = [], 0, 0
    for rel, size in stage.chunk["files"]:
        try:
            have = (root / rel).stat().st_size
        except FileNotFoundError:
            have = None
        if have == size:
            present += 1
            present_bytes += size
        else:
            todo.append((rel, size))
    need = sum(size for _, size in todo)
    # Outputs already written use disk already; count only the episodes left.
    out_dir = output_dir(args.work_root, stage.name)
    pending_out = sum(
        est_out(stage.records[rid], stage.plan["header"])
        for rid in stage.chunk["ids"]
        if not episode_done(out_dir, stage.records[rid])
    )
    shared = same_filesystem(root, args.work_root)
    free = shutil.disk_usage(root).free
    floor = int(round(args.floor_gb * 1e9))
    margin = free - need - (pending_out if shared else 0)
    guard = {
        "free": free,
        "to_download": need,
        "pending_out": pending_out,
        "work_root_same_fs": shared,
        "floor": floor,
        "margin": margin,
    }
    if margin < floor:
        stage.ledger.write(stage.name, "fetch", "fail", reason="disk_guard", **guard)
        raise IngestError(
            f"disk guard: {gb(free)} GB free under {root}, {gb(need)} GB to "
            "download"
            + (
                f" and ~{gb(pending_out)} GB of outputs to write there"
                if shared
                else ""
            )
            + f" would leave {gb(margin)} GB, below the {gb(floor)} GB floor "
            "(--floor-gb); free space or clean a finished chunk first",
            exit_code=EXIT_DISK,
        )
    stage.ledger.write(
        stage.name,
        "fetch",
        "start",
        files=len(todo),
        bytes=need,
        present=present,
        present_bytes=present_bytes,
        provisional_hashes=provisional,
        repo_id=repo_id,
        revision=revision,
        **guard,
    )
    logger.info(
        "%s: %d file(s) present, %d to download (%s GB)",
        stage.name,
        present,
        len(todo),
        gb(need),
    )

    record = stage.state_dir / f"{stage.name}.fetched.txt"
    lock = threading.Lock()
    retries: Counter = Counter()
    got = {"files": 0, "bytes": 0}

    def fetch_one(item: tuple[str, int]) -> str | None:
        rel, size = item
        for attempt in range(args.retries + 1):
            try:
                download_file(repo_id, rel, revision, str(root))
                break
            except Exception as exc:  # retried or reported for this file
                reason = retry_reason(exc)
                if reason is None or attempt == args.retries:
                    return f"{rel}: {type(exc).__name__}: {exc}"
                with lock:
                    retries[reason] += 1
                delay = min(args.backoff_seconds * 2**attempt, 300.0)
                time.sleep(delay * (1 + random.random() / 2))
        local = root / rel
        try:
            written = local.stat().st_size
        except FileNotFoundError:
            return f"{rel}: the download wrote no file"
        # Recorded at once, so a killed fetch can still be cleaned.
        with lock:
            with record.open("a") as f:
                f.write(f"{local}\n")
                f.flush()
                os.fsync(f.fileno())
            got["files"] += 1
            got["bytes"] += written
        if written != size:
            return f"{rel}: {written} bytes, the listing says {size}"
        return None

    failures: list[str] = []
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=args.threads)
    try:
        with sigterm_raises():
            for n, problem in enumerate(pool.map(fetch_one, todo), 1):
                if problem:
                    failures.append(problem)
                    logger.error("FAILED %s", problem)
                if n % 200 == 0:
                    logger.info("%d/%d files, %s GB", n, len(todo), gb(got["bytes"]))
    except (_Interrupted, KeyboardInterrupt) as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        stage.ledger.write(
            stage.name, "fetch", "fail", reason=f"interrupted ({exc!r})", **got
        )
        code = 143 if isinstance(exc, _Interrupted) else 130
        raise IngestError(
            f"fetch interrupted ({exc!r}); rerun to resume", code
        ) from exc
    pool.shutdown()
    seconds = time.monotonic() - started
    details = {
        "files": got["files"],
        "bytes": got["bytes"],
        "seconds": round(seconds, 1),
        "mb_per_s": round(got["bytes"] / 1e6 / seconds, 2) if seconds else None,
        "files_per_s": round(got["files"] / seconds, 2) if seconds else None,
        "retries": sum(retries.values()),
        "retry_reasons": dict(retries),
        "failures": len(failures),
        "failure_examples": failures[:20],
        "present": present,
        "provisional_hashes": provisional,
    }
    stage.ledger.write(stage.name, "fetch", "fail" if failures else "ok", **details)
    print(
        f"{stage.name}: downloaded {got['files']} file(s), {gb(got['bytes'])} GB in "
        f"{seconds:.0f} s, {details['retries']} retries, {len(failures)} failed; "
        f"{present} already present"
    )
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def parse_shard(text: str) -> tuple[int, int]:
    """Parse ``i/n`` (``0 <= i < n``)."""
    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if match is None or not int(match[1]) < int(match[2]):
        raise argparse.ArgumentTypeError(f"expected i/n with 0 <= i < n, not {text!r}")
    return int(match[1]), int(match[2])


def shard_tag(shard: tuple[int, int]) -> str:
    """File-name tag of a shard; empty for a whole chunk."""
    return "" if shard[1] == 1 else f"shard-{shard[0]}-of-{shard[1]}."


def cpu_sets(cpus: list[int], workers: int | None, per_worker: int) -> list[list[int]]:
    """Split ``cpus`` into disjoint sets of ``per_worker`` CPUs, one per worker."""
    if per_worker < 1:
        raise IngestError("--cpus-per-worker must be positive")
    fit = len(cpus) // per_worker
    if fit == 0:
        raise IngestError(
            f"{len(cpus)} CPU(s) available; --cpus-per-worker {per_worker} needs "
            f"at least {per_worker}"
        )
    workers = fit if workers is None else workers
    if not 1 <= workers <= fit:
        raise IngestError(
            f"--workers {workers} x --cpus-per-worker {per_worker} needs "
            f"{workers * per_worker} CPUs; {len(cpus)} available"
        )
    return [cpus[j * per_worker : (j + 1) * per_worker] for j in range(workers)]


def shard_ids(ids: list[str], shard: tuple[int, int]) -> list[str]:
    """Return this job's share of a chunk: every n-th id from the i-th."""
    return sorted(ids, key=id_key)[shard[0] :: shard[1]]


def balance(ids: list[str], lengths: dict[str, int], workers: int) -> list[list[str]]:
    """Deal ``ids`` to ``workers``: longest first, onto the least-loaded worker."""
    bins: list[list[str]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for rid in sorted(ids, key=lambda r: (-lengths[r], id_key(r))):
        j = min(range(workers), key=lambda k: (loads[k], k))
        bins[j].append(rid)
        loads[j] += lengths[rid]
    return [sorted(b, key=id_key) for b in bins]


def squeue_lists(job_id: str) -> bool | None:
    """Return whether Slurm still lists ``job_id``, or None if it cannot tell."""
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%i %T"],
            capture_output=True,
            text=True,
            timeout=60,
            env=child_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return bool(result.stdout.strip())
    if "Invalid job id" in result.stderr:
        return False
    return None


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def holder_alive(info: dict) -> bool:
    """Return whether a lock's holder may still be running.

    On this host the pid decides. On another host Slurm decides; a lock from
    this very job (a requeued job keeps its id) is stale. A holder that can
    not be checked (no job id, or no answer from squeue) counts as alive.
    """
    if info.get("host") == socket.gethostname():
        return _pid_alive(info.get("pid"))
    job = info.get("slurm_job_id")
    if job and job == os.environ.get("SLURM_JOB_ID"):
        return False
    if job:
        listed = squeue_lists(job)
        return True if listed is None else listed
    return True


def lock_files(state_dir: Path, chunk: str) -> list[Path]:
    """Return every lock of ``chunk``: the whole-chunk lock and shard locks."""
    state_dir = Path(state_dir)
    return sorted(state_dir.glob(f"{chunk}.lock")) + sorted(
        state_dir.glob(f"{chunk}.shard-*.lock")
    )


def read_lock(path: Path) -> dict | None:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        info = json.loads(text)
    except json.JSONDecodeError:
        info = None
    # An unreadable lock may be one being written: it names no holder, so it
    # counts as held.
    return info if isinstance(info, dict) else {"unreadable": str(path)}


def describe_lock(info: dict) -> str:
    return (
        f"host {info.get('host')} pid {info.get('pid')} job "
        f"{info.get('slurm_job_id')} since {info.get('started_at')}"
    )


def clear_lock(path: Path, break_lock: bool) -> None:
    """Remove a lock whose holder is gone (any lock with ``break_lock``)."""
    info = read_lock(path)
    if info is None:
        return
    if holder_alive(info):
        if not break_lock:
            raise IngestError(
                f"{path.name} is held by {describe_lock(info)}; wait for it, or "
                "pass --break-lock if that run is gone"
            )
        logger.warning("breaking the lock %s (%s)", path, describe_lock(info))
    else:
        logger.warning("replacing the stale lock %s (%s)", path, describe_lock(info))
    path.unlink(missing_ok=True)


def acquire_lock(
    state_dir: Path, chunk: str, shard: tuple[int, int], break_lock: bool
) -> Path:
    """Take ``<chunk>.lock`` (``<chunk>.shard-i-of-n.lock`` for a shard).

    Another shard of the same ``n`` covers other ids and may run at once;
    any other lock of the chunk must be stale (or broken).
    """
    path = Path(state_dir) / f"{chunk}.{shard_tag(shard)}lock"
    for other in lock_files(state_dir, chunk):
        theirs = _SHARD_LOCK.search(other.name)
        if (
            other != path
            and theirs is not None
            and shard[1] > 1
            and int(theirs[2]) == shard[1]
        ):
            continue
        clear_lock(other, break_lock)
    info = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": utc_now(),
        "chunk": chunk,
        "shard": f"{shard[0]}/{shard[1]}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise IngestError(f"{path} was taken by another run just now") from exc
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(info) + "\n")
    return path


def release_lock(path: Path) -> None:
    info = read_lock(path)
    if (
        info
        and info.get("pid") == os.getpid()
        and info.get("host") == socket.gethostname()
    ):
        path.unlink(missing_ok=True)


def worker_env(per_worker: int) -> dict[str, str]:
    """Environment of one converter: its thread pools sized to its CPUs.

    It is offline and gets no credentials (``child_env``).
    """
    env = child_env()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(per_worker)
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    # The interpreter's installed egomimic may be another checkout's.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p
    )
    return env


def converter_args(stage: Stage, args, ids_file: Path, out_dir: Path) -> list[str]:
    base = (
        shlex.split(args.converter_cmd)
        if args.converter_cmd
        else [args.python, "-m", CONVERTER]
    )
    head_right = "head_right" in stage.plan["header"]["streams"]
    return [
        *base,
        "--dataset-root",
        str(Path(args.dataset_root).resolve()),
        "--manifest",
        str(stage.manifest.resolve()),
        "--recording-ids",
        str(ids_file),
        "--skip-done",
        "--overwrite",
        *(["--include-head-right"] if head_right else []),
        "--output-dir",
        str(out_dir),
    ]


def _count_done(out_dir: Path, records: dict, ids: list[str], seen: dict) -> None:
    """Add ``{id: frames}`` to ``seen`` for ids whose ``conversion.json`` appeared."""
    for rid in ids:
        if rid in seen:
            continue
        path = out_dir / f"{records[rid]['episode_hash']}.conversion.json"
        if not path.exists():
            continue
        try:
            side = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # still being written
        ids = (sharpadex_records.canonical_id(rid), rid)
        if (side.get("vendor_recording_id"), side.get("source_episode_id")) == ids:
            seen[rid] = int(side.get("total_frames") or 0)


def _stop(procs: list[subprocess.Popen], grace: float = 30.0) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + grace
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def cmd_convert(args) -> int:
    if args.monitor_seconds <= 0:
        raise IngestError("--monitor-seconds must be positive")
    stage = open_stage(args)
    provisional = hash_gate(stage.header, args.allow_provisional_hashes)
    sets = cpu_sets(sorted(os.sched_getaffinity(0)), args.workers, args.cpus_per_worker)
    load = load_sample()
    if load["ratio"] > args.max_load_ratio:
        message = (
            f"node load {load['load1']} on {load['cpu_count']} CPUs (ratio "
            f"{load['ratio']}) exceeds --max-load-ratio {args.max_load_ratio}; "
            "an oversubscribed node converts many times slower"
        )
        if not args.ignore_load:
            raise IngestError(message + "; pass --ignore-load to run anyway")
        logger.warning("%s (--ignore-load)", message)
    root = Path(args.dataset_root).resolve()
    out_dir = output_dir(args.work_root, stage.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard, tag = args.shard, shard_tag(args.shard)
    lock = acquire_lock(stage.state_dir, stage.name, shard, args.break_lock)
    try:
        return _convert_locked(stage, args, sets, load, provisional, root, out_dir, tag)
    finally:
        release_lock(lock)


def _convert_locked(stage, args, sets, load, provisional, root, out_dir, tag) -> int:
    records = stage.records
    mine = shard_ids(stage.chunk["ids"], args.shard)
    done = {rid for rid in mine if episode_done(out_dir, records[rid])}
    todo = [rid for rid in mine if rid not in done]
    wanted = set(todo)
    absent = [
        rel
        for rel, size in stage.chunk["files"]
        if file_id(rel) in wanted
        and not ((root / rel).is_file() and (root / rel).stat().st_size == size)
    ]
    if absent:
        raise IngestError(
            f"{len(absent)} input file(s) missing or incomplete under {root}, e.g. "
            f"{absent[:3]}; run fetch first"
        )
    workers = min(len(sets), len(todo))
    shard_label = f"{args.shard[0]}/{args.shard[1]}"
    stage.ledger.write(
        stage.name,
        "convert",
        "start",
        shard=shard_label,
        ids=len(mine),
        done=len(done),
        todo=len(todo),
        workers=workers,
        cpus_per_worker=args.cpus_per_worker,
        cpus=[c for s in sets[:workers] for c in s],
        load=load,
        provisional_hashes=provisional,
        repo_root=str(REPO_ROOT),
        converter=args.converter_cmd or f"{args.python} -m {CONVERTER}",
    )
    if not todo:
        stage.ledger.write(
            stage.name, "convert", "ok", shard=shard_label, converted=0, done=len(done)
        )
        print(f"{stage.name}: all {len(mine)} episode(s) already converted")
        return 0
    lengths = {rid: int(records[rid]["length"]) for rid in todo}
    bins = balance(todo, lengths, workers)
    ids_dir = stage.state_dir / stage.name
    logs_dir = stage.state_dir / "logs" / stage.name
    ids_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = worker_env(args.cpus_per_worker)

    procs: list[subprocess.Popen] = []
    logs, summary = [], []
    seen: dict[str, int] = {}
    samples = [load]
    started = time.monotonic()
    try:
        with sigterm_raises():
            for j, (ids, cpus) in enumerate(zip(bins, sets)):
                ids_file = ids_dir / f"{tag}worker_{j}.ids"
                ids_file.write_text("".join(f"{rid}\n" for rid in ids))
                cmd = converter_args(stage, args, ids_file, out_dir)
                log_path = logs_dir / f"{tag}worker_{j}.log"
                log = log_path.open("a")
                logs.append(log)
                log.write(f"# {utc_now()} cpus {cpus}: {shlex.join(cmd)}\n")
                log.flush()
                procs.append(
                    subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=REPO_ROOT,
                        preexec_fn=functools.partial(os.sched_setaffinity, 0, cpus),
                    )
                )
                summary.append(
                    {"worker": j, "cpus": cpus, "ids": len(ids), "log": str(log_path)}
                )
            logger.info(
                "%s: %d episode(s) on %d worker(s) x %d CPUs (%d already done)",
                stage.name,
                len(todo),
                workers,
                args.cpus_per_worker,
                len(done),
            )
            next_sample = started + args.monitor_seconds
            while any(p.poll() is None for p in procs):
                time.sleep(min(1.0, args.monitor_seconds))
                now = time.monotonic()
                if now < next_sample:
                    continue
                next_sample = now + args.monitor_seconds
                sample = load_sample()
                samples.append(sample)
                _count_done(out_dir, records, todo, seen)
                frames = sum(seen.values())
                try:
                    stage.ledger.write(
                        stage.name,
                        "convert",
                        "progress",
                        shard=shard_label,
                        load=sample,
                        episodes_done=len(seen),
                        frames_done=frames,
                        frames_per_s=round(frames / (now - started), 2),
                        workers_running=sum(p.poll() is None for p in procs),
                    )
                except OSError as exc:  # a missed sample must not stop workers
                    logger.warning("cannot record progress: %s", exc)
                if sample["ratio"] > args.max_load_ratio:
                    logger.warning(
                        "node load ratio %s exceeds %s: conversion slows down",
                        sample["ratio"],
                        args.max_load_ratio,
                    )
    except (
        _Interrupted,
        KeyboardInterrupt,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        _stop(procs)
        for item, proc in zip(summary, procs):
            item["exit"] = proc.returncode
        stage.ledger.write(
            stage.name,
            "convert",
            "fail",
            shard=shard_label,
            reason=f"stopped ({exc!r})",
            workers=summary,
        )
        # 143 = 128 + SIGTERM, 130 = 128 + SIGINT; 1 for a worker that would
        # not start.
        code = {_Interrupted: 143, KeyboardInterrupt: 130}.get(type(exc), 1)
        raise IngestError(f"convert stopped ({exc!r}); rerun to resume", code) from exc
    finally:
        for log in logs:
            log.close()

    seconds = time.monotonic() - started
    for item, proc in zip(summary, procs):
        item["exit"] = proc.returncode
    _count_done(out_dir, records, todo, seen)
    missing = [rid for rid in todo if rid not in seen]
    failed = [item for item in summary if item["exit"] not in CONVERTED]
    frames = sum(seen.values())
    ok = not failed and not missing
    stage.ledger.write(
        stage.name,
        "convert",
        "ok" if ok else "fail",
        shard=shard_label,
        workers=summary,
        converted=len(seen),
        frames=frames,
        seconds=round(seconds, 1),
        frames_per_s=round(frames / seconds, 2) if seconds else None,
        not_done=len(missing),
        not_done_examples=missing[:20],
        load_max_ratio=max(s["ratio"] for s in samples),
    )
    print(
        f"{stage.name}: converted {len(seen)}/{len(todo)} episode(s), {frames} "
        f"frames in {seconds:.0f} s; worker exit codes "
        f"{[item['exit'] for item in summary]}; logs in {logs_dir}"
    )
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _reported_hashes(state_dir: Path, chunk: str) -> dict[str, str]:
    """Return ``{episode_hash: chunk}`` from every other chunk's verify report."""
    out = {}
    for path in sorted(Path(state_dir).glob("*.report.json")):
        name = path.name.removesuffix(".report.json")
        if name == chunk:
            continue
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            logger.warning("cannot read %s; its hashes are not compared", path)
            continue
        for name_hash in report.get("hashes", {}):
            out.setdefault(name_hash, name)
    return out


def verify_chunk(stage: Stage, out_dir: Path) -> dict:
    """Check a converted chunk's sidecars and outputs; return the report.

    An episode passes only if its sidecar, store and report name its
    canonical and source ids, ``read_report`` finds no problem (so it is
    ``complete``, with every check passed), its ``lab`` is ``LAB`` and its
    licence the converter's, the store as it is now passes ``store_problems``,
    its MP4 is not empty and its name is unique. There is no way to accept a
    non-complete episode. The report records each episode's
    ``artifact_inventory`` and the plan's sha256.
    """
    from egomimic.scripts.sharpa_process.sharpa_to_zarr import LICENSE, read_report

    problems: dict[str, list[str]] = defaultdict(list)
    status, reasons, limitations = Counter(), Counter(), Counter()
    errors, warnings = Counter(), Counter()
    error_examples: dict[str, list] = defaultdict(list)
    tcp_rot, tcp_swap, tcp_pos = [], [], []
    annotations = Counter()
    array_bytes: Counter = Counter()
    hashes: dict[str, str] = {}
    labs: dict[str, str | None] = {}
    inventories: dict[str, dict] = {}
    non_complete = []
    frames = mp4_bytes = 0
    seconds = 0.0
    for rid in stage.chunk["ids"]:
        name = stage.records[rid]["episode_hash"]
        ids = (sharpadex_records.canonical_id(rid), rid)
        conv = out_dir / f"{name}.conversion.json"
        try:
            side = json.loads(conv.read_text())
        except FileNotFoundError:
            problems[rid].append(f"no {conv.name}")
            continue
        except ValueError as exc:
            problems[rid].append(f"unreadable {conv.name}: {exc}")
            continue
        named = (side.get("vendor_recording_id"), side.get("source_episode_id"))
        if named != ids:
            problems[rid].append(f"{conv.name} is for {named}, expected {ids}")
            continue
        if name in hashes:
            problems[rid].append(f"episode hash {name} also used by {hashes[name]}")
        hashes[name] = rid
        report = (side.get("attributes") or {}).get("conversion_report") or {}

        verification, found = read_report(out_dir / f"{name}.validation.json", *ids)
        for message in found:
            check = message.split(":", 1)[0]
            errors[check] += 1
            if len(error_examples[check]) < 3:
                error_examples[check].append([rid, message[:200]])
        if found:
            problems[rid] += found
        for warning in (verification or {}).get("warnings", []):
            # "validator: <key>: ..." -> "<key>"
            parts = warning.split(": ")
            warnings[parts[1] if len(parts) > 2 else parts[0]] += 1

        data_status = side.get("data_status")
        status[data_status] += 1
        why = list(report.get("data_status_reasons") or [])
        reasons.update(why)
        limitations.update(report.get("data_limitations") or [])
        if data_status != "complete":
            non_complete.append(
                {"id": rid, "hash": name, "data_status": data_status, "reasons": why}
            )
            problems[rid].append(f"data_status {data_status}: {why}")

        tcp = ((verification or {}).get("checks") or {}).get("tcp_check") or {}
        for values in (tcp.get("sides") or {}).values():
            tcp_rot.append(values.get("rotation_error_deg_median"))
            tcp_swap.append(values.get("swapped_rotation_error_deg_median"))
            tcp_pos.append(values.get("position_error_m_p95"))

        counts = report.get("annotations") or {}
        annotations["episodes"] += 1
        annotations["with_subtasks"] += bool(counts.get("subtasks"))
        annotations["with_skills"] += bool(counts.get("skills"))

        store = out_dir / f"{name}.zarr"
        try:
            attrs = json.loads((store / "zarr.json").read_text())["attributes"]
        except (OSError, ValueError, KeyError, TypeError):
            problems[rid].append(f"no readable {store.name}/zarr.json")
        else:
            stored = (attrs.get("vendor_recording_id"), attrs.get("source_episode_id"))
            if stored != ids:
                problems[rid].append(f"{store.name} is for {stored}, expected {ids}")
            labs[name] = attrs.get("lab")
            if labs[name] != LAB:
                problems[rid].append(f"{store.name} has lab {labs[name]!r}, not {LAB}")
            if attrs.get("license") != LICENSE:
                problems[rid].append(
                    f"{store.name} has license {attrs.get('license')!r}, not {LICENSE}"
                )
            # The report describes the store as written; check it as it is.
            if broken := store_problems(store):
                problems[rid] += [f"store: {e}" for e in broken]
            for entry in os.scandir(store):
                if entry.is_dir(follow_symlinks=False):
                    array_bytes[entry.name] += du(Path(entry.path))
        mp4 = out_dir / f"{name}.mp4"
        if mp4.is_file() and mp4.stat().st_size:
            mp4_bytes += mp4.stat().st_size
        else:
            problems[rid].append(f"no {mp4.name} (or it is empty)")
        inventories[name] = artifact_inventory(out_dir, name)
        frames += int(side.get("total_frames") or 0)
        seconds += float(side.get("seconds") or 0.0)

    collisions = {}
    others = _reported_hashes(stage.state_dir, stage.name)
    for name, rid in hashes.items():
        if name in others:
            collisions[name] = others[name]
            problems[rid].append(f"episode hash {name} is also in {others[name]}")

    zarr_bytes = sum(array_bytes.values())
    per_frame = {k: v / frames for k, v in array_bytes.items()} if frames else {}
    grouped = {
        g: round(sum(per_frame.get(k, 0) for k in keys))
        for g, keys in ARRAY_GROUPS.items()
    }
    grouped["numeric_and_other"] = round(sum(per_frame.values())) - sum(
        grouped.values()
    )
    checked = annotations["episodes"]

    def stat(values, fn):
        values = [v for v in values if v is not None]
        return fn(values) if values else None

    return {
        "chunk": stage.name,
        "created_at": utc_now(),
        "git_sha": git_sha(),
        "manifest_sha256": stage.ledger.manifest_sha256,
        "plan_sha256": stage.plan_sha256,
        "passed": not problems,
        "episodes": len(stage.chunk["ids"]),
        "checked": checked,
        "failed": len(problems),
        "problems": dict(problems),
        "non_complete": non_complete,
        "hash_collisions_with_other_chunks": collisions,
        "data_status": dict(status),
        "reasons": dict(reasons),
        "limitations": dict(limitations),
        "validator_errors_by_check": dict(errors),
        "validator_error_examples": dict(error_examples),
        "validator_warnings_by_check": dict(warnings),
        "tcp": {
            "rotation_error_deg_median_max": stat(tcp_rot, max),
            "swapped_rotation_error_deg_min": stat(tcp_swap, min),
            "position_error_m_p95_max": stat(tcp_pos, max),
        },
        "annotations": dict(annotations),
        "frames": frames,
        "worker_seconds": round(seconds, 1),
        "worker_seconds_per_frame": round(seconds / frames, 4) if frames else None,
        "bytes": {
            "zarr": zarr_bytes,
            "mp4": mp4_bytes,
            "est_out": stage.chunk["est_out"],
            "zarr_per_frame": round(zarr_bytes / frames) if frames else None,
            "mp4_per_episode": round(mp4_bytes / checked) if checked else None,
            "per_frame_by_group": grouped,
            "per_frame_by_array": {
                k: round(v) for k, v in sorted(per_frame.items(), key=lambda kv: -kv[1])
            },
        },
        "hashes": hashes,
        "labs": labs,
        # What upload sends: it refuses an episode whose files differ now.
        "inventories": inventories,
    }


def cmd_verify(args) -> int:
    stage = open_stage(args)
    with chunk_io_lock(stage.state_dir, stage.name):
        return _verify_locked(stage, Path(args.work_root).resolve())


def _verify_locked(stage: Stage, work_root: Path) -> int:
    report = verify_chunk(stage, output_dir(work_root, stage.name))
    # Upload must find these very outputs: the resolved work root is recorded.
    report["work_root"] = str(work_root)
    path = stage.state_dir / f"{stage.name}.report.json"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(report, indent=1) + "\n")
    os.replace(tmp, path)
    summary = {
        k: report[k]
        for k in (
            "episodes",
            "checked",
            "failed",
            "frames",
            "data_status",
            "annotations",
        )
    }
    stage.ledger.write(
        stage.name,
        "verify",
        "ok" if report["passed"] else "fail",
        report=str(path),
        failed_examples=dict(list(report["problems"].items())[:10]),
        **summary,
    )
    for rid, found in list(report["problems"].items())[:20]:
        logger.error("%s: %s", rid, "; ".join(found))
    print(
        f"{stage.name}: {'PASSED' if report['passed'] else 'FAILED'}: "
        f"{report['checked']}/{report['episodes']} episode(s) checked, "
        f"{report['failed']} with problems; {report['frames']} frames; {path}"
    )
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def uploaded_files(inventory: dict, name: str, roots: bool) -> dict[str, int]:
    """The files of a verified inventory that ``upload`` sends, by path.

    Everything but the report; without ``roots``, not the store's root files
    either (its ``zarr.json``, which staging reads the episode from).
    """
    return {
        rel: size
        for rel, size in inventory["files"].items()
        if rel != f"{name}.validation.json"
        and (roots or rel.count("/") != 1 or not rel.startswith(f"{name}.zarr/"))
    }


def upload_batches(
    out_dir: Path, names: list[str], inventories: dict[str, dict]
) -> tuple[list[str], list[str]]:
    """Return the two ``s5cmd run`` batches that upload ``names``.

    One ``cp`` per verified file: the first batch holds the arrays and the
    MP4s, the second the stores' root files, which ``upload`` sends only for
    episodes whose first-batch files it has found remotely.
    """
    data, roots = [], []
    for name in names:
        rest = uploaded_files(inventories[name], name, roots=False)
        for rel in uploaded_files(inventories[name], name, roots=True):
            line = f'cp "{Path(out_dir) / rel}" "{DESTINATION}{rel}"'
            (data if rel in rest else roots).append(line)
    return data, roots


def remote_files(listing: str) -> dict[str, int]:
    """``{path below DESTINATION: size}`` from ``s5cmd --json ls`` output.

    s5cmd omits ``size`` for an empty object.
    """
    found = {}
    for line in listing.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        key = str(entry.get("key") or "") if isinstance(entry, dict) else ""
        if key.startswith(DESTINATION):
            found[key[len(DESTINATION) :]] = int(entry.get("size") or 0)
    return found


def cmd_upload(args) -> int:
    requested = f"s3://{args.bucket}/{args.prefix.strip('/')}/"
    if requested != DESTINATION:
        raise IngestError(
            f"--bucket/--prefix name {requested}; SharpaDex uploads only to "
            f"{DESTINATION}, the folder staging takes the lab {LAB!r} from"
        )
    stage = open_stage(args)
    hash_gate(stage.header, None)
    with chunk_io_lock(stage.state_dir, stage.name):
        return _upload_locked(stage, args, Path(args.work_root).resolve())


def verified(stage: Stage, work_root: Path) -> tuple[dict, dict[str, dict]]:
    """Return the chunk's passing verify report and its episodes' inventories.

    The report must be of this plan, manifest and (resolved) work root, and
    every episode's files must still be what verify passed.
    """
    path = stage.state_dir / f"{stage.name}.report.json"
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise IngestError(f"no {path.name}; run verify first") from exc
    if not report.get("passed"):
        raise IngestError(f"{path.name} did not pass; fix the chunk and rerun verify")
    if (
        report.get("manifest_sha256") != stage.ledger.manifest_sha256
        or report.get("plan_sha256") != stage.plan_sha256
        or sorted(report.get("hashes", {}).values()) != sorted(stage.chunk["ids"])
    ):
        raise IngestError(f"{path.name} is from another manifest or plan; rerun verify")
    if report.get("work_root") != str(work_root):
        raise IngestError(
            f"{path.name} verified the outputs under {report.get('work_root')}, "
            f"not {work_root}; pass that --work-root or rerun verify"
        )
    out_dir = output_dir(work_root, stage.name)
    recorded = report.get("inventories") or {}
    inventories = {name: artifact_inventory(out_dir, name) for name in report["hashes"]}
    changed = sorted(n for n, inv in inventories.items() if recorded.get(n) != inv)
    if changed:
        raise IngestError(
            f"{len(changed)} episode(s) changed since verify, e.g. {changed[:3]}; "
            "rerun verify"
        )
    return report, inventories


def _upload_locked(stage: Stage, args, work_root: Path) -> int:
    report, inventories = verified(stage, work_root)
    out_dir = output_dir(work_root, stage.name)
    names = sorted(report["hashes"])
    # The merge script sets lab from the folder; it must be the stored one.
    labs = report.get("labs") or {}
    wrong = sorted({str(labs.get(n)) for n in names if labs.get(n) != LAB})
    if wrong:
        raise IngestError(
            f"the episodes' lab is {sorted(set(map(str, labs.values()))) or [None]}, "
            f"not {LAB!r} for every episode; the folder must equal the lab"
        )
    receipts = matching_receipts(stage, out_dir, inventories)
    todo = [n for n in names if n not in receipts]
    data, roots = upload_batches(out_dir, todo, inventories)
    batches = []
    for n, lines in ((1, data), (2, roots)):
        batch = stage.state_dir / f"{stage.name}.upload.phase{n}.txt"
        batch.write_text("".join(f"{line}\n" for line in lines))
        batches.append(batch)
    target = DESTINATION
    if not args.execute:
        stage.ledger.write(
            stage.name,
            "upload_dry_run",
            "ok",
            target=target,
            episodes=len(names),
            todo=len(todo),
            batches=[str(b) for b in batches],
        )
        print(
            f"{stage.name}: dry run; {len(receipts)} episode(s) already uploaded; "
            f"{len(todo)} episode(s) to {target}: "
            f"{len(data)} array and MP4 copies in {batches[0]}, then, once those "
            f"are confirmed remotely, {len(roots)} root files in {batches[1]}. "
            "Upload with --execute --licence-approved once the leads approve "
            "the licence."
        )
        return 0
    if not args.licence_approved:
        raise IngestError(
            "--execute needs --licence-approved: SharpaDex is CC-BY-NC-4.0 and "
            "the leads must approve it first"
        )
    needed = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    unset = [k for k in needed if not os.environ.get(k)]
    if unset:
        raise IngestError(f"--execute needs {', '.join(unset)} in the environment")
    s5cmd = shutil.which("s5cmd")
    if s5cmd is None:
        raise IngestError("--execute needs s5cmd on PATH")
    # s5cmd gets the R2 keys and nothing else of this environment.
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    env.update(
        AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
        AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"],
        AWS_DEFAULT_REGION="auto",
        AWS_REGION="auto",
    )
    endpoint = ["--endpoint-url", os.environ["R2_ENDPOINT_URL"]]
    workers = ["--numworkers", str(args.numworkers)]

    def run(batch: Path, lines: list[str] | None = None, listing: bool = False):
        if lines is not None:
            batch.write_text("".join(f"{line}\n" for line in lines))
        flags = ["--json"] if listing else []
        return subprocess.run(
            [s5cmd, *flags, *endpoint, *workers, "run", str(batch)],
            env=env,
            capture_output=listing,
            text=True,
        )

    check = stage.state_dir / f"{stage.name}.upload.confirm.txt"

    def confirmed(episodes: list[str], roots: bool) -> list[str]:
        """The episodes whose files are all remote, at their verified sizes."""
        if not episodes:
            return []
        lines = []
        for name in episodes:
            lines += [
                f'ls "{DESTINATION}{name}.zarr/*"',
                f'ls "{DESTINATION}{name}.mp4"',
            ]
        remote = remote_files(run(check, lines, listing=True).stdout)
        return [
            name
            for name in episodes
            if all(
                remote.get(rel) == size
                for rel, size in uploaded_files(inventories[name], name, roots).items()
            )
        ]

    stage.ledger.write(
        stage.name,
        "upload",
        "start",
        target=target,
        episodes=len(names),
        todo=len(todo),
    )
    started = time.monotonic()
    # Arrays and MP4s first; a root goes up only once the rest of its episode
    # is confirmed, so staging never reads an incomplete episode's metadata.
    if data and (code := run(batches[0]).returncode) != 0:
        stage.ledger.write(stage.name, "upload", "fail", phase=1, exit=code)
        print(f"{stage.name}: s5cmd phase 1 failed; rerun upload to retry")
        return 1
    ready = confirmed(todo, roots=False)
    _, root_lines = upload_batches(out_dir, ready, inventories)
    if root_lines and (code := run(batches[1], root_lines).returncode) != 0:
        stage.ledger.write(stage.name, "upload", "fail", phase=2, exit=code)
        print(f"{stage.name}: s5cmd phase 2 failed; rerun upload to retry")
        return 1
    done = set(confirmed(ready, roots=True))
    by_name = {stage.records[rid]["episode_hash"]: rid for rid in stage.chunk["ids"]}
    with (stage.state_dir / f"{stage.name}.receipts.jsonl").open("a") as f:
        for name in sorted(done):
            entry = receipt(stage, by_name[name], inventories[name])
            f.write(json.dumps({**entry, "ts": utc_now()}, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    missing = [n for n in todo if n not in done]
    stage.ledger.write(
        stage.name,
        "upload",
        "fail" if missing else "ok",
        target=target,
        episodes=len(names),
        already_uploaded=len(receipts),
        uploaded=len(done),
        confirmed=not missing,
        missing=missing[:20],
        seconds=round(time.monotonic() - started, 1),
    )
    print(
        f"{stage.name}: uploaded {len(done)} of {len(todo)} episode(s) to {target}; "
        f"{len(missing)} not confirmed remotely (their roots are not published "
        "unless the rest of the episode is); rerun upload to retry"
        if missing
        else f"{stage.name}: uploaded and confirmed {len(done)} episode(s) to "
        f"{target}; {len(receipts)} were already uploaded"
    )
    return 1 if missing else 0


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def _protected(path: Path, root: Path) -> str | None:
    """Return why ``clean`` must not delete ``path``, or None."""
    try:
        rel = path.resolve().relative_to(root)
    except (ValueError, OSError):
        return "outside --dataset-root"
    parts = rel.parts
    if "meta" in parts or parts[-1:] == ("README.md",):
        return "metadata"
    if len(parts) < 5 or parts[2] not in V21_DIRS or parts[3] not in ("data", "videos"):
        return "not an episode data or video file"
    return None


def cmd_clean(args) -> int:
    stage = open_stage(args)
    with chunk_io_lock(stage.state_dir, stage.name):
        return _clean_locked(stage, args)


def _clean_locked(stage: Stage, args) -> int:
    for path in lock_files(stage.state_dir, stage.name):
        clear_lock(path, args.break_lock)
    root = Path(args.dataset_root).resolve()
    record = stage.state_dir / f"{stage.name}.fetched.txt"
    listed = list(dict.fromkeys(read_lines(record))) if record.is_file() else []
    freed = deleted = 0
    kept: Counter = Counter()
    parents = set()
    for line in listed:
        path = Path(line)
        why = _protected(path, root)
        if why:
            kept[why] += 1
            logger.warning("not deleting %s: %s", path, why)
            continue
        try:
            size = path.lstat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        freed += size
        deleted += 1
        parents.add(path.parent)
    for parent in sorted(parents, key=lambda p: -len(p.parts)):
        while parent != root and root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    out_dir = output_dir(Path(args.work_root).resolve(), stage.name)
    for partial in sorted(out_dir.glob(".*.partial")):
        freed += du(partial)
        if partial.is_dir():
            shutil.rmtree(partial)
        else:
            partial.unlink()
    outputs, uploaded = "absent", {}
    if out_dir.exists() and args.discard_outputs:
        freed += du(out_dir)
        shutil.rmtree(out_dir)
        outputs = "discarded"
    elif out_dir.exists():
        # Only an episode whose current outputs a receipt shows uploaded.
        uploaded = matching_receipts(stage, out_dir)
        for name in uploaded:
            for kind in ("zarr", "mp4", "validation.json", "conversion.json"):
                path = out_dir / f"{name}.{kind}"
                freed += du(path)
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            out_dir.rmdir()
        if not out_dir.exists():
            outputs = "deleted (uploaded)"
        else:
            outputs = "partly deleted (uploaded)" if uploaded else "kept"
            logger.warning(
                "keeping the outputs in %s without a matching upload receipt "
                "(pass --discard-outputs to delete them anyway)",
                out_dir,
            )
    stage.ledger.write(
        stage.name,
        "clean",
        "ok",
        deleted=deleted,
        freed=freed,
        protected=dict(kept),
        outputs=outputs,
        uploaded_deleted=len(uploaded),
    )
    print(
        f"{stage.name}: deleted {deleted} fetched file(s); freed {gb(freed)} GB; "
        f"outputs {outputs}"
    )
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _cell(entries: list[dict], chunk: str, stage: str) -> str:
    entry = last_entry(entries, chunk, stage)
    if stage == "upload" and entry is None:
        return "dry-run" if last_entry(entries, chunk, "upload_dry_run") else "-"
    if entry is None:
        return "-"
    shard = (entry.get("details") or {}).get("shard")
    text = "started" if entry["state"] == "start" else entry["state"]
    if (
        stage == "upload"
        and entry["state"] == "ok"
        and not upload_confirmed(entries, chunk)
    ):
        text = "unconfirmed"
    return f"{text}({shard})" if shard and shard != "0/1" else text


def cmd_status(args) -> int:
    path = plan_file(args)
    if not path.is_file():
        raise IngestError(f"no plan at {path}")
    plan = json.loads(path.read_text())
    entries = read_ledger(args.state_dir)
    columns = ["chunk", "episodes", "frames", "in_GB", "out_GB", *STAGES]
    rows = []
    done = Counter()
    for chunk in plan["chunks"]:
        name = chunk["chunk"]
        cells = [_cell(entries, name, s) for s in STAGES]
        rows.append(
            [
                name,
                str(chunk["episodes"]),
                str(chunk["frames"]),
                gb(chunk["est_in"]),
                gb(chunk["est_out"]),
                *cells,
            ]
        )
        for stage, cell in zip(STAGES, cells):
            if cell.startswith("ok"):
                done[stage] += 1
                done[f"{stage}_episodes"] += chunk["episodes"]
                done[f"{stage}_frames"] += chunk["frames"]
    if args.json:
        print(
            json.dumps({"columns": columns, "rows": rows, "done": dict(done)}, indent=1)
        )
        return 0
    widths = [max(len(r[i]) for r in [columns, *rows]) for i in range(len(columns))]
    for row in [columns, *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    totals = plan["header"]["totals"]
    print(
        f"\n{totals['chunks']} chunk(s), {totals['episodes']} episode(s), "
        f"{totals['frames']} frames, {gb(totals['est_in'])} GB in, "
        f"~{gb(totals['est_out'])} GB out"
    )
    for stage in STAGES:
        print(
            f"  {stage:8s} ok: {done[stage]} chunk(s), {done[f'{stage}_episodes']} "
            f"episode(s), {done[f'{stage}_frames']} frames"
        )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {TOOL}", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="Holds the plan, ledger, shard files, logs and reports.",
    )
    common.add_argument(
        "--plan", type=Path, help="Plan file (default: <state-dir>/plan.json)."
    )
    chunk = argparse.ArgumentParser(add_help=False)
    chunk.add_argument(
        "--chunk", required=True, help="A planned chunk, e.g. chunk_0000."
    )
    manifest = argparse.ArgumentParser(add_help=False)
    manifest.add_argument(
        "--manifest",
        type=Path,
        help="Default: the planned manifest; must match its sha256.",
    )
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local copy of the release (<task>/<season>/lerobot_v2.1/...).",
    )
    work = argparse.ArgumentParser(add_help=False)
    work.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="Outputs go to <work-root>/out/<chunk>.",
    )
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument(
        "--allow-provisional-hashes",
        action="store_true",
        help="Run although the manifest's hashes are not collision-checked "
        "(trial runs; recorded in the ledger).",
    )
    per_chunk = [common, chunk, manifest]

    p = sub.add_parser("plan", parents=[common], help="Pack kept episodes into chunks.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--sizes",
        type=Path,
        required=True,
        help="CSV 'path,size' of every release file at --revision.",
    )
    p.add_argument(
        "--budget-gb", type=float, required=True, help="Inputs + outputs per chunk."
    )
    p.add_argument(
        "--out-bytes-per-frame",
        type=float,
        required=True,
        help="Estimated Zarr bytes per frame (verify reports the measured value).",
    )
    p.add_argument("--mp4-bytes-per-episode", type=float, default=2e6)
    p.add_argument(
        "--streams",
        nargs="+",
        choices=STREAMS,
        default=list(STREAMS),
        help="Video streams to fetch (head_right adds --include-head-right).",
    )
    p.add_argument("--only-ids", type=Path, help="Plan only these recording ids.")
    p.add_argument("--tasks", default="*", help="Glob over task folders.")
    p.add_argument("--max-episodes", type=int, help="Plan the first N episodes only.")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--force", action="store_true", help="Replace an existing plan.")

    p = sub.add_parser(
        "fetch",
        parents=[*per_chunk, data, work, gate],
        help="Download a chunk's missing input files.",
    )
    p.add_argument("--repo-id", help="Default: the plan's.")
    p.add_argument("--revision", help="Default: the plan's.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--floor-gb",
        type=float,
        default=300.0,
        help="Free space that must remain after the chunk's downloads and "
        "outputs (default 300).",
    )
    p.add_argument("--retries", type=int, default=5, help="Per file, on HTTP 429/5xx.")
    p.add_argument("--backoff-seconds", type=float, default=5.0)

    p = sub.add_parser(
        "convert",
        parents=[*per_chunk, data, work, gate],
        help="Convert a chunk with CPU-pinned converter processes.",
    )
    p.add_argument(
        "--workers", type=int, help="Default: available CPUs // --cpus-per-worker."
    )
    p.add_argument("--cpus-per-worker", type=int, default=4)
    p.add_argument(
        "--shard",
        type=parse_shard,
        default=(0, 1),
        help="i/n: convert every n-th id of the chunk from the i-th, so n jobs "
        "can split it.",
    )
    p.add_argument("--python", default=sys.executable, help="Interpreter for workers.")
    p.add_argument(
        "--converter-cmd",
        help="Replace 'PYTHON -m sharpa_to_zarr' (one shell-quoted string; tests).",
    )
    p.add_argument("--break-lock", action="store_true", help="Override a live lock.")
    p.add_argument(
        "--max-load-ratio",
        type=float,
        default=1.0,
        help="Refuse to start when load1 / CPU count exceeds this (default 1.0).",
    )
    p.add_argument("--ignore-load", action="store_true")
    p.add_argument(
        "--monitor-seconds",
        type=float,
        default=300.0,
        help="Interval of the ledger's load/progress samples.",
    )

    sub.add_parser(
        "verify", parents=[*per_chunk, work], help="Check a converted chunk."
    )

    p = sub.add_parser(
        "upload",
        parents=[*per_chunk, work],
        help="Write (or with --execute run) the s5cmd upload of a verified chunk.",
    )
    p.add_argument("--bucket", default=BUCKET, help=f"Must be {BUCKET}.")
    p.add_argument(
        "--prefix",
        default=f"{S3_ROOT}{LAB}",
        help=f"Must be {S3_ROOT}{LAB}: the lab's folder.",
    )
    p.add_argument("--numworkers", type=int, default=64)
    p.add_argument("--execute", action="store_true", help="Upload (needs credentials).")
    p.add_argument(
        "--licence-approved",
        action="store_true",
        help="The leads approved CC-BY-NC-4.0 for this bucket.",
    )

    p = sub.add_parser(
        "clean",
        parents=[*per_chunk, data, work],
        help="Delete a chunk's fetched inputs (and uploaded outputs).",
    )
    p.add_argument(
        "--discard-outputs",
        action="store_true",
        help="Also delete outputs without a matching upload receipt.",
    )
    p.add_argument("--break-lock", action="store_true")

    p = sub.add_parser("status", parents=[common], help="Show each chunk's stages.")
    p.add_argument("--json", action="store_true")
    return parser


COMMANDS = {
    "plan": cmd_plan,
    "fetch": cmd_fetch,
    "convert": cmd_convert,
    "verify": cmd_verify,
    "upload": cmd_upload,
    "clean": cmd_clean,
    "status": cmd_status,
}


def main(argv=None) -> int:
    """Run one subcommand.

    Returns:
        ``0`` on success, ``1`` if the stage failed or refused, ``3`` if the
        disk guard refused a fetch.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        return COMMANDS[args.command](args)
    except IngestError as exc:
        if exc.exit_code == 1:
            raise SystemExit(f"sharpadex_ingest: {exc}") from exc
        print(f"sharpadex_ingest: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except (OSError, ValueError) as exc:
        raise SystemExit(f"sharpadex_ingest: {exc!r}") from exc


if __name__ == "__main__":
    sys.exit(main())
