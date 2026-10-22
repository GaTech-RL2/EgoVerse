"""Convert a delivery now, commit exactly that result later, after a staff approval.

    prepare()        validate + convert into a staged folder; no Lance writes
    commit_staged()  commit the staged result, refusing anything that moved since

A staged folder holds everything the commit needs, pinned by stage.json:

    <staged>/stage.json      the pin: delivery fingerprint, converter revision and
                             settings, a sha256 of every staged file (diff included),
                             the recording's state hash in Lance before (base) and after
                             (staged) the commit, and the conversion report
    <staged>/steps.parquet   the steps rows
    <staged>/videos/*.mp4    one per camera
    <staged>/rows.json       the recording, tasks, alias and label rows
    <staged>/diff.json       what committing would change in Lance (for the reviewer)

A recording's state is everything Lance holds for it, table by table and in the form
Lance stores it: its steps rows (count and timestamps), alias, task and annotation rows,
and the recording row (whose content digest covers steps and videos). commit_staged
needs the approval's stage_digest. It applies the stage while Lance is in the base
state; a Lance already in the staged state is a retried commit (no-op); a mix where
every table is in either its base or its staged state is a commit of ours that was cut
off between tables (Lance commits each table on its own), which it finishes; anything
else is stale. It also refuses a changed staged file and a newer upload of the delivery.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

STAGE = "stage.json"
ROWS = "rows.json"
STEPS = "steps.parquet"
VIDEOS = "videos"
DIFF = "diff.json"


class StageCorrupt(Exception):
    """A staged file does not match its pinned checksum."""


class StaleStage(Exception):
    """The staged result no longer applies; stage the delivery again."""


@dataclass
class Prepared:
    """What preparing a delivery came to (what the watcher writes into its status)."""

    status: str  # staged | rejected | failed (| unsupported, from formats.py)
    recording_id: str = ""
    problems: list[str] = field(default_factory=list)  # why it was rejected
    error: str = ""  # why a validated delivery could not be converted
    staged: Path | None = None
    report: dict = field(default_factory=dict)
    diff: dict = field(default_factory=dict)


# ---------------------------------------------------------------- pins


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git() -> dict:
    """The converter's revision: the checkout this code runs from, else the REVISION file
    deploy.sh writes next to this module (a copy has no .git), else "unknown"."""
    here = Path(__file__).resolve().parent
    try:
        rev = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {"revision": rev, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        pass
    deployed = here / "REVISION"
    if deployed.is_file():
        return {"revision": deployed.read_text().strip(), "dirty": False}
    return {"revision": "unknown", "dirty": None}


def _canon(schema, rows: list[dict], drop=()) -> list[dict]:
    """Rows as Lance stores them under `schema` (float32 rounding, timestamp types)."""
    names = [n for n in schema.names if n not in drop]
    table = pa.Table.from_pylist(
        [{n: r.get(n) for n in schema.names} for r in rows], schema=schema
    )
    return [{n: r[n] for n in names} for r in table.to_pylist()]


def _rows(root: Path, table: str, rid: str) -> list[dict]:
    import lance

    from egomimic.rldb.lance import commit

    uri = Path(root) / table
    if not commit._exists(uri):
        return []
    return lance.dataset(uri).to_table(filter=f"recording_id = '{rid}'").to_pylist()


def _steps(timestamps) -> dict:
    """A recording's steps rows as their count and timestamps: enough to tell base rows,
    staged rows, and a cut-off commit's duplicates apart (the values are covered by the
    recording row's content digest, which is written last)."""
    ts = np.asarray(timestamps, dtype="<i8")
    return {
        "rows": int(len(ts)),
        "timestamps": hashlib.sha256(ts.tobytes()).hexdigest(),
    }


def committed_state(root: Path, rid: str, lab: str) -> dict:
    """Everything Lance holds for one recording, as comparable rows. video_path is left
    out: it is derived from content_digest and the camera names it already covers."""
    import lance

    from egomimic.rldb.lance import commit
    from egomimic.rldb.lance.schema import ALIASES, ANNOTATIONS, RECORDINGS, TASKS

    uri = commit.steps_uri(root, lab)
    ts = []
    if commit._exists(uri):
        t = lance.dataset(uri).to_table(
            columns=["timestamp_ns"], filter=f"recording_id = '{rid}'"
        )
        ts = t.column(0).to_numpy()
    return {
        "steps": _steps(ts),
        "recording": _canon(
            RECORDINGS, _rows(root, "recordings.lance", rid), ["video_path"]
        ),
        "alias": _canon(ALIASES, _rows(root, "aliases.lance", rid)),
        "tasks": sorted(
            _canon(TASKS, _rows(root, "tasks.lance", rid)),
            key=lambda t: t["task_instance_id"],
        ),
        "annotations": sorted(
            _canon(ANNOTATIONS, _rows(root, "annotations.lance", rid)),
            key=lambda a: a["annotation_id"],
        ),
    }


def proposed_state(converted, content_digest: str, base: dict) -> dict:
    """The recording's state once `converted` is committed on top of `base`: its rows
    replaced, its vendors' labels exactly the delivered ones, other labels kept."""
    from egomimic.rldb.lance.schema import ALIASES, ANNOTATIONS, RECORDINGS, TASKS

    d = converted.delivery
    new = {r["annotation_id"]: r for r in converted.labels}
    vendors = {converted.vendor, *(r["annotation_vendor"] for r in converted.labels)}
    kept = {
        a["annotation_id"]: a
        for a in base["annotations"]
        if a["annotation_id"] in new or a["annotation_vendor"] not in vendors
    }
    added = [
        r for i, r in new.items() if i not in kept
    ]  # existing ids keep their grades
    rec = {**d.recording, "content_digest": content_digest}
    return {
        "steps": _steps(d.steps.column("timestamp_ns").to_numpy()),
        "recording": _canon(RECORDINGS, [rec], ["video_path"]),
        "alias": _canon(ALIASES, [d.alias]),
        "tasks": sorted(_canon(TASKS, d.tasks), key=lambda t: t["task_instance_id"]),
        "annotations": sorted(
            list(kept.values()) + _canon(ANNOTATIONS, added),
            key=lambda a: a["annotation_id"],
        ),
    }


def state_hash(state: dict) -> str:
    blob = json.dumps(state, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def table_hashes(state: dict) -> dict[str, str]:
    """One hash per table of a recording's state."""
    return {k: state_hash(v) for k, v in state.items()}


def stage_digest(staged: Path) -> str:
    """The pin's own hash: an approval records it, so it applies to exactly this stage."""
    return _sha256(Path(staged) / STAGE)


# ---------------------------------------------------------------- saving / loading


def _jsonable(rec: dict) -> dict:
    out = dict(rec)
    out["captured_at"] = rec["captured_at"].isoformat()
    for k in ("intrinsics", "extrinsics"):
        out[k] = [[name, list(v)] for name, v in rec[k]]
    return out


def _recording(rec: dict) -> dict:
    out = dict(rec)
    out["captured_at"] = datetime.datetime.fromisoformat(rec["captured_at"])
    for k in ("intrinsics", "extrinsics"):
        out[k] = [(name, v) for name, v in rec[k]]
    return out


def _save(converted, staged: Path) -> None:
    d = converted.delivery
    (staged / VIDEOS).mkdir(parents=True)
    pq.write_table(d.steps, staged / STEPS, compression="zstd")
    for name, src in d.videos.items():
        shutil.copyfile(src, staged / VIDEOS / f"{name}.mp4")
    rows = {
        "recording": _jsonable(d.recording),
        "tasks": d.tasks,
        "alias": d.alias,
        "labels": converted.labels,
        "vendor": converted.vendor,
        "stats": converted.stats,
    }
    (staged / ROWS).write_text(json.dumps(rows, indent=1, sort_keys=True))


def load(staged: Path):
    """The staged result as the converter returned it, after checking every file
    against the pin (StageCorrupt otherwise)."""
    from egomimic.rldb.lance import commit
    from egomimic.scripts.mcap_process.mcap_to_lance import Converted

    staged = Path(staged)
    pin = read_pin(staged)
    for rel, digest in pin["files"].items():
        p = staged / rel
        if not p.is_file() or _sha256(p) != digest:
            raise StageCorrupt(f"{p} does not match the staged checksum")
    extra = {
        p.relative_to(staged).as_posix()
        for p in staged.rglob("*")
        if p.is_file() and p.name != STAGE
    } - set(pin["files"])
    if extra:
        raise StageCorrupt(f"{staged} holds unstaged files {sorted(extra)}")
    rows = json.loads((staged / ROWS).read_text())
    videos = {p.stem: p for p in sorted((staged / VIDEOS).glob("*.mp4"))}
    delivery = commit.Delivery(
        _recording(rows["recording"]),
        rows["tasks"],
        rows["alias"],
        pq.read_table(staged / STEPS),
        videos,
    )
    return Converted(delivery, rows["labels"], rows["vendor"], rows["stats"])


def read_pin(staged: Path) -> dict:
    return json.loads((Path(staged) / STAGE).read_text())


# ---------------------------------------------------------------- diff


def diff(root: Path, converted) -> dict:
    """What committing `converted` would change in Lance, for the reviewer."""
    import lance

    from egomimic.rldb.lance import commit
    from egomimic.scripts.mcap_process.mcap_to_lance import (
        committed_labels,
        stale_labels,
    )

    root = Path(root)
    d = converted.delivery
    rid = d.recording_id
    lab = d.recording["lab"]
    old = None
    if commit._exists(root / "recordings.lance"):
        rows = (
            lance.dataset(root / "recordings.lance")
            .to_table(filter=f"recording_id = '{rid}'")
            .to_pylist()
        )
        old = rows[0] if rows else None

    old_tasks = {}
    if old and commit._exists(root / "tasks.lance"):
        for t in (
            lance.dataset(root / "tasks.lance")
            .to_table(filter=f"recording_id = '{rid}'")
            .to_pylist()
        ):
            old_tasks[t["task_instance_id"]] = t
    new_tasks = {t["task_instance_id"]: t for t in d.tasks}

    def changed(a: dict, b: dict) -> bool:
        keys = ("t_start_ns", "t_end_ns", "task_name", "task_description", "objects")
        return any(a.get(k) != b.get(k) for k in keys)

    old_valid = None
    steps_uri = commit.steps_uri(root, lab)
    if old and commit._exists(steps_uri):
        cols = ["head_pose_valid", "left_valid", "right_valid"]
        t = lance.dataset(steps_uri).to_table(
            columns=cols, filter=f"recording_id = '{rid}'"
        )
        if t.num_rows:
            old_valid = {
                k.removesuffix("_pose_valid").removesuffix("_valid"): float(
                    pc.mean(t.column(k).cast(pa.float64())).as_py()
                )
                for k in cols
            }

    have = {r["annotation_id"] for r in committed_labels(root, rid)}
    vendors = {converted.vendor, *(r["annotation_vendor"] for r in converted.labels)}
    return {
        "recording_id": rid,
        "action": "replace" if old else "new",
        "frames": {"new": d.recording["num_frames"], "old": old and old["num_frames"]},
        "videos": {
            "new": sorted(d.videos),
            "old": sorted(k for k, _ in old["video_path"]) if old else [],
        },
        "tasks": {
            "added": sorted(
                new_tasks[k]["task_name"] for k in set(new_tasks) - set(old_tasks)
            ),
            "removed": sorted(
                old_tasks[k]["task_name"] for k in set(old_tasks) - set(new_tasks)
            ),
            "changed": sorted(
                new_tasks[k]["task_name"]
                for k in set(new_tasks) & set(old_tasks)
                if changed(new_tasks[k], old_tasks[k])
            ),
        },
        "labels": {
            "added": sorted(
                r["text"] for r in converted.labels if r["annotation_id"] not in have
            ),
            "removed": sorted(
                r["text"] for r in stale_labels(root, rid, converted.labels, vendors)
            ),
        },
        "valid_frac": {"new": converted.stats["valid_frac"], "old": old_valid},
    }


# ---------------------------------------------------------------- prepare / commit


def mano_model_digest(model_dir: str | None = None) -> str:
    """sha256 over MANO_LEFT.pkl and MANO_RIGHT.pkl in model_dir (default: where the
    converter looks), so a stage records which model fitted its hands."""
    from egomimic.scripts.aria_process.aria_utils import _default_mano_model_dir

    folder = Path(model_dir or _default_mano_model_dir())
    h = hashlib.sha256()
    for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
        h.update(name.encode() + b"\0" + (folder / name).read_bytes())
    return h.hexdigest()


def prepare(
    delivery: Path,
    staged: Path,
    lance_root: Path,
    *,
    fingerprint: str,
    mano: bool = True,
    mano_iters: int = 2400,
    mano_model_dir: str | None = None,
    ffmpeg: str = "ffmpeg",
    codec: str = "libsvtav1",
) -> Prepared:
    """Validate and convert a downloaded delivery into `staged` (replacing whatever was
    staged there before, only once the new result is complete). Nothing is written to
    Lance. `fingerprint` is the delivery's (store.Listed.fingerprint)."""
    from egomimic.rldb.lance import commit
    from egomimic.rldb.mcap.validate import validate_episode
    from egomimic.scripts.mcap_process.mcap_to_lance import Rejected, convert

    delivery, staged, lance_root = Path(delivery), Path(staged), Path(lance_root)
    try:
        v = validate_episode(delivery)
    except Exception as e:
        return Prepared("failed", error=f"{type(e).__name__}: {e}")
    if v.problems:
        return Prepared(
            "rejected", v.recording_id, problems=[str(p) for p in v.problems]
        )

    settings = {"mano": mano, "mano_iters": mano_iters, "codec": codec}
    pinned = dict(settings)
    if mano:  # which model fitted the hands, by content (the folder differs per host)
        pinned["mano_model"] = mano_model_digest(mano_model_dir)
    staged.parent.mkdir(parents=True, exist_ok=True)
    tmp = staged.with_name(f".{staged.name}.{uuid.uuid4().hex}")
    work = staged.with_name(f".{staged.name}.work.{uuid.uuid4().hex}")
    try:
        try:
            c = convert(
                delivery, work, ffmpeg=ffmpeg, mano_model_dir=mano_model_dir, **settings
            )
        except Rejected as e:  # validated, but Lance would refuse it (e.g. labels)
            return Prepared("rejected", v.recording_id, problems=e.problems)
        except Exception as e:  # unsupported embodiment / lens, broken video, ...
            return Prepared("failed", v.recording_id, error=f"{type(e).__name__}: {e}")
        d = c.delivery
        _save(c, tmp)
        # From the staged files, as commit_staged will load them (a parquet round trip
        # can change how the steps table serializes).
        digest = commit.content_digest(
            pq.read_table(tmp / STEPS),
            {name: tmp / VIDEOS / f"{name}.mp4" for name in d.videos},
        )
        base = committed_state(lance_root, d.recording_id, d.recording["lab"])
        change = diff(lance_root, c)
        (tmp / DIFF).write_text(json.dumps(change, indent=1, sort_keys=True))
        files = {
            p.relative_to(tmp).as_posix(): _sha256(p)
            for p in sorted(tmp.rglob("*"))
            if p.is_file()
        }
        report = {
            "num_frames": d.recording["num_frames"],
            "filler_frames": c.stats["filler_frames"],
            "tasks": len(d.tasks),
            "labels": len(c.labels),
            "videos": sorted(d.videos),
            "valid_frac": c.stats["valid_frac"],
        }
        pin = {
            "format": "mcap",
            "recording_id": d.recording_id,
            "lab": d.recording["lab"],
            "fingerprint": fingerprint,
            "converter": _git(),
            "settings": pinned,
            "files": files,
            "content_digest": digest,
            "base_state": table_hashes(base),
            "staged_state": table_hashes(proposed_state(c, digest, base)),
            "report": report,
            "staged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (tmp / STAGE).write_text(json.dumps(pin, indent=1, sort_keys=True))
        if staged.exists():
            shutil.rmtree(staged)
        tmp.rename(staged)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    return Prepared("staged", d.recording_id, staged=staged, report=report, diff=change)


def _finish(converted, lance_root: Path) -> tuple[int, int]:
    """Finish a commit of ours that stopped between tables: every table is in either its
    base or its staged state. Rewrite every table to the staged rows (dropping the
    orphans of the cut-off attempt), then the labels."""
    import lance

    from egomimic.rldb.lance import commit
    from egomimic.scripts.mcap_process.mcap_to_lance import _sync_labels

    d = converted.delivery
    rid, lab = d.recording_id, d.recording["lab"]
    if rid in commit.committed_ids(lance_root):
        commit.replace(lance_root, d, lab=lab)  # every table, rows for rid swapped
    else:  # not published yet: clear the partial rows, then publish in order
        pred = f"recording_id = '{rid}'"
        for uri in (
            commit.steps_uri(lance_root, lab),
            lance_root / "aliases.lance",
            lance_root / "tasks.lance",
        ):
            if commit._exists(uri):
                commit._replace(uri, pred, lance.dataset(uri).schema.empty_table())
        commit.append(lance_root, [d], lab=lab)
    vendors = {converted.vendor, *(r["annotation_vendor"] for r in converted.labels)}
    return _sync_labels(lance_root, rid, converted.labels, vendors)


def commit_staged(
    staged: Path,
    lance_root: Path,
    *,
    approved_stage: str,
    fingerprint: str | None = None,
) -> dict:
    """Commit exactly the staged result that was approved. `approved_stage` is the
    stage_digest the approval was given for (required: this is the approval boundary;
    mcap_to_lance.convert_and_commit is the unreviewed path). `fingerprint`, if given,
    is the delivery's current one: a newer upload makes this stage stale. Returns status
    (committed | replaced | resumed | unchanged), labels added / deleted, and every
    table's Lance version."""
    import lance

    from egomimic.rldb.lance import commit
    from egomimic.scripts.mcap_process.mcap_to_lance import commit_converted

    staged, lance_root = Path(staged), Path(lance_root)
    if approved_stage != stage_digest(staged):
        raise StaleStage(f"{staged}: the approval was given for a different stage")
    pin = read_pin(staged)
    if fingerprint is not None and fingerprint != pin["fingerprint"]:
        raise StaleStage(f"{staged}: the delivery was re-uploaded since it was staged")
    c = load(staged)  # checks every staged file against the pin
    rid, lab = pin["recording_id"], pin["lab"]
    base, target = pin["base_state"], pin["staged_state"]
    current = table_hashes(committed_state(lance_root, rid, lab))
    added = deleted = 0
    if current == target:
        status = "unchanged"  # a retried commit: Lance already holds exactly this
    elif current == base:
        status, added, deleted = commit_converted(c, lance_root)
    elif all(current[k] in (base[k], target[k]) for k in target):
        added, deleted = _finish(c, lance_root)  # cut off between tables last time
        status = "resumed"
    else:
        raise StaleStage(f"{rid} changed in Lance since it was staged; stage it again")
    if status != "unchanged":
        if table_hashes(committed_state(lance_root, rid, lab)) != target:
            raise RuntimeError(
                f"{rid}: Lance does not hold the staged state after commit"
            )
    versions = {
        str(uri.relative_to(lance_root)): lance.dataset(uri).version
        for uri in commit.table_uris(lance_root, lab)
        if commit._exists(uri)
    }
    return {
        "status": status,
        "recording_id": rid,
        "annotations_added": added,
        "annotations_deleted": deleted,
        "versions": versions,
    }
