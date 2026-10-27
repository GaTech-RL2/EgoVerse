"""Convert a delivery now; hand exactly that result to the committer after a staff approval.

    prepare()   validate + convert into a staged folder; nothing is written to Lance
    verify()    check a stored stage against its pin before handing it off

A staged folder is a staged delivery as egomimic.rldb.lance.commit.stage_delivery writes it
(steps, videos, catalog rows, labels, meta.json), plus:

    <staged>/stage.json   the pin: delivery fingerprint, converter revision and settings,
                          sha256 and size of every other file, and the conversion report
    <staged>/diff.json    what committing would change in Lance, for the reviewer

stage_digest (the pin's sha256) is what an approval records, so the approval applies to
exactly that result. On approval the watcher copies the staged delivery to the
committer's staging prefix (egomimic.scripts.lance_commit, the only Lance writer), which
commits it unless the recording is already committed (that needs --replace).
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

import pyarrow as pa
import pyarrow.compute as pc

STAGE = "stage.json"
DIFF = "diff.json"
META = "meta.json"
REVIEW_ONLY = (STAGE, DIFF)  # not handed to the committer


class StageCorrupt(Exception):
    """A stored stage does not match its pin."""


@dataclass
class Prepared:
    """What preparing a delivery came to (what the conversion job records)."""

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


def converter_revision() -> dict:
    """The converter's revision: the checkout this code runs from, else the REVISION file
    the image build writes next to this module (a copy has no .git), else "unknown"."""
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


def stage_digest(staged: Path) -> str:
    """The pin's own hash: an approval records it, so it applies to exactly this stage."""
    return _sha256(Path(staged) / STAGE)


def read_pin(staged: Path) -> dict:
    return json.loads((Path(staged) / STAGE).read_text())


def verify(pin: dict, sizes: dict[str, int]) -> None:
    """A stored stage's files (name -> size, from a listing) are exactly its pin's."""
    want = {rel: f["size"] for rel, f in pin["files"].items()}
    got = {rel: size for rel, size in sizes.items() if rel not in REVIEW_ONLY}
    if got != want:
        raise StageCorrupt(f"stored files {sorted(got)} do not match the pin")


# ---------------------------------------------------------------- diff


def _rows(root, table: str, rid: str, columns=None) -> list[dict]:
    from egomimic.rldb.lance import commit

    uri = commit.join(root, table)
    if not commit._exists(uri):
        return []
    return (
        commit._open(uri)
        .to_table(columns=columns, filter=f"recording_id = {commit._sql_str(rid)}")
        .to_pylist()
    )


def diff(root, converted) -> dict:
    """What committing `converted` would change in Lance, for the reviewer."""
    from egomimic.rldb.lance import commit

    d = converted.delivery
    rid = d.recording_id
    old = next(iter(_rows(root, "recordings.lance", rid)), None)
    old_tasks = (
        {t["task_instance_id"]: t for t in _rows(root, "tasks.lance", rid)}
        if old
        else {}
    )
    new_tasks = {t["task_instance_id"]: t for t in d.tasks}

    def changed(a: dict, b: dict) -> bool:
        keys = ("t_start_ns", "t_end_ns", "task_name", "task_description", "objects")
        return any(a.get(k) != b.get(k) for k in keys)

    old_valid = None
    steps_uri = commit.steps_uri(root, *d.steps_key)
    if old and commit._exists(steps_uri):
        cols = ["head_pose_valid", "left_valid", "right_valid"]
        t = commit._open(steps_uri).to_table(
            columns=cols, filter=f"recording_id = {commit._sql_str(rid)}"
        )
        if t.num_rows:
            old_valid = {
                k.removesuffix("_pose_valid").removesuffix("_valid"): float(
                    pc.mean(t.column(k).cast(pa.float64())).as_py()
                )
                for k in cols
            }

    committed = _rows(
        root,
        "annotations.lance",
        rid,
        ["annotation_id", "annotation_vendor", "text"],
    )
    have = {r["annotation_id"] for r in committed}
    new_ids = {r["annotation_id"] for r in converted.labels}
    vendors = {
        converted.label_source,
        *(r["annotation_vendor"] for r in converted.labels),
    }
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
                r["text"]
                for r in committed
                if r["annotation_vendor"] in vendors
                and r["annotation_id"] not in new_ids
            ),
        },
        "valid_frac": {"new": converted.stats["valid_frac"], "old": old_valid},
    }


# ---------------------------------------------------------------- prepare


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
    lance_root,
    *,
    fingerprint: str,
    vendor: str,
    mano: bool = True,
    mano_iters: int = 2400,
    mano_model_dir: str | None = None,
    ffmpeg: str = "ffmpeg",
    codec: str = "libsvtav1",
) -> Prepared:
    """Validate and convert a downloaded delivery into `staged` (replacing whatever was
    staged there before, only once the new result is complete). Lance is only read, for
    the reviewer's diff. `fingerprint` is the delivery's (store.Listed.fingerprint);
    `vendor` is who delivered it (its bucket), the namespace of its recording id."""
    from egomimic.rldb.lance import commit
    from egomimic.rldb.lance.schema import ANNOTATIONS
    from egomimic.rldb.mcap import spec
    from egomimic.rldb.mcap.validate import validate_episode
    from egomimic.scripts.mcap_process.mcap_to_lance import SOURCE, Rejected, convert

    delivery, staged = Path(delivery), Path(staged)
    try:
        # Metadata waived with missing_metadata goes to review (report["waived"]) instead
        # of being refused: staff decide case by case when approving.
        v = validate_episode(delivery, allow_missing=True)
    except Exception as e:
        return Prepared("failed", error=f"{type(e).__name__}: {e}")
    rid = (
        spec.recording_id(vendor, v.vendor_recording_id)
        if v.vendor_recording_id
        else ""
    )
    if v.problems:
        return Prepared("rejected", rid, problems=[str(p) for p in v.problems])

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
                delivery,
                work,
                vendor=vendor,
                ffmpeg=ffmpeg,
                mano_model_dir=mano_model_dir,
                **settings,
            )
        except Rejected as e:  # validated, but Lance would refuse it (e.g. labels)
            return Prepared("rejected", rid, problems=e.problems)
        except Exception as e:  # unsupported embodiment / lens, broken video, ...
            return Prepared("failed", rid, error=f"{type(e).__name__}: {e}")
        d = c.delivery
        converter = converter_revision()
        labels = pa.Table.from_pylist(
            [{k: r.get(k) for k in ANNOTATIONS.names} for r in c.labels],
            schema=ANNOTATIONS,
        )
        commit.stage_delivery(
            d,
            str(tmp),
            labels=labels,
            meta={
                "label_vendors": sorted(
                    {c.label_source, *(r["annotation_vendor"] for r in c.labels)}
                ),
                "provenance": {
                    "converter": converter,
                    "settings": pinned,
                    "fingerprint": fingerprint,
                },
            },
        )
        change = diff(lance_root, c)
        (tmp / DIFF).write_text(json.dumps(change, indent=1, sort_keys=True))
        files = {
            p.name: {"sha256": _sha256(p), "size": p.stat().st_size}
            for p in sorted(tmp.iterdir())
            if p.is_file() and p.name not in REVIEW_ONLY
        }
        report = {
            "num_frames": d.recording["num_frames"],
            "filler_frames": c.stats["filler_frames"],
            "tasks": len(d.tasks),
            "labels": len(c.labels),
            "videos": sorted(d.videos),
            "valid_frac": c.stats["valid_frac"],
        }
        if v.warnings:
            report["waived"] = v.warnings
        pin = {
            "format": "mcap",
            "recording_id": d.recording_id,
            "vendor_recording_id": d.recording["vendor_recording_id"],
            "lab": d.recording["lab"],
            "source": SOURCE,  # the committer's staging prefix
            "fingerprint": fingerprint,
            "converter": converter,
            "settings": pinned,
            "files": files,
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
