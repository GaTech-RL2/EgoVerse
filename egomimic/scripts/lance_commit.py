"""The committer: the only writer of the Lance tables.

Converters stage deliveries (steps.parquet, MP4s, meta.json last) under
`<staging>/<source>/<vendor id>/` (commit.stage), or upload the MP4s straight to the Lance
root and stage the rest (commit.stage_direct); those videos are only checked by size, never
downloaded. Each run validates the complete ones, commits them in batches (each batch tagged
`<source>-<UTC time>` on every table and recorded in `changes`), and deletes their staged
files. A delivery that fails validation gets REJECTED.json (errors plus a digest of its
files) and is skipped until its files change.
Aria recording and task metadata come from SQL; EVA's and zarr's from the delivery's meta.json.
A delivery's meta.json "provenance" is kept in its batch's `changes` row, and its
"annotations" (zarr_to_lance's, mcap_to_lance's) are committed right after its batch as one
annotation change.

Vendor MCAP deliveries are staged per lab under `<staging>/mcap-<lab>/`, because a vendor id
is only unique within its lab (`--source mcap-<lab>`).

`--source annotations` commits annotation change files that `lance_annotations stage` put under
`<staging>/annotations/<id>/`, one commit and tag each, oldest first; each is reviewed again
against the tables as they are now and rejected like a bad delivery if it no longer applies.
See egomimic/rldb/lance/commit.py for the table operations.

    python -m egomimic.scripts.lance_commit --staging s3://rldb/lance_staging \\
        --source eva --lance s3://rldb/lance --work-dir /coc/flash7/scratch/committer
"""

import argparse
import hashlib
import json
import re
import shutil
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import text

from egomimic.rldb.lance import commit, lock, mp4_ranges
from egomimic.rldb.lance.annotations import committed, read_changes, review
from egomimic.rldb.lance.dataset import read_range
from egomimic.rldb.lance.metadata import objects_list, text_or_none
from egomimic.rldb.lance.schema import ANNOTATIONS as ANNOTATIONS_SCHEMA
from egomimic.rldb.lance.schema import task_instance_id
from egomimic.rldb.zarr.required_metadata import UNWAIVABLE_FIELDS
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_timestamp_ms,
    timestamp_ms_to_episode_hash,
)

REJECTED = "REJECTED.json"
ANNOTATIONS = "annotations"
BATCH_SIZE = 64
# In a recording's `changes` provenance: files_digest of the staged delivery it came from.
STAGED_FILES = "staged_files"


def sql_rows(hashes: list[str]) -> pd.DataFrame:
    q = text(
        """select episode_hash, operator, lab, embodiment, rig_name, scene, task,
                  task_description, objects, license
           from app.episodes where episode_hash = any(:h)"""
    )
    with create_default_engine().connect() as c:
        return pd.read_sql(q, c, params={"h": hashes}).set_index("episode_hash")


def check_delivery(d: Path, meta: dict, steps: pa.Table) -> list[str]:
    errs = []
    n = meta["num_frames"]
    if steps.num_rows != n:
        errs.append(f"steps rows {steps.num_rows} != num_frames {n}")
    ts = steps.column("timestamp_ns").to_numpy()
    if not (np.diff(ts) > 0).all():
        errs.append("timestamps not strictly increasing")
    placed = meta.get("lance_videos")
    for cam in meta.get("videos", ("front_1", "front_1_480")):
        if placed:
            if cam not in placed["videos"]:
                errs.append(f"{cam}.mp4 is not in lance_videos")
                continue
            frames = placed["videos"][cam]["num_frames"]
        else:
            frames = mp4_ranges.read_index(
                lambda a, b, p=d / f"{cam}.mp4": read_range(str(p), a, b)
            ).num_frames
        if frames != n:
            errs.append(f"{cam}.mp4 has {frames} frames, steps has {n}")
    return errs


def _lower(value) -> str | None:
    """A SQL lab or embodiment as the planner and drivers write it: one spelling per
    steps table."""
    value = text_or_none(value)
    return value.lower() if value else value


def load_deliveries(
    dirs: list[Path],
) -> tuple[list[commit.Delivery], dict[str, list[str]]]:
    """Validated deliveries for `dirs` (each holding a meta.json) and {vendor id: errors} for the rest."""
    metas = {d: json.loads((d / "meta.json").read_text()) for d in dirs}
    hashes = [
        timestamp_ms_to_episode_hash(m["vendor_recording_id"]) for m in metas.values()
    ]
    sql = sql_rows(hashes)

    deliveries, rejected = [], {}
    for d, m in sorted(metas.items(), key=lambda kv: kv[1]["recording_id"]):
        steps = pq.read_table(d / "steps.parquet")
        errs = check_delivery(d, m, steps)
        h = timestamp_ms_to_episode_hash(m["vendor_recording_id"])
        if h not in sql.index:
            errs.append("no SQL row")
        elif m["embodiment"].lower() != _lower(sql.loc[h].embodiment):
            # Hand validity in steps was computed for the converter's embodiment.
            errs.append(
                f"converted as {m['embodiment']}, SQL says {sql.loc[h].embodiment}"
            )
        if errs:
            rejected[m["vendor_recording_id"]] = errs
            continue
        s = sql.loc[h]
        rid = m["recording_id"]
        recording = dict(
            recording_id=rid,
            vendor_recording_id=m["vendor_recording_id"],
            lab=_lower(s.lab),
            embodiment=_lower(s.embodiment),
            end_effector=m.get("end_effector"),
            rig_name=text_or_none(s.rig_name),
            operator=text_or_none(s.operator),
            scene=text_or_none(s.scene),
            captured_at=pd.Timestamp(
                episode_hash_to_timestamp_ms(h), unit="ms", tz="UTC"
            ),
            fps=m["fps"],
            num_frames=m["num_frames"],
            intrinsics=list(m["intrinsics"].items()),
            extrinsics=list(m["extrinsics"].items()),
            license=text_or_none(s.license),
        )
        task = dict(
            task_instance_id=task_instance_id(
                rid, text_or_none(s.task), m["t_start_ns"]
            ),
            recording_id=rid,
            t_start_ns=m["t_start_ns"],
            t_end_ns=m["t_end_ns"],
            task_name=text_or_none(s.task),
            task_description=text_or_none(s.task_description),
            objects=objects_list(s.objects),
        )
        alias = dict(
            episode_hash=h,
            recording_id=rid,
            t_start_ns=m["t_start_ns"],
            t_end_ns=m["t_end_ns"],
        )
        delivery = _delivery(
            d, m, recording, [task], alias, steps, ("front_1", "front_1_480")
        )
        try:
            commit.check_required(delivery)
        except ValueError as e:
            rejected[m["vendor_recording_id"]] = [str(e)]
            continue
        deliveries.append(delivery)
    return deliveries, rejected


def _delivery(d: Path, meta: dict, recording, tasks, alias, steps, cams):
    placed = meta.get("lance_videos")
    videos = {} if placed else {cam: d / f"{cam}.mp4" for cam in cams}
    return commit.Delivery(
        recording,
        tasks,
        alias,
        steps,
        videos,
        provenance=meta.get("provenance"),
        placed=placed,
    )


def captured_at(meta: dict) -> pd.Timestamp | None:
    """None for an episode with no capture time (abc's UUID-named zarrs)."""
    ms = meta.get("captured_at_ms", meta["vendor_recording_id"])
    return None if ms is None else pd.Timestamp(int(ms), unit="ms", tz="UTC")


def load_meta_deliveries(
    dirs: list[Path],
) -> tuple[list[commit.Delivery], dict[str, list[str]]]:
    """Deliveries whose meta.json carries their recording and task metadata (EVA, zarr,
    MCAP). A `tasks` list (MCAP) replaces the one task over the whole recording."""
    deliveries, rejected = [], {}
    for d in sorted(dirs):
        m = json.loads((d / "meta.json").read_text())
        steps = pq.read_table(d / "steps.parquet")
        errs = check_delivery(d, m, steps)
        errs += [f"no {k}" for k in UNWAIVABLE_FIELDS if not m.get(k)]
        if "tasks" in m and not (
            m["tasks"] and all(t.get("task_name") for t in m["tasks"])
        ):
            errs.append("a task has no task_name")
        if errs:
            rejected[m["vendor_recording_id"]] = errs
            continue
        rid = m["recording_id"]
        recording = dict(
            recording_id=rid,
            vendor_recording_id=m["vendor_recording_id"],
            lab=m["lab"],
            embodiment=m["embodiment"],
            end_effector=m.get("end_effector"),
            rig_name=m.get("rig_name", "eva"),
            operator=m["operator"],
            scene=m["scene"],
            captured_at=captured_at(m),
            fps=m["fps"],
            num_frames=m["num_frames"],
            intrinsics=list(m["intrinsics"].items()),
            extrinsics=list(m["extrinsics"].items()),
            license=m.get("license"),
        )
        tasks = m.get("tasks") or [
            dict(
                task_instance_id=task_instance_id(rid, m["task_name"], m["t_start_ns"]),
                recording_id=rid,
                t_start_ns=m["t_start_ns"],
                t_end_ns=m["t_end_ns"],
                task_name=m["task_name"],
                task_description=m["task_description"],
                objects=m["objects"],
            )
        ]
        alias = dict(
            episode_hash=m["episode_hash"],
            recording_id=rid,
            t_start_ns=m["t_start_ns"],
            t_end_ns=m["t_end_ns"],
        )
        try:
            commit.steps_key(recording, steps)
        except ValueError as e:
            rejected[m["vendor_recording_id"]] = [str(e)]
            continue
        deliveries.append(_delivery(d, m, recording, tasks, alias, steps, m["videos"]))
    return deliveries, rejected


LOADERS = {
    "aria": load_deliveries,
    "eva": load_meta_deliveries,
    "zarr": load_meta_deliveries,
}
MCAP_PREFIX = "mcap-"


def mcap_source(lab: str) -> str:
    """The staging folder of a lab's MCAP deliveries. `lab` comes from the vendor's file, so
    it must be one plain name: it cannot reach another source's folder or leave staging."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", lab):
        raise ValueError(f"lab {lab!r}: only lowercase letters, digits and _")
    return MCAP_PREFIX + lab


def _loader(source: str):
    return load_meta_deliveries if source.startswith(MCAP_PREFIX) else LOADERS[source]


def _source_arg(source: str) -> str:
    if source == ANNOTATIONS or source in LOADERS:
        return source
    if source.startswith(MCAP_PREFIX):
        return mcap_source(source[len(MCAP_PREFIX) :])
    raise argparse.ArgumentTypeError(
        f"{source!r}: one of {sorted([*LOADERS, ANNOTATIONS])} or mcap-<lab>"
    )


# A delivery whose files cannot be read or parsed. Anything else (SQL, network) aborts the
# run, so an outage never turns into REJECTED.json notes that stick until files change.
UNREADABLE = (OSError, ValueError, KeyError, TypeError, IndexError, struct.error)


def _unreadable(e: Exception) -> bool:
    return isinstance(e, UNREADABLE) and not isinstance(
        e, (ConnectionError, TimeoutError)
    )


# Rejections that can clear without the delivery changing; pending() retries them.
RETRYABLE = {"no SQL row"}


def _load(source: str, dirs: list[Path]):
    """LOADERS[source] over `dirs`, rejections keyed by staged directory name; a delivery
    the loader cannot even read is rejected alone."""
    rejected = {}
    for d in dirs:
        try:
            vid = str(json.loads((d / "meta.json").read_text())["vendor_recording_id"])
        except UNREADABLE:
            continue  # the loader rejects it below
        if vid != d.name:
            rejected[d.name] = [
                f"meta.json vendor_recording_id {vid} != staged directory {d.name}"
            ]
    dirs = [d for d in dirs if d.name not in rejected]
    try:
        deliveries, bad = _loader(source)(dirs)
        return deliveries, {**rejected, **bad}
    except Exception as e:
        if not _unreadable(e):
            raise
        deliveries = []
        for d in dirs:
            try:
                ok, bad = _loader(source)([d])
            except Exception as e:
                if not _unreadable(e):
                    raise
                ok, bad = [], {d.name: [f"unreadable: {type(e).__name__}: {e}"]}
            deliveries += ok
            rejected.update(bad)
        return deliveries, rejected


def files_digest(files: dict[str, tuple[int, str]]) -> str:
    """Changes whenever a staged file is replaced (S3 ETag, or local size and mtime)."""
    h = hashlib.sha256()
    for name, (size, version) in sorted(files.items()):
        if name != REJECTED:
            h.update(f"{name}\0{size}\0{version}\0".encode())
    return h.hexdigest()


def pending(staging: str, source: str, replace: bool = False) -> list[tuple[str, str]]:
    """(vendor id, files digest) of complete staged deliveries not rejected as they are;
    with `replace`, ones rejected only for being committed already come back. Ones
    rejected only for RETRYABLE reasons come back every run."""
    out = []
    for vid in commit.list_dirs(commit.join(staging, source)):
        prefix = commit.join(staging, source, vid)
        files = commit.list_files(prefix)
        if "meta.json" not in files:
            continue
        digest = files_digest(files)
        if REJECTED in files:
            with tempfile.TemporaryDirectory() as t:
                commit.get_file(commit.join(prefix, REJECTED), Path(t) / REJECTED)
                note = json.loads((Path(t) / REJECTED).read_text())
            sticky = not set(note["errors"]) <= RETRYABLE
            if (
                sticky
                and note["digest"] == digest
                and not (replace and note["committed"])
            ):
                continue
        out.append((vid, digest))
    return out


def _reject(staging: str, source: str, vid: str, errs, digest: str, committed=False):
    note = {
        "errors": errs,
        "digest": digest,
        "committed": committed,
        "rejected_at": time.time(),
    }
    commit.put_bytes(
        json.dumps(note, indent=1).encode(), commit.join(staging, source, vid, REJECTED)
    )


def _unstage(staging: str, source: str, vid: str) -> None:
    prefix = commit.join(staging, source, vid)
    commit.remove([commit.join(prefix, name) for name in commit.list_files(prefix)])


def commit_staged_annotations(
    staging: str, lance_root: str, work_dir: Path, lock_wait: float | None = None
) -> dict:
    """Commit every pending staged annotation change, oldest first, one commit each.
    Holds the writer lock on `lance_root` throughout (`lock_wait`: see writer_lock)."""
    with lock.writer_lock(lance_root, wait=lock_wait):
        report = {"batches": [], "rejected": {}}
        todo = []
        for sid, digest in pending(staging, ANNOTATIONS):
            prefix = commit.join(staging, ANNOTATIONS, sid)
            meta = json.loads(commit.get_bytes(commit.join(prefix, "meta.json")))
            todo.append((meta["staged_at"], sid, digest, meta))
        for _, sid, digest, meta in sorted(todo):
            local = work_dir / ANNOTATIONS / sid
            commit.get_file(
                commit.join(staging, ANNOTATIONS, sid, "changes.parquet"),
                local / "changes.parquet",
            )
            rv = review(lance_root, read_changes(local / "changes.parquet"))
            if rv.errors:
                _reject(staging, ANNOTATIONS, sid, rv.errors, digest)
                report["rejected"][sid] = rv.errors
            else:
                out = commit.commit_annotations(
                    lance_root,
                    rv.adds,
                    rv.deletes,
                    operator=meta["approver"],
                    message=f"{meta['source']}, approved by {meta['approver']}",
                    provenance={
                        k: meta[k] for k in ("approver", "source", "staged_at")
                    },
                )
                if out:
                    report["batches"].append(out)
                _unstage(staging, ANNOTATIONS, sid)
            shutil.rmtree(local, ignore_errors=True)
        return report


def _annotation_rows(d: commit.Delivery, meta: dict) -> list[dict]:
    """Change rows for the annotations a delivery carries (zarr_to_lance's), by its lab."""
    return [
        dict(
            op="add",
            recording_id=d.recording_id,
            t_start_ns=a["t_start_ns"],
            t_end_ns=a["t_end_ns"],
            annotation_type=a.get("annotation_type") or "language",
            text=a["text"],
            annotation_vendor=a.get("annotation_vendor") or meta["lab"],
        )
        for a in meta.get("annotations") or ()
    ]


def _commit_annotations(
    lance_root,
    rows: list[dict],
    done: set,
    replaced: dict[str, set],
    batch_tag: str,
    operator,
) -> dict:
    """Commit the annotations of a batch's committed recordings (`done`) as one annotation
    change; rows that fail review are reported and skipped, never the recordings.
    `replaced` maps each replaced recording whose delivery carries annotations to its
    annotation vendors: its committed rows from those vendors that the new delivery no
    longer has are deleted, so a re-conversion does not leave stale spans behind."""
    rows = [r for r in rows if r["recording_id"] in done]
    rv = review(lance_root, pa.Table.from_pylist(rows)) if rows else None
    adds = rv.adds if rv else ANNOTATIONS_SCHEMA.empty_table()
    keep = {*adds.column("annotation_id").to_pylist(), *(rv.unchanged if rv else ())}
    deletes = []
    if replaced:
        deletes = [
            r["annotation_id"]
            for r in committed(lance_root).to_pylist()
            if r["annotation_vendor"] in replaced.get(r["recording_id"], ())
            and r["annotation_id"] not in keep
        ]
    out = None
    if adds.num_rows or deletes:
        out = commit.commit_annotations(
            lance_root,
            adds,
            deletes,
            operator=operator,
            provenance={"source": "delivery", "batch": batch_tag},
        )
    return {
        "tag": out and out["tag"],
        "rows": adds.num_rows,
        "deleted": len(deletes),
        "errors": rv.errors if rv else [],
    }


def commit_staged(
    staging: str,
    source: str,
    lance_root: str,
    work_dir: Path,
    batch_size: int = BATCH_SIZE,
    replace: bool = False,
    operator: str | None = None,
    lock_wait: float | None = None,
) -> dict:
    """Commit every pending staged delivery of `source`, `batch_size` at a time; returns
    {"batches": [commit_batch results], "rejected": {vendor id: errors}}.
    Holds the writer lock on `lance_root` throughout, so the committed-or-not reads
    and the writes they decide see no other writer (`lock_wait`: see writer_lock)."""
    with lock.writer_lock(lance_root, wait=lock_wait):
        report = {"batches": [], "rejected": {}}
        todo = pending(staging, source, replace)
        have = commit.committed_ids(lance_root)
        for b in range(0, len(todo), batch_size):
            batch = dict(todo[b : b + batch_size])
            local = {}
            for vid in batch:
                src = commit.join(staging, source, vid)
                local[vid] = work_dir / source / vid
                for name in commit.list_files(src):
                    if name != REJECTED:
                        commit.get_file(commit.join(src, name), local[vid] / name)
            deliveries, rejected = _load(source, list(local.values()))
            for d in [d for d in deliveries if d.placed]:
                if errs := commit.check_placed(lance_root, d.recording_id, d.placed):
                    deliveries.remove(d)
                    rejected[d.recording["vendor_recording_id"]] = errs
            skipped, resumed, unrecorded = [], [], []
            if not replace:
                again = [d for d in deliveries if d.recording_id in have]
                ids = [d.recording_id for d in again]
                digests = commit.committed_digests(lance_root, ids)
                recorded = commit.recorded_provenance(lance_root, ids)
                logged = commit.recorded_ids(lance_root, ids)
                for d in again:
                    deliveries.remove(d)
                    vid = d.recording["vendor_recording_id"]
                    # This very delivery, committed by a run that stopped before finishing it:
                    # its annotations may be missing and it is still staged. The same staged
                    # files, or the same content if the run stopped before recording them. Same
                    # content alone is not enough: a correction may change only metadata or labels.
                    prov = recorded.get(d.recording_id)
                    same = (
                        prov.get(STAGED_FILES) == batch[vid]
                        if prov is not None
                        else digests.get(d.recording_id) == commit.delivery_digest(d)
                    )
                    if same:
                        resumed.append(d)
                        if d.recording_id not in logged:
                            unrecorded.append(d)
                        continue
                    skipped.append(d)
                    rejected[d.recording["vendor_recording_id"]] = [
                        "already committed; rerun with --replace to replace it"
                    ]
            spans, vendors = [], {}
            for d in [*deliveries, *unrecorded]:
                d.provenance = {
                    **(d.provenance or {}),
                    STAGED_FILES: batch[d.recording["vendor_recording_id"]],
                }
            for d in [*deliveries, *resumed]:
                meta = json.loads(
                    (
                        local[d.recording["vendor_recording_id"]] / "meta.json"
                    ).read_text()
                )
                rows = _annotation_rows(d, meta)
                spans += rows
                if "annotations" in meta:
                    vendors[d.recording_id] = {
                        meta["lab"],
                        *(r["annotation_vendor"] for r in rows),
                    }
            # Appended by a run that stopped before recording them: record and tag them now.
            recovered = commit.record_recovered(
                lance_root, unrecorded, source, operator
            )
            if recovered:
                report["batches"].append(recovered)
            result = commit.commit_batch(
                lance_root,
                deliveries,
                source,
                replace_committed=replace,
                operator=operator,
            )
            result_tag = (result or recovered or {}).get("tag")
            done, replaced = {d.recording_id for d in resumed}, {}
            if result:
                report["batches"].append(result)
                have |= set(result["added"])
                done |= {*result["added"], *result["replaced"]}
                replaced = {r: vendors[r] for r in result["replaced"] if r in vendors}
            if done:
                ann = _commit_annotations(
                    lance_root, spans, done, replaced, result_tag, operator
                )
                report.setdefault("annotations", []).append(ann)
            if resumed:
                report.setdefault("resumed", []).extend(d.recording_id for d in resumed)
            skipped_ids = {d.recording["vendor_recording_id"] for d in skipped}
            for vid, errs in rejected.items():
                _reject(staging, source, vid, errs, batch[vid], vid in skipped_ids)
            report["rejected"].update(rejected)
            for d in [*deliveries, *resumed]:
                _unstage(staging, source, d.recording["vendor_recording_id"])
            for path in local.values():
                shutil.rmtree(path, ignore_errors=True)
        return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--staging",
        required=True,
        help="staging root (s3:// or local); deliveries are under <staging>/<source>/<vendor id>/",
    )
    ap.add_argument(
        "--source",
        type=_source_arg,
        required=True,
        help=f"one of {sorted([*LOADERS, ANNOTATIONS])} or mcap-<lab>",
    )
    ap.add_argument("--lance", required=True, help="Lance root (s3:// or local)")
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="local scratch for downloaded deliveries",
    )
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument(
        "--replace",
        action="store_true",
        help="replace committed recordings with restaged deliveries",
    )
    ap.add_argument("--operator", help="recorded in changes; defaults to $USER")
    args = ap.parse_args()
    if args.source == ANNOTATIONS:
        if args.replace or args.operator or args.batch_size != BATCH_SIZE:
            ap.error(
                "--replace, --operator and --batch-size do not apply to --source "
                "annotations: each change is its own commit, by its approver"
            )
        report = commit_staged_annotations(args.staging, args.lance, args.work_dir)
    else:
        report = commit_staged(
            args.staging,
            args.source,
            args.lance,
            args.work_dir,
            args.batch_size,
            args.replace,
            args.operator,
        )
    for b in report["batches"]:
        print(f"{b['tag']}: added {len(b['added'])}, replaced {len(b['replaced'])}")
    for vid, errs in report["rejected"].items():
        print(f"rejected {vid}: {'; '.join(errs)}")


if __name__ == "__main__":
    main()
