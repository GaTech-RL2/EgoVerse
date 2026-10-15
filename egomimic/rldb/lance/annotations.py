"""Review of annotation change files before `commit.apply_annotations` commits them.

A change file is a Lance or parquet table with an `op` column ("add" or "delete") and the
ANNOTATIONS fields. An add row may leave `annotation_id` null, since it is computed from
the row's content; a delete row needs only `annotation_id`.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq

from egomimic.rldb.lance.commit import _exists, annotations_uri
from egomimic.rldb.lance.schema import ANNOTATION_TYPES, ANNOTATIONS, annotation_id

CONTENT = ("recording_id", "annotation_vendor", "annotation_type", "t_start_ns", "t_end_ns", "text")  # fmt: skip


@dataclass
class Review:
    adds: pa.Table
    deletes: list[str]
    errors: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def read_changes(path: str | Path) -> pa.Table:
    path = str(path).rstrip("/")
    return (
        lance.dataset(path).to_table()
        if path.endswith(".lance")
        else pq.read_table(path)
    )


def committed(root: Path) -> pa.Table:
    uri = annotations_uri(root)
    if not _exists(uri):
        return ANNOTATIONS.empty_table()
    return lance.dataset(str(uri)).to_table()


def _task_bounds(root: Path) -> dict[str, list[tuple[int, int]]]:
    cols = ["recording_id", "t_start_ns", "t_end_ns"]
    out = defaultdict(list)
    for t in (
        lance.dataset(str(Path(root) / "tasks.lance"))
        .to_table(columns=cols)
        .to_pylist()
    ):
        out[t["recording_id"]].append((t["t_start_ns"], t["t_end_ns"]))
    return out


def _add_errors(r: dict, tasks: dict) -> list[str]:
    errs = [f"{k} is missing" for k in CONTENT if r.get(k) in (None, "")]
    if errs:
        return errs
    if r["annotation_type"] not in ANNOTATION_TYPES:
        errs.append(
            f"annotation_type {r['annotation_type']!r} not in {ANNOTATION_TYPES}"
        )
    t0, t1 = r["t_start_ns"], r["t_end_ns"]
    if t0 >= t1:
        errs.append(f"t_start_ns {t0} is not before t_end_ns {t1}")
    if r["recording_id"] not in tasks:
        errs.append(f"recording {r['recording_id']} has no tasks")
    elif not any(s <= t0 and t1 <= e for s, e in tasks[r["recording_id"]]):
        errs.append(f"[{t0}, {t1}) is not inside any task of its recording")
    q = r.get("quality_score")
    if q is not None and not 0 <= q <= 5:
        errs.append(f"quality_score {q} is outside 0-5")
    return errs


def review(root: Path, changes: pa.Table) -> Review:
    """Validate `changes` against the committed tables; nothing is written."""
    tasks = _task_bounds(root)
    have = set(committed(root).column("annotation_id").to_pylist())
    adds, deletes, errors, unchanged, seen = [], [], [], [], set()
    for i, r in enumerate(changes.to_pylist()):
        op, given = r.get("op"), r.get("annotation_id")
        if op == "delete":
            if not given:
                errors.append(f"row {i}: delete without annotation_id")
            elif given not in have:
                errors.append(f"row {i}: delete of {given}, which is not committed")
            elif given in seen:
                errors.append(f"row {i}: {given} appears twice")
            else:
                deletes.append(given)
            seen.add(given)
            continue
        if op != "add":
            errors.append(f"row {i}: op {op!r} is not add or delete")
            continue
        errs = _add_errors(r, tasks)
        if errs:
            errors += [f"row {i}: {e}" for e in errs]
            continue
        aid = annotation_id(*(r[k] for k in CONTENT))
        if given and given != aid:
            errors.append(
                f"row {i}: annotation_id {given} does not match its content ({aid})"
            )
        elif aid in seen:
            errors.append(f"row {i}: {aid} appears twice")
        elif aid in have:
            unchanged.append(aid)
        else:
            adds.append(
                {**{k: r.get(k) for k in ANNOTATIONS.names}, "annotation_id": aid}
            )
        seen.add(aid)
    return Review(
        pa.Table.from_pylist(adds, schema=ANNOTATIONS), deletes, errors, unchanged
    )


def diff(root: Path, rv: Review) -> str:
    """Per recording: rows added, deleted and already committed, with their texts."""
    old = {r["annotation_id"]: r for r in committed(root).to_pylist()}
    per = defaultdict(lambda: {"+": [], "-": [], "=": 0})
    for r in rv.adds.to_pylist():
        per[r["recording_id"]]["+"].append(r)
    for aid in rv.deletes:
        per[old[aid]["recording_id"]]["-"].append(old[aid])
    for aid in rv.unchanged:
        per[old[aid]["recording_id"]]["="] += 1
    lines = []
    for rid in sorted(per):
        p = per[rid]
        lines.append(f"{rid}: +{len(p['+'])} -{len(p['-'])} ={p['=']}")
        for sign in "-+":
            for r in sorted(p[sign], key=lambda r: r["t_start_ns"]):
                lines.append(
                    f"  {sign} [{r['t_start_ns']}, {r['t_end_ns']}) "
                    f"{r['annotation_vendor']}/{r['annotation_type']}: {r['text']}"
                )
    return "\n".join(lines)
