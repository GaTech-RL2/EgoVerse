"""Committer operations on the Lance tables (https://lance.org/format/table/).

Each operation is one Lance commit per table it touches. Lance has no multi-table
transaction, so writes go steps -> aliases -> tasks -> recordings: a recording is
published by its `recordings` row, and readers ignore steps rows nobody references.
Videos live outside Lance under `videos/<recording_id>/<content digest>/` and are never
overwritten, so older table versions keep pointing at the files they were written with.
"""

import datetime
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance import LanceOperation
from lance.fragment import write_fragments

from egomimic.rldb.lance.schema import ALIASES, ANNOTATIONS, RECORDINGS, TASKS

CATALOG = {
    "aliases.lance": ALIASES,
    "tasks.lance": TASKS,
    "recordings.lance": RECORDINGS,
}


@dataclass
class Delivery:
    """One validated recording: catalog rows without `video_path`/`content_digest`, its steps and local MP4s."""

    recording: dict
    task: dict
    alias: dict
    steps: pa.Table
    videos: dict[str, Path]

    @property
    def recording_id(self) -> str:
        return self.recording["recording_id"]


def steps_uri(root: Path, lab: str) -> Path:
    return Path(root) / "steps" / f"{lab}.lance"


def annotations_uri(root: Path) -> Path:
    return Path(root) / "annotations.lance"


def table_uris(root: Path) -> list[Path]:
    """Every table, in write order: each lab's steps, then the catalog."""
    return [
        *sorted((Path(root) / "steps").glob("*.lance")),
        *(Path(root) / n for n in CATALOG),
        annotations_uri(root),
    ]


def content_digest(steps: pa.Table, videos: dict[str, Path]) -> str:
    h = hashlib.sha256()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, steps.schema) as w:
        w.write_table(steps.combine_chunks())
    h.update(sink.getvalue())
    for cam in sorted(videos):
        h.update(cam.encode() + b"\0")
        with open(videos[cam], "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def _exists(uri: Path) -> bool:
    return (Path(uri) / "_versions").exists()


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _append(uri: Path, table: pa.Table) -> None:
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, uri, mode="append" if _exists(uri) else "create")


def _replace(uri: Path, predicate: str | None, table: pa.Table) -> None:
    """Delete rows matching `predicate` and append `table` in one commit.

    merge_insert would also be one commit, but it writes matched rows before inserted
    ones, so a recording's steps would no longer be in timestamp order.
    """
    ds = lance.dataset(uri)
    removed, updated = [], []
    for frag in ds.get_fragments() if predicate else []:
        if frag.count_rows(predicate) == 0:
            continue
        meta = frag.delete(predicate)
        (removed.append(frag.fragment_id) if meta is None else updated.append(meta))
    new = write_fragments(table, uri, schema=ds.schema) if table.num_rows else []
    op = LanceOperation.Update(
        removed_fragment_ids=removed, updated_fragments=updated, new_fragments=new
    )
    lance.LanceDataset.commit(uri, op, read_version=ds.version)


def _stage(root: Path, d: Delivery) -> dict[str, pa.Table]:
    """Copy the videos into place and build each table's rows for one delivery."""
    digest = content_digest(d.steps, d.videos)
    paths = {}
    for cam, src in d.videos.items():
        rel = f"videos/{d.recording_id}/{digest[:16]}/{cam}.mp4"
        dst = Path(root) / rel
        if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        paths[cam] = rel
    rec = {**d.recording, "video_path": sorted(paths.items()), "content_digest": digest}
    return {
        "steps": d.steps,
        "aliases.lance": pa.Table.from_pylist([d.alias], schema=ALIASES),
        "tasks.lance": pa.Table.from_pylist([d.task], schema=TASKS),
        "recordings.lance": pa.Table.from_pylist([rec], schema=RECORDINGS),
    }


def committed_ids(root: Path) -> set[str]:
    uri = Path(root) / "recordings.lance"
    if not _exists(uri):
        return set()
    return set(
        lance.dataset(uri).to_table(columns=["recording_id"]).column(0).to_pylist()
    )


def append(root: Path, deliveries: list[Delivery], lab: str = "aria") -> list[str]:
    """Commit new recordings; ones already in `recordings` are skipped, so re-running is a no-op."""
    have = committed_ids(root)
    todo = [d for d in deliveries if d.recording_id not in have]
    if not todo:
        return []
    staged = [_stage(root, d) for d in todo]
    _append(steps_uri(root, lab), pa.concat_tables([s["steps"] for s in staged]))
    for name in CATALOG:
        _append(Path(root) / name, pa.concat_tables([s[name] for s in staged]))
    _ensure_annotations(root)
    return [d.recording_id for d in todo]


def _ensure_annotations(root: Path) -> None:
    # Exists from the first commit on, so a release tag always has an annotations version to pin.
    if not _exists(annotations_uri(root)):
        _append(annotations_uri(root), ANNOTATIONS.empty_table())


def apply_annotations(root: Path, adds: pa.Table, deletes: list[str]) -> dict[str, int]:
    """Add rows and delete rows by `annotation_id`, in one commit. Adds whose id is already
    committed are skipped, so resubmitting a file is a no-op."""
    _ensure_annotations(root)
    uri = annotations_uri(root)
    have = lance.dataset(uri).to_table(columns=["annotation_id"]).column(0)
    adds = adds.select(ANNOTATIONS.names).cast(ANNOTATIONS)
    adds = adds.filter(
        pc.invert(pc.is_in(adds["annotation_id"], have.combine_chunks()))
    )
    if not adds.num_rows and not deletes:
        return {"added": 0, "deleted": 0}
    pred = (
        f"annotation_id IN ({', '.join(map(_sql_str, deletes))})" if deletes else None
    )
    _replace(uri, pred, adds)
    return {"added": adds.num_rows, "deleted": len(deletes)}


def drop_task_annotations(root: Path) -> None:
    """One-off for tables written while annotations were nested in tasks."""
    ds = lance.dataset(Path(root) / "tasks.lance")
    if "annotations" in ds.schema.names:
        ds.drop_columns(["annotations"])


def replace(root: Path, d: Delivery, lab: str = "aria") -> None:
    """Swap a committed recording for a reprocessed delivery; its steps rows move to the end of the table."""
    if d.recording_id not in committed_ids(root):
        raise KeyError(f"recording {d.recording_id} is not committed")
    staged = _stage(root, d)
    pred = f"recording_id = {_sql_str(d.recording_id)}"
    _replace(steps_uri(root, lab), pred, staged["steps"])
    for name in CATALOG:
        _replace(Path(root) / name, pred, staged[name])


def edit(root: Path, table: str, rows: pa.Table, on: str) -> None:
    """Update columns of existing rows keyed by `on` (label and metadata fixes); every key must exist."""
    if table not in CATALOG and table != "annotations.lance":
        # merge_insert rewrites matched rows at the end of the table, which would split a recording's steps.
        raise ValueError(f"edit only applies to {list(CATALOG)}; use replace for steps")
    ds = lance.dataset(Path(root) / table)
    stats = ds.merge_insert(on).when_matched_update_all().execute(rows)
    if stats["num_updated_rows"] != rows.num_rows:
        raise KeyError(
            f"{table}: {rows.num_rows - stats['num_updated_rows']} of {rows.num_rows} keys not found"
        )


def add_columns(root: Path, table: str, columns: dict[str, str]) -> None:
    """New columns from SQL expressions over existing ones; written as new data files, nothing is rewritten."""
    lance.dataset(Path(root) / table).add_columns(columns)


def tag(root: Path, name: str) -> dict[str, int]:
    """Tag the current version of every table with `name`; returns {table: version}."""
    _ensure_annotations(root)
    out = {}
    for uri in table_uris(root):
        ds = lance.dataset(uri)
        if name in ds.tags.list():
            raise ValueError(f"tag {name} already exists on {uri}")
        ds.tags.create(name, ds.version)
        out[str(Path(uri).relative_to(root))] = ds.version
    return out


def index(root: Path, lab: str = "aria") -> None:
    lance.dataset(steps_uri(root, lab)).create_scalar_index("recording_id", "BTREE")


def compact(root: Path) -> None:
    for uri in filter(_exists, table_uris(root)):
        ds = lance.dataset(uri)
        ds.optimize.compact_files()
        if lance.dataset(uri).describe_indices():
            lance.dataset(uri).optimize.optimize_indices()


def cleanup(
    root: Path,
    older_than: datetime.timedelta = datetime.timedelta(days=14),
) -> None:
    """Drop untagged versions older than `older_than`, then videos no remaining version references.

    `older_than` must exceed the longest training run that resolved an untagged version.
    """
    for uri in filter(_exists, table_uris(root)):
        lance.dataset(uri).cleanup_old_versions(
            older_than=older_than, error_if_tagged_old_versions=False
        )
    gc_videos(root)


def gc_videos(root: Path) -> list[Path]:
    root = Path(root)
    uri = root / "recordings.lance"
    keep = set()
    for v in lance.dataset(uri).versions():
        tbl = lance.dataset(uri, version=v["version"]).to_table(columns=["video_path"])
        for m in tbl.column(0).to_pylist():
            keep.update(root / p for _, p in m)
    removed = [p for p in (root / "videos").rglob("*.mp4") if p not in keep]
    for p in removed:
        p.unlink()
    for d in sorted((root / "videos").rglob("*"), key=lambda p: -len(p.parts)):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return removed
