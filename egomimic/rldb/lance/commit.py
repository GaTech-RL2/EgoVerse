"""Committer operations on the Lance tables (https://lance.org/format/table/).

Each operation is one Lance commit per table it touches. Lance has no multi-table
transaction, so writes go steps -> aliases -> tasks -> recordings: a recording is
published by its `recordings` row, and readers ignore steps rows nobody references.
Videos live outside Lance under `videos/<recording_id>/<content digest>/` and are never
overwritten, so older table versions keep pointing at the files they were written with.

`root` is a local directory or an s3:// prefix; credentials come from the R2_*
environment (dataset.storage_options).
"""

import datetime
import getpass
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance import LanceOperation
from lance.fragment import write_fragments

from egomimic.rldb.embodiment.embodiment import END_EFFECTOR
from egomimic.rldb.lance.dataset import _is_remote, _s3_client, storage_options
from egomimic.rldb.lance.schema import (
    ALIASES,
    ANNOTATIONS,
    CHANGES,
    RECORDINGS,
    STEPS_KEY,
    TASKS,
    steps_path,
    steps_paths,
)

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

    @property
    def steps_key(self) -> tuple[str, str, str]:
        return steps_key(self.recording)


def steps_key(recording: dict) -> tuple[str, str, str]:
    """(lab, embodiment, end_effector) of a recordings row; rejects missing or unknown values."""
    for c in STEPS_KEY:
        if not recording.get(c):
            raise ValueError(f"recording {recording.get('recording_id')}: no {c}")
    END_EFFECTOR(recording["end_effector"])
    return tuple(recording[c] for c in STEPS_KEY)


def join(root, *parts: str) -> str:
    return "/".join([str(root).rstrip("/"), *parts])


def _split(uri: str) -> tuple[str, str]:
    bucket, key = uri[len("s3://") :].split("/", 1)
    return bucket, key


def list_files(prefix) -> dict[str, tuple[int, str]]:
    """{path relative to `prefix`: (size, etag or mtime)} of every file under it."""
    prefix = str(prefix).rstrip("/")
    if not _is_remote(prefix):
        root = Path(prefix)
        return {
            str(p.relative_to(root)): (p.stat().st_size, str(p.stat().st_mtime_ns))
            for p in (root.rglob("*") if root.exists() else ())
            if p.is_file()
        }
    bucket, key = _split(prefix)
    out = {}
    for page in (
        _s3_client()
        .get_paginator("list_objects_v2")
        .paginate(Bucket=bucket, Prefix=key + "/")
    ):
        for o in page.get("Contents", []):
            out[o["Key"][len(key) + 1 :]] = (o["Size"], o["ETag"])
    return out


def list_dirs(prefix) -> list[str]:
    """Names of the directories (common prefixes) directly under `prefix`."""
    prefix = str(prefix).rstrip("/")
    if not _is_remote(prefix):
        p = Path(prefix)
        return sorted(d.name for d in p.iterdir() if d.is_dir()) if p.exists() else []
    bucket, key = _split(prefix)
    out = []
    for page in (
        _s3_client()
        .get_paginator("list_objects_v2")
        .paginate(Bucket=bucket, Prefix=key + "/", Delimiter="/")
    ):
        out += [
            c["Prefix"][len(key) + 1 :].rstrip("/")
            for c in page.get("CommonPrefixes", [])
        ]
    return sorted(out)


def _nonempty(prefix: str) -> bool:
    if not _is_remote(prefix):
        p = Path(prefix)
        return p.is_dir() and any(p.iterdir())
    bucket, key = _split(prefix.rstrip("/"))
    resp = _s3_client().list_objects_v2(Bucket=bucket, Prefix=key + "/", MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def put_file(src: Path, dst: str) -> None:
    if _is_remote(dst):
        _s3_client().upload_file(str(src), *_split(dst))
    else:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def put_bytes(data: bytes, dst: str) -> None:
    if _is_remote(dst):
        bucket, key = _split(dst)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    else:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(data)


def get_file(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _is_remote(src):
        _s3_client().download_file(*_split(src), str(dst))
    else:
        shutil.copy2(src, dst)


def remove(uris: list[str]) -> None:
    remote = [u for u in uris if _is_remote(u)]
    for u in uris:
        if not _is_remote(u):
            Path(u).unlink(missing_ok=True)
    for i in range(0, len(remote), 1000):
        by_bucket: dict[str, list] = {}
        for u in remote[i : i + 1000]:
            bucket, key = _split(u)
            by_bucket.setdefault(bucket, []).append({"Key": key})
        for bucket, objs in by_bucket.items():
            _s3_client().delete_objects(Bucket=bucket, Delete={"Objects": objs})


def stage(delivery: Path, dst) -> None:
    """Upload a delivery directory to the staging prefix `dst`, meta.json last: the
    committer only picks up deliveries whose meta.json exists, so it never sees a
    half-uploaded one."""
    files = sorted(p for p in Path(delivery).iterdir() if p.is_file())
    for p in sorted(files, key=lambda p: p.name == "meta.json"):
        put_file(p, join(dst, p.name))


def _size(uri: str) -> int | None:
    if not _is_remote(uri):
        return Path(uri).stat().st_size if Path(uri).exists() else None
    from botocore.exceptions import ClientError

    bucket, key = _split(uri)
    try:
        return _s3_client().head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError:
        return None


def _open(uri: str, version=None) -> lance.LanceDataset:
    return lance.dataset(uri, version=version, storage_options=storage_options(uri))


def steps_uri(root, lab: str, embodiment: str, end_effector: str) -> str:
    return join(root, steps_path(lab, embodiment, end_effector))


def steps_uris(root) -> list[str]:
    """The steps tables `recordings` references."""
    uri = join(root, "recordings.lance")
    if not _exists(uri):
        return []
    tbl = _open(uri).to_table(columns=list(STEPS_KEY))
    return [join(root, p) for p in steps_paths(tbl)]


def annotations_uri(root) -> str:
    return join(root, "annotations.lance")


def changes_uri(root) -> str:
    return join(root, "changes.lance")


def table_uris(root) -> list[str]:
    """Every table, in write order: the steps tables, the catalog, changes, annotations."""
    return [
        *steps_uris(root),
        *(join(root, n) for n in CATALOG),
        changes_uri(root),
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


def _exists(uri: str) -> bool:
    return _nonempty(join(uri, "_versions"))


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _append(uri: str, table: pa.Table) -> None:
    lance.write_dataset(
        table,
        uri,
        mode="append" if _exists(uri) else "create",
        storage_options=storage_options(uri),
    )


def _replace(uri: str, predicate: str | None, table: pa.Table) -> None:
    """Delete rows matching `predicate` and append `table` in one commit.

    merge_insert would also be one commit, but it writes matched rows before inserted
    ones, so a recording's steps would no longer be in timestamp order.
    """
    ds = _open(uri)
    removed, updated = [], []
    for frag in ds.get_fragments() if predicate else []:
        if frag.count_rows(predicate) == 0:
            continue
        meta = frag.delete(predicate)
        (removed.append(frag.fragment_id) if meta is None else updated.append(meta))
    opts = storage_options(uri)
    new = (
        write_fragments(table, uri, schema=ds.schema, storage_options=opts)
        if table.num_rows
        else []
    )
    op = LanceOperation.Update(
        removed_fragment_ids=removed, updated_fragments=updated, new_fragments=new
    )
    lance.LanceDataset.commit(uri, op, read_version=ds.version, storage_options=opts)


def _stage(root, d: Delivery) -> dict[str, pa.Table]:
    """Copy the videos into place and build each table's rows for one delivery."""
    digest = content_digest(d.steps, d.videos)
    paths = {}
    for cam, src in d.videos.items():
        rel = f"videos/{d.recording_id}/{digest[:16]}/{cam}.mp4"
        dst = join(root, rel)
        if _size(dst) != Path(src).stat().st_size:
            put_file(src, dst)
        paths[cam] = rel
    rec = {**d.recording, "video_path": sorted(paths.items()), "content_digest": digest}
    return {
        "steps": d.steps,
        "aliases.lance": pa.Table.from_pylist([d.alias], schema=ALIASES),
        "tasks.lance": pa.Table.from_pylist([d.task], schema=TASKS),
        "recordings.lance": pa.Table.from_pylist([rec], schema=RECORDINGS),
    }


def committed_ids(root) -> set[str]:
    uri = join(root, "recordings.lance")
    if not _exists(uri):
        return set()
    return set(_open(uri).to_table(columns=["recording_id"]).column(0).to_pylist())


def append(root, deliveries: list[Delivery]) -> list[str]:
    """Commit new recordings; ones already in `recordings` are skipped, so re-running is a no-op."""
    have = committed_ids(root)
    todo = [d for d in deliveries if d.recording_id not in have]
    if not todo:
        return []
    keys = [d.steps_key for d in todo]
    staged = [_stage(root, d) for d in todo]
    for key in sorted(set(keys)):
        _append(
            steps_uri(root, *key),
            pa.concat_tables([s["steps"] for s, k in zip(staged, keys) if k == key]),
        )
    for name in CATALOG:
        _append(join(root, name), pa.concat_tables([s[name] for s in staged]))
    _ensure_tables(root)
    return [d.recording_id for d in todo]


def _ensure_tables(root) -> None:
    # Exist from the first commit on, so a tag always has a version of them to pin.
    for uri, schema in (
        (annotations_uri(root), ANNOTATIONS),
        (changes_uri(root), CHANGES),
    ):
        if not _exists(uri):
            _append(uri, schema.empty_table())


def apply_annotations(root, adds: pa.Table, deletes: list[str]) -> dict[str, int]:
    """Add rows and delete rows by `annotation_id`, in one commit. Adds whose id is already
    committed are skipped, so resubmitting a file is a no-op."""
    _ensure_tables(root)
    uri = annotations_uri(root)
    have = _open(uri).to_table(columns=["annotation_id"]).column(0)
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


def drop_task_annotations(root) -> None:
    """One-off for tables written while annotations were nested in tasks."""
    ds = _open(join(root, "tasks.lance"))
    if "annotations" in ds.schema.names:
        ds.drop_columns(["annotations"])


def replace(root, d: Delivery) -> None:
    """Swap a committed recording for a reprocessed delivery; its steps rows move to the end of the table."""
    pred = f"recording_id = {_sql_str(d.recording_id)}"
    uri = join(root, "recordings.lance")
    old = (
        _open(uri).to_table(columns=list(STEPS_KEY), filter=pred).to_pylist()
        if _exists(uri)
        else []
    )
    if not old:
        raise KeyError(f"recording {d.recording_id} is not committed")
    if tuple(old[0][c] for c in STEPS_KEY) != d.steps_key:
        raise ValueError(
            f"recording {d.recording_id}: {STEPS_KEY} changed from {old[0]}; its steps would move tables"
        )
    staged = _stage(root, d)
    _replace(steps_uri(root, *d.steps_key), pred, staged["steps"])
    for name in CATALOG:
        _replace(join(root, name), pred, staged[name])


def commit_batch(
    root,
    deliveries: list[Delivery],
    lab: str,
    replace_committed: bool = False,
    operator: str | None = None,
) -> dict | None:
    """Append new recordings (and replace committed ones when asked), record the batch
    in `changes` and tag every table `<lab>-<UTC time to the ms>`. None when nothing changed."""
    added = append(root, deliveries)
    replaced = []
    if replace_committed:
        for d in deliveries:
            if d.recording_id not in added:
                replace(root, d)
                replaced.append(d.recording_id)
    if not added and not replaced:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    name = f"{lab}-{now:%Y%m%dT%H%M%S.%f}"[:-3]
    row = dict(
        tag=name,
        lab=lab,
        committed_at=now,
        operator=operator or getpass.getuser(),
        added=added,
        replaced=replaced,
    )
    _append(changes_uri(root), pa.Table.from_pylist([row], schema=CHANGES))
    return {
        "tag": name,
        "added": added,
        "replaced": replaced,
        "versions": tag(root, name),
    }


def edit(root, table: str, rows: pa.Table, on: str) -> None:
    """Update columns of existing rows keyed by `on` (label and metadata fixes); every key must exist."""
    if table not in CATALOG and table != "annotations.lance":
        # merge_insert rewrites matched rows at the end of the table, which would split a recording's steps.
        raise ValueError(f"edit only applies to {list(CATALOG)}; use replace for steps")
    ds = _open(join(root, table))
    stats = ds.merge_insert(on).when_matched_update_all().execute(rows)
    if stats["num_updated_rows"] != rows.num_rows:
        raise KeyError(
            f"{table}: {rows.num_rows - stats['num_updated_rows']} of {rows.num_rows} keys not found"
        )


def add_columns(root, table: str, columns: dict[str, str]) -> None:
    """New columns from SQL expressions over existing ones; written as new data files, nothing is rewritten."""
    _open(join(root, table)).add_columns(columns)


def tag(root, name: str) -> dict[str, int]:
    """Tag the current version of every table with `name`; returns {table: version}."""
    _ensure_tables(root)
    out = {}
    for uri in table_uris(root):
        ds = _open(uri)
        if name in ds.tags.list():
            raise ValueError(f"tag {name} already exists on {uri}")
        ds.tags.create(name, ds.version)
        out[uri[len(str(root).rstrip("/")) + 1 :]] = ds.version
    return out


def index(root) -> None:
    for uri in steps_uris(root):
        _open(uri).create_scalar_index("recording_id", "BTREE")


def compact(root) -> None:
    for uri in filter(_exists, table_uris(root)):
        _open(uri).optimize.compact_files()
        if _open(uri).describe_indices():
            _open(uri).optimize.optimize_indices()


def cleanup(
    root,
    older_than: datetime.timedelta = datetime.timedelta(days=14),
) -> None:
    """Drop untagged versions older than `older_than`, then videos no remaining version references.

    `older_than` must exceed the longest training run that resolved an untagged version.
    """
    for uri in filter(_exists, table_uris(root)):
        _open(uri).cleanup_old_versions(
            older_than=older_than, error_if_tagged_old_versions=False
        )
    gc_videos(root)


def gc_videos(root) -> list[str]:
    """Delete MP4s under videos/ that no version of `recordings` references."""
    uri = join(root, "recordings.lance")
    keep = set()
    for v in _open(uri).versions():
        tbl = _open(uri, v["version"]).to_table(columns=["video_path"])
        for m in tbl.column(0).to_pylist():
            keep.update(p for _, p in m)
    removed = [
        join(root, "videos", p)
        for p in list_files(join(root, "videos"))
        if p.endswith(".mp4") and f"videos/{p}" not in keep
    ]
    remove(removed)
    if not _is_remote(str(root)):
        vids = Path(root) / "videos"
        for d in sorted(vids.rglob("*"), key=lambda p: -len(p.parts)):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    return removed
