r"""One-off: copy pilot Lance roots written with `lab` = source format ("aria"/"eva")
and `site` = collecting lab into one new root with one `lab` column (the collecting lab),
an `end_effector` column, and steps tables keyed `steps/{lab}/{embodiment}/{end_effector}.lance`.

recording_id is unchanged (its hash input stays "<source>/<vendor id>"), so tasks, aliases,
annotations and changes of every `--src` are unioned; a recording_id or alias
episode_hash found twice fails the run. The legacy tasks.annotations column is dropped
(--dry-run counts the rows that carry one). Sources are never written.

Lance history does not carry over: every new table starts fresh and its migrated version
is tagged `--tag`; tags on the source tables are listed and dropped, and `changes.tag` keeps
naming them (they resolve only in the source root). videos/ and any other non-table
file are copied byte for byte; `video_path` is relative to the root, so it stays valid.

    python -m egomimic.scripts.lance_pilot.migrate_lab_end_effector \
        --src /coc/flash7/scratch/lance_pilot/lance \
        --src /coc/flash7/scratch/lance_pilot/lance_eva --dst <new root> --dry-run
"""

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc

from egomimic.rldb.embodiment.end_effector import END_EFFECTOR
from egomimic.rldb.lance.commit import (
    _exists,
    _is_remote,
    _nonempty,
    _open,
    _s3_client,
    _split,
    _sql_str,
    drop_task_annotations,
    get_file,
    join,
    list_dirs,
    list_files,
    put_file,
)
from egomimic.rldb.lance.dataset import storage_options
from egomimic.rldb.lance.schema import CHANGES, RECORDINGS, steps_path

# Old `lab` (the converter) -> end_effector.
END_EFFECTOR_OF_SOURCE = {
    "aria": END_EFFECTOR.HUMAN_HAND.value,
    "eva": END_EFFECTOR.PARALLEL_JAW_GRIPPER.value,
}

COPIED_TABLES = ("tasks.lance", "aliases.lance", "annotations.lance")


def migrate_recordings(old: pa.Table, lab_for_null_site: str | None) -> pa.Table:
    """Old recordings rows -> new ones, same row order."""
    if "site" not in old.column_names:
        raise ValueError("recordings has no `site` column: not an old-schema root")
    unknown = sorted(set(old["lab"].to_pylist()) - set(END_EFFECTOR_OF_SOURCE))
    if unknown:
        raise ValueError(
            f"recordings.lab {unknown}: expected one of {list(END_EFFECTOR_OF_SOURCE)}"
        )
    site = old["site"]
    missing = old.filter(pc.is_null(site))["recording_id"].to_pylist()
    if missing and lab_for_null_site is None:
        raise ValueError(
            f"{len(missing)} recordings have no site, and lab is required; pass "
            f"--lab-for-null-site to set one: {missing}"
        )
    if missing:
        site = pc.fill_null(site, lab_for_null_site)
    end_effector = pa.array(
        [END_EFFECTOR_OF_SOURCE[s] for s in old["lab"].to_pylist()], pa.string()
    )
    cols = {
        **{n: old[n] for n in old.column_names},
        "lab": site,
        "end_effector": end_effector,
    }
    extra = sorted(set(cols) - set(RECORDINGS.names) - {"site"})
    if extra:
        raise ValueError(
            f"recordings has columns this migration does not know: {extra}"
        )
    # content_digest is absent from roots written before it existed.
    n = old.num_rows
    return pa.table(
        {f.name: cols.get(f.name, pa.nulls(n, f.type)) for f in RECORDINGS}
    ).cast(RECORDINGS)


def migrate_changes(old: pa.Table, key_of: dict[str, str]) -> pa.Table:
    """Old changes rows -> CHANGES rows, one each: the old `lab` (the converter, or
    "annotations") becomes `source`, and `steps_tables` lists the steps tables its
    recordings live in ([] for annotation commits)."""
    out = []
    for row in old.to_pylist():
        row = {**row, "source": row.pop("lab"), "steps_tables": []}
        if row["source"] in END_EFFECTOR_OF_SOURCE:
            ids = (row["added"] or []) + (row["replaced"] or [])
            lost = sorted(set(ids) - set(key_of))
            if lost:
                raise ValueError(
                    f"changes {row['tag']}: recordings not committed: {lost}"
                )
            row["steps_tables"] = sorted({key_of[r] for r in ids})
        out.append(row)
    return pa.Table.from_pylist(out, schema=CHANGES)


def old_steps_uris(root) -> dict[str, str]:
    """{old lab: steps table uri}."""
    return {
        d[: -len(".lance")]: join(root, "steps", d)
        for d in list_dirs(join(root, "steps"))
        if d.endswith(".lance")
    }


def new_steps_uris(root) -> dict[str, str]:
    """{steps key: uri} of a new-layout root."""
    out = {}
    base = join(root, "steps")
    for lab in list_dirs(base):
        for emb in list_dirs(join(base, lab)):
            for t in list_dirs(join(base, lab, emb)):
                if t.endswith(".lance"):
                    key = steps_path(lab, emb, t[: -len(".lance")])
                    out[key] = join(root, key)
    return out


def _union(tables: list[pa.Table]) -> pa.Table:
    """Rows of every source; a column some sources lack is null in their rows."""
    return pa.concat_tables(tables, promote_options="default")


def _nonempty_count(col: pa.ChunkedArray) -> int:
    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
        return int(pc.sum(pc.greater(pc.list_value_length(col), 0)).as_py() or 0)
    if pa.types.is_string(col.type):
        return int(pc.sum(pc.greater(pc.utf8_length(col), 0)).as_py() or 0)
    return col.length() - col.null_count


def _duplicates(values) -> list:
    return sorted(v for v, n in Counter(values).items() if n > 1)


def plan(
    srcs: list[str], lab_for_null_site: str | None, drop_orphan_steps: bool
) -> dict:
    """Read the sources and work out every new table's rows, without writing."""
    old_recs = [_open(join(s, "recordings.lance")).to_table() for s in srcs]
    new_rec = _union([migrate_recordings(r, lab_for_null_site) for r in old_recs])
    clash = _duplicates(new_rec["recording_id"].to_pylist())
    if clash:
        raise ValueError(f"recording_id in more than one source: {clash}")
    key_of = {
        r["recording_id"]: steps_path(r["lab"], r["embodiment"], r["end_effector"])
        for r in new_rec.select(
            ["recording_id", "lab", "embodiment", "end_effector"]
        ).to_pylist()
    }
    steps_rows: Counter = Counter()
    by_key: dict[str, list[tuple[str, list[str]]]] = {}
    orphans: dict[str, int] = {}
    for src, old_rec in zip(srcs, old_recs):
        source = dict(
            zip(old_rec["recording_id"].to_pylist(), old_rec["lab"].to_pylist())
        )
        for old_lab, uri in old_steps_uris(src).items():
            rids = Counter(_open(uri).to_table(columns=["recording_id"])[0].to_pylist())
            wrong = sorted(r for r in rids if r in source and source[r] != old_lab)
            if wrong:
                raise ValueError(
                    f"{uri} has rows of recordings from another lab: {wrong}"
                )
            orphans.update({r: n for r, n in rids.items() if r not in source})
            groups: dict[str, list[str]] = {}
            for r in rids:
                if r in source:
                    groups.setdefault(key_of[r], []).append(r)
                    steps_rows[key_of[r]] += rids[r]
            for key, ids in groups.items():
                by_key.setdefault(key, []).append((uri, sorted(ids)))

    def union(name):
        parts = [
            _open(join(s, name)).to_table() for s in srcs if _exists(join(s, name))
        ]
        return _union(parts) if parts else None

    # A crashed append leaves catalog rows of its recordings too, not only steps rows.
    copied, orphan_rows = {}, {}
    committed = new_rec["recording_id"].combine_chunks()
    for n in COPIED_TABLES:
        t = union(n)
        if t is not None:
            keep = pc.is_in(t["recording_id"], value_set=committed)
            gone = t.filter(pc.invert(keep))["recording_id"].to_pylist()
            if gone:
                orphan_rows[n] = len(gone)
                t = t.filter(keep)
        copied[n] = t
    if (orphans or orphan_rows) and not drop_orphan_steps:
        raise ValueError(
            f"steps rows of {len(orphans)} recordings not in recordings.lance "
            f"({sum(orphans.values())} rows, plus {orphan_rows or 'no'} catalog rows); "
            f"pass --drop-orphan-steps: {sorted(orphans)}"
        )
    for key, parts in by_key.items():
        if len({_open(u).schema for u, _ in parts}) > 1:
            raise ValueError(f"{key}: source tables differ in schema")

    if copied["aliases.lance"] is not None:
        clash = _duplicates(copied["aliases.lance"]["episode_hash"].to_pylist())
        if clash:
            raise ValueError(f"alias episode_hash in more than one row: {clash}")
    tasks = copied["tasks.lance"]
    task_annotations = (
        _nonempty_count(tasks["annotations"])
        if tasks is not None and "annotations" in tasks.column_names
        else None
    )
    changes = union("changes.lance")
    if changes is not None:
        changes = migrate_changes(changes, key_of)
    tables = {"recordings.lance": new_rec.num_rows}
    tables.update({n: t.num_rows for n, t in copied.items() if t is not None})
    if changes is not None:
        tables["changes.lance"] = changes.num_rows
    tables.update({k: steps_rows[k] for k in sorted(steps_rows)})

    files: dict[str, str] = {}
    for src in srcs:
        for rel in list_files(src):
            if any(part.endswith(".lance") for part in rel.split("/")):
                continue
            if rel in files:
                raise ValueError(f"{rel} is in {files[rel]} and {src}")
            files[rel] = src
    known = {"recordings.lance", "changes.lance", *COPIED_TABLES}
    others: dict[str, str] = {}
    for src in srcs:
        for rel in tables_under(src):
            if rel in known or rel.startswith("steps/"):
                continue
            if rel in others:
                raise ValueError(f"{rel} is in {others[rel]} and {src}")
            others[rel] = src
            tables[rel] = _open(join(src, rel)).count_rows()
    return dict(
        srcs=srcs,
        recordings=new_rec,
        copied=copied,
        changes=changes,
        key_of=key_of,
        steps_sources=by_key,
        orphans=orphans,
        orphan_rows=orphan_rows,
        files=files,
        others=others,
        task_annotations=task_annotations,
        tables=tables,
    )


def _write(uri: str, data, schema: pa.Schema | None = None) -> lance.LanceDataset:
    return lance.write_dataset(
        data, uri, schema=schema, mode="create", storage_options=storage_options(uri)
    )


def _copy_file(src: str, dst: str) -> None:
    if _is_remote(src) and _is_remote(dst):
        _s3_client().copy(
            {"Bucket": _split(src)[0], "Key": _split(src)[1]}, *_split(dst)
        )
    elif not _is_remote(src) and not _is_remote(dst):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "f"
            get_file(src, local)
            put_file(local, dst)


def _copy_steps(src_parts: list[tuple[str, list[str]]], dst_uri: str) -> None:
    """One new steps table from slices of old ones, in source row order so each
    recording's rows stay contiguous and in timestamp order."""
    first = _open(src_parts[0][0])

    def batches():
        for uri, ids in src_parts:
            pred = f"recording_id IN ({', '.join(map(_sql_str, ids))})"
            yield from _open(uri).scanner(filter=pred, scan_in_order=True).to_batches()

    reader = pa.RecordBatchReader.from_batches(first.schema, batches())
    new = _write(dst_uri, reader, schema=first.schema)
    if any(_open(u).describe_indices() for u, _ in src_parts):
        new.create_scalar_index("recording_id", "BTREE")


def tables_under(root) -> list[str]:
    """Every Lance table under `root`, relative to it."""
    out = [d for d in list_dirs(root) if d.endswith(".lance")]
    out += [
        f"steps/{d}" for d in list_dirs(join(root, "steps")) if d.endswith(".lance")
    ]
    out += [u[len(str(root).rstrip("/")) + 1 :] for u in new_steps_uris(root).values()]
    return sorted(set(out))


def migrate(dst, p: dict, tag: str) -> None:
    for rel, src in sorted(p["files"].items()):
        _copy_file(join(src, rel), join(dst, rel))
    for key, parts in sorted(p["steps_sources"].items()):
        _copy_steps(parts, join(dst, key))
    for name, t in p["copied"].items():
        if t is not None:
            _write(join(dst, name), t)
    if p["copied"]["tasks.lance"] is not None:
        drop_task_annotations(dst)
    if p["changes"] is not None:
        _write(join(dst, "changes.lance"), p["changes"])
    for rel, src in p["others"].items():
        ds = _open(join(src, rel))
        _write(join(dst, rel), ds.to_table(), schema=ds.schema)
    # Last, as in commit.py: a recording is published by its recordings row.
    _write(join(dst, "recordings.lance"), p["recordings"])
    for rel in tables_under(dst):
        ds = _open(join(dst, rel))
        ds.tags.create(tag, ds.version)


def verify(dst, p: dict) -> None:
    """Raise unless the new root holds exactly the sources' recordings and steps rows."""
    old_ids = set()
    old_rows: Counter = Counter()
    for src in p["srcs"]:
        old_ids |= set(
            _open(join(src, "recordings.lance"))
            .to_table(columns=["recording_id"])[0]
            .to_pylist()
        )
        for uri in old_steps_uris(src).values():
            old_rows.update(
                _open(uri).to_table(columns=["recording_id"])[0].to_pylist()
            )
    for r in p["orphans"]:
        del old_rows[r]
    new_rec = _open(join(dst, "recordings.lance")).to_table()
    for name in COPIED_TABLES:
        if _exists(join(dst, name)):
            ids = _open(join(dst, name)).to_table(columns=["recording_id"])[0]
            if not set(ids.to_pylist()) <= set(new_rec["recording_id"].to_pylist()):
                raise AssertionError(f"{name} has rows of unknown recordings")
    if set(new_rec["recording_id"].to_pylist()) != old_ids:
        raise AssertionError("recording_id sets differ")
    if new_rec.num_rows != len(old_ids):
        raise AssertionError("recordings rows duplicated")
    if new_rec.schema != RECORDINGS:
        raise AssertionError(f"recordings schema {new_rec.schema}")
    seen: dict[str, str] = {}
    new_rows: Counter = Counter()
    for key, uri in new_steps_uris(dst).items():
        ids = _open(uri).to_table(columns=["recording_id"])[0].to_pylist()
        runs = [r for i, r in enumerate(ids) if i == 0 or ids[i - 1] != r]
        if len(runs) != len(set(runs)):
            raise AssertionError(f"{uri}: a recording's rows are not contiguous")
        for r in set(runs):
            if r in seen:
                raise AssertionError(f"{r} in steps tables {seen[r]} and {key}")
            if p["key_of"][r] != key:
                raise AssertionError(
                    f"{r} in {key}, its recording says {p['key_of'][r]}"
                )
            seen[r] = key
        new_rows.update(ids)
    if new_rows != old_rows:
        raise AssertionError(
            f"steps rows differ: {sum(old_rows.values())} -> {sum(new_rows.values())}"
        )
    for name, n in p["tables"].items():
        if _open(join(dst, name)).count_rows() != n:
            raise AssertionError(f"{name}: expected {n} rows")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        action="append",
        required=True,
        help="old-schema root, local or s3://; repeat to merge roots",
    )
    ap.add_argument("--dst", required=True, help="new root; must not exist or be empty")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--lab-for-null-site", help="lab for recordings whose old `site` is null"
    )
    ap.add_argument(
        "--drop-orphan-steps",
        action="store_true",
        help="drop steps, tasks, aliases and annotations rows of recordings missing "
        "from recordings.lance",
    )
    ap.add_argument("--tag", default="migrate-lab-end-effector")
    a = ap.parse_args(argv)
    srcs, dst = [s.rstrip("/") for s in a.src], a.dst.rstrip("/")
    if len(set(srcs)) != len(srcs):
        raise SystemExit("--src given twice")
    for src in srcs:
        if src == dst or dst.startswith(src + "/") or src.startswith(dst + "/"):
            raise SystemExit(f"--dst must not overlap --src {src}")
    p = plan(srcs, a.lab_for_null_site, a.drop_orphan_steps)
    for name, n in p["tables"].items():
        print(f"{n:>10}  {name}")
    if p["orphans"]:
        print(f"dropping {sum(p['orphans'].values())} orphan steps rows")
    for name, n in p["orphan_rows"].items():
        print(f"dropping {n} {name} rows of recordings not in recordings.lance")
    if p["task_annotations"] is not None:
        print(
            f"dropping tasks.annotations: {p['task_annotations']} rows have a "
            "non-empty value"
        )
    tagged = {
        join(src, rel): sorted(_open(join(src, rel)).tags.list())
        for src in srcs
        for rel in tables_under(src)
    }
    if any(tagged.values()):
        print(
            f"source tags not carried over: { {k: v for k, v in tagged.items() if v} }"
        )
    if a.dry_run:
        return
    if _nonempty(dst):
        raise SystemExit(f"{dst} is not empty")
    migrate(dst, p, a.tag)
    verify(dst, p)
    print(f"verified {dst}; tables tagged {a.tag}")


if __name__ == "__main__":
    main()
