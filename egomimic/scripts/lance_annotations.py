"""Review and commit annotation change files, and build them from zarr episodes or the
Scale annotation bucket.

    review       validate a change file against the tables and print the per-recording diff
    commit       the same, then apply it as one commit on annotations.lance, recorded in
                 changes.lance and tagged annotations-<UTC time>
    from-zarr    change file from zarr episodes' annotations, mapped onto the recording clock
                 through each episode's obs_rgb_timestamps_ns
    from-bucket  change file from <episode_hash>_<key>.json span lists, as
                 language_process/scale_to_bucket_annotation_parallel.py writes them

from-bucket counts a span's frames from its alias's first recording frame. That is exact for
EVA, whose zarr episodes hold every recording frame, but not for Aria clips that dropped
frames: map those with from-zarr.

`--lance` is a local directory or an s3:// prefix. See egomimic/rldb/lance/annotations.py for
the change file format.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from egomimic.rldb.lance import commit
from egomimic.rldb.lance.annotations import diff, read_changes, review
from egomimic.rldb.lance.schema import STEPS_KEY

CHANGE_SCHEMA = pa.schema(
    [
        ("op", pa.string()),
        ("recording_id", pa.string()),
        ("t_start_ns", pa.int64()),
        ("t_end_ns", pa.int64()),
        ("annotation_type", pa.string()),
        ("text", pa.string()),
        ("annotation_vendor", pa.string()),
    ]
)


def _review(args):
    rv = review(args.lance, read_changes(args.file))
    print(diff(args.lance, rv) or "no changes")
    for e in rv.errors:
        print(f"error: {e}")
    return rv


def cmd_review(args):
    sys.exit(1 if _review(args).errors else 0)


def cmd_commit(args):
    rv = _review(args)
    if rv.errors:
        sys.exit("not committed: fix the errors above")
    out = commit.commit_annotations(args.lance, rv.adds, rv.deletes, args.approver)
    if out is None:
        print(f"nothing to commit from {args.file}")
        return
    print(
        f"committed +{out['rows']['added']} -{out['rows']['deleted']} from {args.file} as {out['tag']}, "
        f"approved by {args.approver} on {datetime.date.today()}"
    )


def _aliases(root) -> dict[str, dict]:
    rows = commit._open(commit.join(root, "aliases.lance")).to_table().to_pylist()
    return {a["episode_hash"]: a for a in rows}


def _clocks(root, rids: set[str]) -> dict[str, tuple[np.ndarray, float]]:
    """{recording_id: (sorted steps timestamp_ns, fps)}, one read per steps table."""
    recs = commit._open(commit.join(root, "recordings.lance")).to_table(
        columns=["recording_id", "fps", *STEPS_KEY]
    )
    out = {}
    by_key: dict[tuple, list[dict]] = {}
    for r in recs.to_pylist():
        if r["recording_id"] in rids:
            by_key.setdefault(tuple(r[c] for c in STEPS_KEY), []).append(r)
    for key, mine in by_key.items():
        ids = ", ".join(commit._sql_str(r["recording_id"]) for r in mine)
        t = commit._open(commit.steps_uri(root, *key)).to_table(
            columns=["recording_id", "timestamp_ns"], filter=f"recording_id IN ({ids})"
        )
        rid_col = np.asarray(t.column("recording_id").to_pylist())
        ts = t.column("timestamp_ns").to_numpy()
        for r in mine:
            out[r["recording_id"]] = (
                np.sort(ts[rid_col == r["recording_id"]]),
                r["fps"],
            )
    return out


def _add(rid, t0, t1, text, vendor) -> dict:
    return dict(
        op="add",
        recording_id=rid,
        t_start_ns=int(t0),
        t_end_ns=int(t1),
        annotation_type="language",
        text=str(text),
        annotation_vendor=vendor,
    )


def _write(rows, dropped, out):
    pq.write_table(pa.Table.from_pylist(rows, schema=CHANGE_SCHEMA), out)
    print(
        f"wrote {len(rows)} annotations to {out}; dropped {sum(map(len, dropped.values()))}"
    )
    for h, why in sorted(dropped.items()):
        print(f"  {h}: {len(why)} dropped, e.g. {why[0]}")


def _zarr_annotations(path: Path) -> tuple[list[dict], np.ndarray]:
    z = zarr.open_group(str(path), mode="r")
    if "annotations" not in z:
        return [], np.empty(0, np.int64)
    anns = []
    for raw in z["annotations"][:]:
        d = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        if isinstance(d, dict):
            anns.append(d)
    # Arrays are padded to the chunk length; total_frames is the real count.
    n = int(z.attrs["total_frames"])
    return anns, np.asarray(z["obs_rgb_timestamps_ns"][:n], np.int64).reshape(-1)


def cmd_from_zarr(args):
    found = {}
    for h, a in _aliases(args.lance).items():
        path = next(
            (p for p in (args.zarr / h, args.zarr / f"{h}.zarr") if p.exists()), None
        )
        if path is not None:
            found[h] = (a["recording_id"], path)
    clocks = _clocks(args.lance, {rid for rid, _ in found.values()})
    rows, dropped = [], {}
    for h, (rid, path) in found.items():
        anns, zts = _zarr_annotations(path)
        if not anns:
            continue
        rts, fps = clocks[rid]
        period = int(round(1e9 / fps))

        def frame_of(t) -> int | None:
            i = int(np.searchsorted(rts, t))
            return i if i < len(rts) and rts[i] == t else None

        for d in anns:
            s, e = int(d.get("start_idx", -1)), int(d.get("end_idx", -1))
            if not 0 <= s < e <= len(zts):
                dropped.setdefault(h, []).append(f"bad range [{s}, {e})")
                continue
            i0 = frame_of(int(zts[s]))
            # An annotation running to the clip's end ends at the recording frame after its last.
            last = frame_of(int(zts[e if e < len(zts) else -1]))
            if i0 is None or last is None:
                dropped.setdefault(h, []).append(
                    f"[{s}, {e}) is off the recording clock"
                )
                continue
            if e < len(zts):
                t1 = rts[last]
            else:
                t1 = rts[last + 1] if last + 1 < len(rts) else rts[-1] + period
            rows.append(_add(rid, rts[i0], t1, d.get("text", ""), args.vendor))
    _write(rows, dropped, args.out)


def cmd_from_bucket(args):
    suffix = f"_{args.key}.json"
    aliases = _aliases(args.lance)
    names = [n for n in commit.list_files(args.bucket) if n.endswith(suffix)]
    hashes = {n[: -len(suffix)]: n for n in names}
    dropped = {h: ["no alias"] for h in hashes if h not in aliases}
    todo = {h: aliases[h] for h in hashes if h in aliases}
    clocks = _clocks(args.lance, {a["recording_id"] for a in todo.values()})
    rows = []
    for h, a in sorted(todo.items()):
        rts, _ = clocks[a["recording_id"]]
        clip = rts[(rts >= a["t_start_ns"]) & (rts < a["t_end_ns"])]
        spans = json.loads(commit.get_bytes(commit.join(args.bucket, hashes[h])))
        for d in spans:
            s, e = int(d["start_idx"]), int(d["end_idx"])
            if not 0 <= s < e <= len(clip):
                dropped.setdefault(h, []).append(
                    f"[{s}, {e}) is outside the alias's {len(clip)} frames"
                )
                continue
            t1 = clip[e] if e < len(clip) else a["t_end_ns"]
            rows.append(_add(a["recording_id"], clip[s], t1, d["text"], args.vendor))
    _write(rows, dropped, args.out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lance", required=True, help="Lance root (dir or s3:// prefix)")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("review", help="validate a change file and print its diff")
    p.add_argument("file", type=Path)
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("commit", help="review, then commit and tag a change file")
    p.add_argument("file", type=Path)
    p.add_argument("--approver", required=True)
    p.set_defaults(fn=cmd_commit)

    p = sub.add_parser("from-zarr", help="change file from zarr annotations")
    p.add_argument(
        "--zarr", type=Path, required=True, help="dir of <episode_hash> zarrs"
    )
    p.add_argument("--vendor", required=True, help="annotation_vendor for every row")
    p.add_argument("--out", type=Path, required=True, help="parquet change file")
    p.set_defaults(fn=cmd_from_zarr)

    p = sub.add_parser("from-bucket", help="change file from bucket span lists")
    p.add_argument("--bucket", required=True, help="dir or s3:// prefix of the JSONs")
    p.add_argument("--key", default="annotations", help="<episode_hash>_<key>.json")
    p.add_argument("--vendor", required=True, help="annotation_vendor for every row")
    p.add_argument("--out", type=Path, required=True, help="parquet change file")
    p.set_defaults(fn=cmd_from_bucket)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
