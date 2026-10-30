"""Review and commit annotation change files, and build one from existing zarr annotations.

    review       validate a change file against the tables and print the per-recording diff
    commit       the same, then apply it as one commit on annotations.lance (optionally tagged)
    migrate-zarr write a change file from the zarr episodes' annotations, mapped onto the
                 recording clock through each episode's obs_rgb_timestamps_ns

See egomimic/rldb/lance/annotations.py for the change file format.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from egomimic.rldb.lance import commit
from egomimic.rldb.lance.annotations import diff, read_changes, review
from egomimic.rldb.lance.schema import STEPS_KEY, steps_path

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
    message = f"{args.file}, approved by {args.approver} on {datetime.date.today()}"
    counts = commit.apply_annotations(args.lance, rv.adds, rv.deletes, message)
    print(f"committed {counts} from {message}")
    if args.tag:
        print(f"tagged {args.tag}: {commit.tag(args.lance, args.tag)}")


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


def cmd_migrate_zarr(args):
    root = args.lance
    aliases = lance.dataset(str(root / "aliases.lance")).to_table().to_pylist()
    recs = (
        lance.dataset(str(root / "recordings.lance"))
        .to_table(columns=["recording_id", "fps", *STEPS_KEY])
        .to_pylist()
    )
    fps = {r["recording_id"]: r["fps"] for r in recs}
    steps_of = {r["recording_id"]: steps_path(*(r[c] for c in STEPS_KEY)) for r in recs}
    tables = {p: lance.dataset(str(root / p)) for p in set(steps_of.values())}
    rows, dropped = [], {}
    for a in aliases:
        h, rid = a["episode_hash"], a["recording_id"]
        path = next(
            (p for p in (args.zarr / h, args.zarr / f"{h}.zarr") if p.exists()), None
        )
        if path is None:
            continue
        anns, zts = _zarr_annotations(path)
        if not anns:
            continue
        rts = (
            tables[steps_of[rid]]
            .to_table(columns=["timestamp_ns"], filter=f"recording_id = '{rid}'")
            .column(0)
        )
        rts = np.sort(rts.to_numpy())
        period = int(round(1e9 / fps[rid]))

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
            t0 = int(rts[i0])
            if e < len(zts):
                t1 = int(rts[last])
            else:
                t1 = (
                    int(rts[last + 1]) if last + 1 < len(rts) else int(rts[-1]) + period
                )
            rows.append(
                dict(
                    op="add",
                    recording_id=rid,
                    t_start_ns=t0,
                    t_end_ns=t1,
                    annotation_type="language",
                    text=str(d.get("text", "")),
                    annotation_vendor=args.vendor,
                )
            )
    pq.write_table(pa.Table.from_pylist(rows, schema=CHANGE_SCHEMA), args.out)
    n_dropped = sum(map(len, dropped.values()))
    print(f"wrote {len(rows)} annotations to {args.out}; dropped {n_dropped}")
    for h, why in sorted(dropped.items()):
        print(f"  {h}: {len(why)} dropped, e.g. {why[0]}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lance", type=Path, required=True, help="Lance root")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("review", help="validate a change file and print its diff")
    p.add_argument("file", type=Path)
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("commit", help="review, then commit a change file")
    p.add_argument("file", type=Path)
    p.add_argument("--approver", required=True)
    p.add_argument("--tag", help="tag every table's resulting version")
    p.set_defaults(fn=cmd_commit)

    p = sub.add_parser("migrate-zarr", help="change file from zarr annotations")
    p.add_argument(
        "--zarr", type=Path, required=True, help="dir of <episode_hash> zarrs"
    )
    p.add_argument("--vendor", required=True, help="annotation_vendor for every row")
    p.add_argument("--out", type=Path, required=True, help="parquet change file")
    p.set_defaults(fn=cmd_migrate_zarr)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
