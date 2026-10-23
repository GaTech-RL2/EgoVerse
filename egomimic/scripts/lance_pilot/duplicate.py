"""Duplicate the pilot recordings K times in both formats for loader scale tests.

Each copy gets a new recording_id / episode hash, so neither format can share index
entries or decoders between copies. `--link` hardlinks files (index-scale test, near-zero
disk, page-cache-resident); without it, files are real copies (IO-scale test).
"""

import argparse
import os
import shutil
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from egomimic.rldb.lance.schema import only_steps_path, recording_id
from egomimic.scripts.lance_pilot import PILOT


def place(src: Path, dst: Path, link: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def place_tree(src: Path, dst: Path, link: bool):
    if dst.exists():
        return
    shutil.copytree(src, dst, copy_function=os.link if link else shutil.copy2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=PILOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--factor", type=int, required=True)
    ap.add_argument("--hashes", type=Path, required=True)
    ap.add_argument("--link", action="store_true")
    ap.add_argument("--lance-only", action="store_true")
    args = ap.parse_args()

    hashes = set(args.hashes.read_text().split())
    src_l = args.src / "lance"
    rec = lance.dataset(str(src_l / "recordings.lance")).to_table().to_pandas()
    tasks = lance.dataset(str(src_l / "tasks.lance")).to_table().to_pandas()
    aliases = lance.dataset(str(src_l / "aliases.lance")).to_table().to_pandas()
    aliases = aliases[aliases.episode_hash.isin(hashes)]
    rec = rec[rec.recording_id.isin(aliases.recording_id)]
    steps_rel = only_steps_path(rec)
    steps_ds = lance.dataset(str(src_l / steps_rel))
    steps = steps_ds.to_table(
        filter=pc.field("recording_id").isin(list(rec.recording_id))
    )
    by_rid = aliases.set_index("recording_id").episode_hash

    out_l = args.out / "lance"
    new_rec, new_tasks, new_alias = [], [], []
    steps_uri = str(out_l / steps_rel)
    for k in range(args.factor):
        rid_map = {
            r: recording_id("aria", f"dup{k:04d}/{v}")
            for r, v in zip(rec.recording_id, rec.vendor_recording_id)
        }
        for r in rec.itertuples():
            h = by_rid[r.recording_id]
            nh = f"{h}-dup{k:04d}"
            nr = rid_map[r.recording_id]
            vids = []
            for cam, rel in r.video_path:
                nrel = f"videos/{nr}/{cam}.mp4"
                place(src_l / rel, out_l / nrel, args.link)
                vids.append((cam, nrel))
            new_rec.append(r._asdict() | {"recording_id": nr, "video_path": vids})
            if not args.lance_only:
                place_tree(args.src / "zarr" / h, args.out / "zarr" / nh, args.link)
            t = tasks[tasks.recording_id == r.recording_id].iloc[0].to_dict()
            new_tasks.append(
                t
                | {
                    "recording_id": nr,
                    "task_instance_id": f"{t['task_instance_id']}-{k}",
                }
            )
            a = aliases[aliases.recording_id == r.recording_id].iloc[0].to_dict()
            new_alias.append(a | {"recording_id": nr, "episode_hash": nh})
        ids = pc.cast(steps.column("recording_id"), pa.string()).to_pylist()
        copy = steps.set_column(0, "recording_id", pa.array([rid_map[i] for i in ids]))
        lance.write_dataset(
            copy,
            steps_uri,
            mode="append" if k else "overwrite",
            max_rows_per_file=4 << 20,
        )
        print(f"copy {k + 1}/{args.factor}", flush=True)

    for name, rows, src_name in (
        ("recordings", new_rec, "recordings"),
        ("tasks", new_tasks, "tasks"),
        ("aliases", new_alias, "aliases"),
    ):
        df = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
        schema = lance.dataset(str(src_l / f"{src_name}.lance")).schema
        lance.write_dataset(
            pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False),
            str(out_l / f"{name}.lance"),
            mode="overwrite",
        )
    print(
        f"{len(new_rec)} recordings, {steps.num_rows * args.factor:,} frames -> {args.out}"
    )


if __name__ == "__main__":
    main()
