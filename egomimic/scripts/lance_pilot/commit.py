"""Validate Aria deliveries and write them into the Lance tables (the pilot's committer).

Recording and task metadata come from SQL; per-frame data and video from each delivery
written by aria_to_lance. New recordings are appended; committed ones are skipped unless
--replace. See egomimic/rldb/lance/commit.py for the table operations.
"""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import text
from torchcodec.decoders import VideoDecoder

from egomimic.rldb.lance import commit
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_timestamp_ms,
    timestamp_ms_to_episode_hash,
)


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
    for cam in ("front_1", "front_1_480"):
        frames = VideoDecoder(
            str(d / f"{cam}.mp4"), seek_mode="exact"
        ).metadata.num_frames
        if frames != n:
            errs.append(f"{cam}.mp4 has {frames} frames, steps has {n}")
    return errs


def text_or_none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def objects_list(v) -> list[str]:
    if v is None:
        return []
    items = list(v) if not isinstance(v, str) else v.strip("{}").split(",")
    return [str(o).strip('"') for o in items if o and str(o) not in ("None", "NULL")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--deliveries",
        type=Path,
        default=Path("/coc/flash7/scratch/lance_pilot/deliveries"),
    )
    ap.add_argument(
        "--lance", type=Path, default=Path("/coc/flash7/scratch/lance_pilot/lance")
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="replace committed recordings with these deliveries",
    )
    ap.add_argument("--tag", help="tag every table's resulting version")
    args = ap.parse_args()

    dirs = sorted(p for p in args.deliveries.iterdir() if (p / "meta.json").exists())
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
        elif m["embodiment"] != sql.loc[h].embodiment:
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
            lab="aria",
            site=text_or_none(s.lab),
            embodiment=text_or_none(s.embodiment),
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
            task_instance_id=str(uuid.uuid4()),
            recording_id=rid,
            t_start_ns=m["t_start_ns"],
            t_end_ns=m["t_end_ns"],
            task_name=text_or_none(s.task),
            task_description=text_or_none(s.task_description),
            objects=objects_list(s.objects),
            annotations=[],
        )
        alias = dict(
            episode_hash=h,
            recording_id=rid,
            t_start_ns=m["t_start_ns"],
            t_end_ns=m["t_end_ns"],
        )
        videos = {cam: d / f"{cam}.mp4" for cam in ("front_1", "front_1_480")}
        deliveries.append(commit.Delivery(recording, task, alias, steps, videos))

    have = commit.committed_ids(args.lance)
    added = commit.append(args.lance, deliveries)
    replaced = []
    if args.replace:
        for d in deliveries:
            if d.recording_id in have:
                commit.replace(args.lance, d)
                replaced.append(d.recording_id)
    print(
        f"appended {len(added)}, replaced {len(replaced)}, skipped {len(deliveries) - len(added) - len(replaced)} already committed"
    )
    if args.tag:
        print(f"tagged {args.tag}: {commit.tag(args.lance, args.tag)}")
    for k, v in rejected.items():
        print(f"rejected {k}: {'; '.join(v)}")


if __name__ == "__main__":
    main()
