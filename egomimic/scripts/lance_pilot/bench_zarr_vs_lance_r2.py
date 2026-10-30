"""Zarr vs Lance read throughput from R2 or local disk, numerics with or without one 480p frame per sample.

  take    one random 31-frame window per sample (today's random-access loader pattern)
  window  each worker reads whole recordings (prefetching the next), cuts every 31-frame window,
          and serves them through a shuffle buffer; frames stay compressed in memory until served

zarr reads `processed_v3/aria/<hash>.zarr` (sharded zarr v3, JPEG frames); Lance reads the pilot
tables and faststart MP4s under `--lance-root`. Window mode times one full pass over all recordings.
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from egomimic.rldb.lance.dataset import _decoder, _s3_client, _table, open_table

H = 31
ZARR_KEYS = [
    f"{s}.obs_{k}"
    for s in ("left", "right")
    for k in ("ee_pose", "keypoints", "wrist_pose")
] + ["obs_head_pose"]
LANCE_COLS = [
    f"{s}.{k}"
    for s in ("left", "right")
    for k in ("ee_pose", "keypoints", "wrist_pose")
] + ["head_pose"]
ZARR_IMG = "images.front_1"


def _zarr_group(h, root):
    import zarr

    if not root.startswith("s3://"):
        return zarr.open_group(f"{root}/{h}", mode="r")
    from obstore.store import S3Store

    store = S3Store(
        "rldb",
        prefix=f"{root[len('s3://rldb/'):].rstrip('/')}/{h}.zarr",
        endpoint=os.environ["R2_ENDPOINT_URL"],
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region="auto",
    )
    return zarr.open_group(
        store=zarr.storage.ObjectStore(store, read_only=True), mode="r"
    )


def _jpeg(b):
    import simplejpeg

    return torch.from_numpy(simplejpeg.decode_jpeg(b, colorspace="RGB"))


def _flat(tbl, col):
    if col in tbl.schema.names:
        return np.asarray(tbl.column(col).combine_chunks().flatten()).reshape(
            tbl.num_rows, -1
        )
    path = col.split(".")
    c = tbl.column(path[0]).combine_chunks()
    for p in path[1:]:
        c = c.field(p)
    return np.asarray(c.flatten()).reshape(len(c), -1)


class Recs:
    """Per-recording metadata, built in the parent and pickled to workers."""

    def __init__(self, fmt, hashes, lance_root, zarr_root, cam):
        self.fmt, self.lance_root, self.zarr_root = fmt, lance_root, zarr_root
        rec = open_table(f"{lance_root}/recordings.lance").to_table().to_pandas()
        ali = open_table(f"{lance_root}/aliases.lance").to_table().to_pandas()
        rec = rec.merge(ali, on="recording_id").set_index("episode_hash").loc[hashes]
        steps = open_table(f"{lance_root}/steps/aria.lance")
        self.steps_uri, self.steps_version = (
            f"{lance_root}/steps/aria.lance",
            steps.version,
        )
        ids = (
            steps.to_table(columns=["recording_id"])
            .column(0)
            .to_numpy(zero_copy_only=False)
        )
        change = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        starts = dict(zip(ids[change], change))
        self.items = []
        for h, row in rec.iterrows():
            self.items.append(
                dict(
                    hash=h,
                    row_start=int(starts[row.recording_id]),
                    n=int(row.num_frames),
                    video=f"{lance_root}/{dict(row.video_path)[cam]}",
                )
            )


class TakeDS(Dataset):
    def __init__(self, recs, images):
        self.recs, self.images = recs, images
        self._z = {}

    def __len__(self):
        return 10**9

    def _zarr(self, h):
        if h not in self._z:
            g = _zarr_group(h, self.recs.zarr_root)
            self._z[h] = (g, g[ZARR_KEYS[0]].shape[0])
        return self._z[h]

    def __getitem__(self, _):
        it = random.choice(self.recs.items)
        if self.recs.fmt == "zarr":
            g, n = self._zarr(it["hash"])
            s = random.randrange(n - H)
            out = {k: torch.from_numpy(np.asarray(g[k][s : s + H])) for k in ZARR_KEYS}
            if self.images:
                out["img"] = _jpeg(g[ZARR_IMG][s : s + 1][0])
            return out
        s = random.randrange(it["n"] - H)
        t = _table(self.recs.steps_uri, self.recs.steps_version).take(
            list(range(it["row_start"] + s, it["row_start"] + s + H)),
            columns=LANCE_COLS,
        )
        out = {c: torch.from_numpy(_flat(t, c)) for c in LANCE_COLS}
        if self.images:
            out["img"] = _decoder(it["video"]).get_frame_at(s).data
        return out


class WindowDS(IterableDataset):
    def __init__(self, recs, images, buffer):
        self.recs, self.images, self.buffer = recs, images, buffer

    def _load(self, it):
        if self.recs.fmt == "zarr":
            g = _zarr_group(it["hash"], self.recs.zarr_root)
            arrs = {k: np.asarray(g[k][:]) for k in ZARR_KEYS}
            frames = g[ZARR_IMG][:] if self.images else None
            return arrs, (lambda i: _jpeg(frames[i])) if self.images else None
        from torchcodec.decoders import VideoDecoder

        n = it["n"]
        t = _table(self.recs.steps_uri, self.recs.steps_version).to_table(
            columns=LANCE_COLS, offset=it["row_start"], limit=n
        )
        arrs = {c: _flat(t, c) for c in LANCE_COLS}
        if not self.images:
            return arrs, None
        if it["video"].startswith("s3://"):
            bucket, key = it["video"][len("s3://") :].split("/", 1)
            mp4 = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        else:
            mp4 = Path(it["video"]).read_bytes()
        dec = VideoDecoder(mp4, seek_mode="approximate", dimension_order="NHWC")
        # torchcodec does not hold a reference to the bytes; freeing them segfaults the next decode.
        return arrs, lambda i, _keep=mp4: dec.get_frame_at(i).data

    def __iter__(self):
        w = get_worker_info()
        items = self.recs.items[w.id :: w.num_workers] if w else list(self.recs.items)
        rng = random.Random(0 if w is None else w.id)
        rng.shuffle(items)
        buf = []
        with cf.ThreadPoolExecutor(1) as pool:
            nxt = pool.submit(self._load, items[0]) if items else None
            for i in range(len(items)):
                arrs, frame = nxt.result()
                nxt = (
                    pool.submit(self._load, items[i + 1])
                    if i + 1 < len(items)
                    else None
                )
                n = len(next(iter(arrs.values())))
                for s in range(n - H):
                    buf.append((arrs, frame, s))
                    if len(buf) >= self.buffer:
                        yield self._emit(buf.pop(rng.randrange(len(buf))))
        rng.shuffle(buf)
        for e in buf:
            yield self._emit(e)

    @staticmethod
    def _emit(e):
        arrs, frame, s = e
        out = {k: torch.from_numpy(a[s : s + H]) for k, a in arrs.items()}
        if frame is not None:
            out["img"] = frame(s)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=["zarr", "lance"], required=True)
    ap.add_argument("--mode", choices=["take", "window"], required=True)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--hashes", type=Path, required=True)
    ap.add_argument("--lance-root", required=True)
    ap.add_argument(
        "--zarr-root",
        default="s3://rldb/processed_v3/aria",
        help="s3://bucket/prefix of <hash>.zarr, or a local dir of <hash>/",
    )
    ap.add_argument(
        "--camera",
        default="front_1_480",
        help="Lance video: front_1_480 or front_1 (360p)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=60, help="take mode only")
    ap.add_argument("--buffer", type=int, default=3000)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    hashes = [h for h in args.hashes.read_text().split() if h]
    recs = Recs(args.format, hashes, args.lance_root, args.zarr_root, args.camera)
    ds = (
        TakeDS(recs, args.images)
        if args.mode == "take"
        else WindowDS(recs, args.images, args.buffer)
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        multiprocessing_context="forkserver" if args.workers else None,
        prefetch_factor=4 if args.workers else None,
    )
    t0 = time.perf_counter()
    it = iter(loader)
    next(it)
    first = time.perf_counter() - t0
    t1, n = time.perf_counter(), 0
    for i, b in enumerate(it):
        n += len(b["obs_head_pose" if args.format == "zarr" else "head_pose"])
        if args.mode == "take" and i + 1 >= args.batches:
            break
    dt = time.perf_counter() - t1
    print(
        json.dumps(
            dict(
                label=args.label,
                format=args.format,
                camera=args.camera if args.format == "lance" else "jpeg480",
                mode=args.mode,
                images=args.images,
                workers=args.workers,
                first_s=round(first, 1),
                samples=n,
                seconds=round(dt, 1),
                samples_per_s=round(n / dt, 1),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
