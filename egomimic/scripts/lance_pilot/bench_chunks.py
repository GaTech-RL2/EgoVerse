"""Random-sample read throughput with frames from K-frame video chunks stored as Lance blob rows.

Each sample: one random 31-row steps window and the 360p frame at its first row.

  per_sample  __getitem__: one `take` and one `read_blobs` per sample (today's reader shape)
  batched     __getitems__: one `take` for the batch's windows and one `read_blobs` for its
              chunks, so a worker pays about two round trips per batch

`--lance-root` holds steps/catalog tables and `video_chunks/<camera>_k<K>.lance`
(`chunk_videos.py`); local path or s3://. Prints one JSON line per run.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from egomimic.rldb.lance.dataset import _runs, _table, open_table

H = 31
COLS = [
    f"{s}.{k}"
    for s in ("left", "right")
    for k in ("ee_pose", "keypoints", "wrist_pose")
] + ["head_pose"]


def _flat(tbl, col):
    path = col.split(".")
    c = tbl.column(path[0]).combine_chunks()
    for p in path[1:]:
        c = c.field(p)
    return np.asarray(c.flatten()).reshape(len(c), -1)


def _starts(uri):
    ds = open_table(uri)
    values, starts = _runs(ds.to_table(columns=["recording_id"]).column(0))
    return ds.version, dict(zip(values, starts.tolist()))


class Recs:
    """Per-recording offsets into steps and the chunk table; built once, pickled to workers."""

    def __init__(self, root, hashes, camera, k):
        rec = open_table(f"{root}/recordings.lance").to_table().to_pandas()
        ali = open_table(f"{root}/aliases.lance").to_table().to_pandas()
        rec = rec.merge(ali, on="recording_id").set_index("episode_hash").loc[hashes]
        self.steps_uri = f"{root}/steps/aria.lance"
        self.chunks_uri = f"{root}/video_chunks/{camera}_k{k}.lance"
        self.steps_version, steps = _starts(self.steps_uri)
        self.chunks_version, chunks = _starts(self.chunks_uri)
        self.k = k
        self.items = [
            (steps[r.recording_id], chunks[r.recording_id], int(r.num_frames))
            for r in rec.itertuples()
        ]


class ChunkDS(Dataset):
    def __init__(self, recs, images):
        self.recs, self.images = recs, images
        self.timing = {"take": 0.0, "blobs": 0.0, "decode": 0.0}

    def __len__(self):
        return 10**9

    def _draw(self):
        s0, c0, n = random.choice(self.recs.items)
        f = random.randrange(n - H)
        return s0 + f, c0 + f // self.recs.k, f % self.recs.k

    def _fetch(self, draws):
        r = self.recs
        t = time.perf_counter()
        rows = np.concatenate([np.arange(s, s + H) for s, _, _ in draws])
        uniq, inv = np.unique(rows, return_inverse=True)
        tbl = _table(r.steps_uri, r.steps_version).take(uniq.tolist(), columns=COLS)
        cols = {c: _flat(tbl, c)[inv].reshape(len(draws), H, -1) for c in COLS}
        self.timing["take"] += time.perf_counter() - t
        out = [
            {c: torch.from_numpy(cols[c][i]) for c in COLS} for i in range(len(draws))
        ]
        if not self.images:
            return out
        from torchcodec.decoders import VideoDecoder

        t = time.perf_counter()
        blobs = _table(r.chunks_uri, r.chunks_version).read_blobs(
            "data", indices=[c for _, c, _ in draws], preserve_order=True
        )
        self.timing["blobs"] += time.perf_counter() - t
        t = time.perf_counter()
        for o, (_, _, off), (_, data) in zip(out, draws, blobs):
            dec = VideoDecoder(data, seek_mode="exact", dimension_order="NHWC")
            o["img"] = dec.get_frame_at(off).data
            del dec
        self.timing["decode"] += time.perf_counter() - t
        return out

    def __getitem__(self, _):
        return self._fetch([self._draw()])[0]


class BatchedChunkDS(ChunkDS):
    def __getitems__(self, idxs):
        return self._fetch([self._draw() for _ in idxs])


def run(args, recs):
    cls = BatchedChunkDS if args.mode == "batched" else ChunkDS
    ds = cls(recs, args.images)
    if args.workers == 0:
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    else:
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.workers,
            multiprocessing_context="forkserver",
            prefetch_factor=4,
            persistent_workers=False,
        )
    it = iter(dl)
    t0 = time.perf_counter()
    lat = []
    for _ in range(args.warmup):
        next(it)
    first = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(args.batches):
        tb = time.perf_counter()
        next(it)
        lat.append(time.perf_counter() - tb)
    secs = time.perf_counter() - t1
    out = dict(
        label=args.label,
        mode=args.mode,
        k=args.k,
        images=args.images,
        workers=args.workers,
        batch_size=args.batch_size,
        root="r2" if args.lance_root.startswith("s3://") else "local",
        warmup_s=round(first, 2),
        samples_per_s=round(args.batches * args.batch_size / secs, 1),
        batch_wait_p50_ms=round(1000 * float(np.percentile(lat, 50)), 1),
        batch_wait_p95_ms=round(1000 * float(np.percentile(lat, 95)), 1),
    )
    if args.workers == 0:
        n = (args.warmup + args.batches) * args.batch_size
        out.update(
            {
                f"{k}_ms_per_sample": round(1000 * ds.timing[k] / n, 2)
                for k in ("take", "blobs", "decode")
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lance-root", required=True)
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--camera", default="front_1")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", choices=["per_sample", "batched"], default="batched")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    hashes = open(args.hashes).read().split()
    recs = Recs(args.lance_root.rstrip("/"), hashes, args.camera, args.k)
    print(json.dumps(run(args, recs)), flush=True)


if __name__ == "__main__":
    main()
