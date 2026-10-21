"""Random-sample read throughput with frames fetched from the MP4s by byte range.

Each sample is one random 31-row steps window and the frame at its first row. Frames
come from the faststart MP4s through egomimic.rldb.lance.mp4_ranges:
each worker reads a recording's moov once (one range GET), then a batch issues one range
GET per frame (its keyframe through itself), concurrently, and decodes in memory.
Prints one JSON line per run.
"""

import argparse
import concurrent.futures as cf
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from egomimic.rldb.lance import mp4_ranges
from egomimic.rldb.lance.dataset import _runs, _s3_client, _table, open_table

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


def run(args, ds):
    """Times `ds` through a DataLoader; `args` carries the loader and report settings."""
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
            {f"{k}_ms_per_sample": round(1000 * ds.timing[k] / n, 2) for k in ds.timing}
        )
    return out


class Recs:
    def __init__(self, root, hashes, camera):
        rec = open_table(f"{root}/recordings.lance").to_table().to_pandas()
        ali = open_table(f"{root}/aliases.lance").to_table().to_pandas()
        rec = rec.merge(ali, on="recording_id").set_index("episode_hash").loc[hashes]
        self.steps_uri = f"{root}/steps/aria.lance"
        self.steps_version, steps = _starts(self.steps_uri)
        self.items = [
            (
                steps[r.recording_id],
                f"{root}/{dict(r.video_path)[camera]}",
                int(r.num_frames),
            )
            for r in rec.itertuples()
        ]


def _read(url: str, a: int, b: int) -> bytes:
    if not url.startswith("s3://"):
        with open(url, "rb") as f:
            f.seek(a)
            return f.read(b - a)
    bucket, key = url[len("s3://") :].split("/", 1)
    return (
        _s3_client()
        .get_object(Bucket=bucket, Key=key, Range=f"bytes={a}-{b - 1}")["Body"]
        .read()
    )


class RangeDS(Dataset):
    def __init__(self, recs, images, threads):
        self.recs, self.images, self.threads = recs, images, threads
        self.timing = {"take": 0.0, "index": 0.0, "get": 0.0, "decode": 0.0}
        self._pool = None
        self._dec: dict[str, mp4_ranges.Decoder] = {}

    def __len__(self):
        return 10**9

    def _draw(self):
        s0, url, n = random.choice(self.recs.items)
        f = random.randrange(n - H)
        return s0 + f, url, f

    def _decoder(self, url):
        if url not in self._dec:
            idx = mp4_ranges.read_index(lambda a, b: _read(url, a, b))
            self._dec[url] = mp4_ranges.Decoder(idx)
        return self._dec[url]

    def _fetch(self, draws):
        if self._pool is None:
            _s3_client()
            self._pool = cf.ThreadPoolExecutor(self.threads)
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
        t = time.perf_counter()
        urls = sorted({u for _, u, _ in draws})
        decs = dict(zip(urls, self._pool.map(self._decoder, urls)))
        self.timing["index"] += time.perf_counter() - t
        t = time.perf_counter()
        jobs = [
            (u, rg)
            for u in urls
            for rg in mp4_ranges.ranges(
                decs[u].index, [f for _, uu, f in draws if uu == u]
            )
        ]
        got = self._pool.map(lambda j: _read(j[0], *j[1]), jobs)
        fetched: dict[str, dict] = {u: {} for u in urls}
        for (u, rg), data in zip(jobs, got):
            fetched[u][rg] = data
        self.timing["get"] += time.perf_counter() - t
        t = time.perf_counter()
        for o, (_, u, f) in zip(out, draws):
            o["img"] = torch.from_numpy(decs[u].frames([f], fetched[u])[0])
        self.timing["decode"] += time.perf_counter() - t
        return out

    def __getitem__(self, _):
        return self._fetch([self._draw()])[0]

    def __getitems__(self, idxs):
        return self._fetch([self._draw() for _ in idxs])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lance-root", required=True)
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--camera", default="front_1")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    hashes = open(args.hashes).read().split()
    recs = Recs(args.lance_root.rstrip("/"), hashes, args.camera)
    print(json.dumps(run(args, RangeDS(recs, args.images, args.threads))), flush=True)


if __name__ == "__main__":
    main()
