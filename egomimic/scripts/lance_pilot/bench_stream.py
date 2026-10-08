"""Numeric-only read throughput: random-access `take` windows vs Lance's streaming loader.

  take    one 31-row window per sample at a random eligible row (our reader's access pattern)
  stream  lance.torch.data.LanceDataset batches in table order (sequential scan)
  window  the stream cut into 31-row windows within each recording, through a shuffle
          buffer: what a streaming training loader would hand the model

Frames are not decoded here; the MP4 path is the same for all three.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from egomimic.rldb.lance.dataset import _table, open_table, storage_options

COLS = [
    "left.ee_pose",
    "left.keypoints",
    "left.wrist_pose",
    "right.ee_pose",
    "right.keypoints",
    "right.wrist_pose",
    "head_pose",
]
H = 31


def flat(tbl, col):
    if col in tbl.schema.names:
        c = tbl.column(col)
        c = c.combine_chunks() if hasattr(c, "combine_chunks") else c
        return np.asarray(c.flatten()).reshape(tbl.num_rows, -1)
    path = col.split(".")
    c = tbl.column(path[0])
    c = c.combine_chunks() if hasattr(c, "combine_chunks") else c
    for p in path[1:]:
        c = c.field(p)
    return np.asarray(c.flatten()).reshape(len(c), -1)


class TakeWindows(Dataset):
    def __init__(self, uri, version, n_rows, n):
        self.uri, self.version, self.n_rows, self.n = uri, version, n_rows, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = random.randrange(self.n_rows - H)
        t = _table(self.uri, self.version).take(list(range(s, s + H)), columns=COLS)
        return {c: torch.from_numpy(flat(t, c)) for c in COLS}


class StreamWindows(IterableDataset):
    """Scans fragments (split across workers), cuts windows inside a recording, shuffles."""

    def __init__(self, uri, version, buffer):
        self.uri, self.version, self.buffer = uri, version, buffer

    def __iter__(self):
        ds = _table(self.uri, self.version)
        frags = ds.get_fragments()
        w = get_worker_info()
        if w is not None:
            frags = frags[w.id :: w.num_workers]
        rng = random.Random(0 if w is None else w.id)
        buf = []
        for frag in frags:
            for batch in ds.to_batches(
                columns=["recording_id", *COLS], fragments=[frag], batch_size=4096
            ):
                rid = batch.column(0).to_numpy(zero_copy_only=False)
                arrs = {c: flat(batch, c) for c in COLS}
                for s in range(0, len(rid) - H, 1):
                    if rid[s] != rid[s + H - 1]:
                        continue
                    buf.append({c: torch.from_numpy(arrs[c][s : s + H]) for c in COLS})
                    if len(buf) >= self.buffer:
                        yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        yield from buf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uri", required=True, help="steps table, local path or s3://")
    ap.add_argument("--mode", choices=["take", "stream", "window"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--buffer", type=int, default=10000)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    ds = open_table(args.uri)
    n_rows, version = ds.count_rows(), ds.version
    ctx = "forkserver" if args.workers else None

    t0 = time.perf_counter()
    if args.mode == "stream":
        from lance.torch.data import LanceDataset

        it = iter(
            LanceDataset(
                args.uri,
                batch_size=4096,
                # Its projection only takes top-level fields, so whole hand structs are read.
                columns=["left", "right", "head_pose"],
                dataset_options={"storage_options": storage_options(args.uri)},
                batch_readahead=16,
                to_tensor_fn=lambda batch, **_: batch,
            )
        )
        next(it)
        first = time.perf_counter() - t0
        rows, t1 = 0, time.perf_counter()
        for batch in it:
            rows += batch.num_rows
            if rows >= args.batches * args.batch_size * H:
                break
        rate = rows / (time.perf_counter() - t1)
        out = dict(rows_per_s=round(rate), samples_per_s_equiv=round(rate / H, 1))
    else:
        data = (
            TakeWindows(args.uri, version, n_rows, 10**9)
            if args.mode == "take"
            else StreamWindows(args.uri, version, args.buffer)
        )
        loader = DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            multiprocessing_context=ctx,
            persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers else None,
        )
        it = iter(loader)
        next(it)
        first = time.perf_counter() - t0
        t1 = time.perf_counter()
        for _ in range(args.batches):
            next(it)
        rate = args.batches * args.batch_size / (time.perf_counter() - t1)
        out = dict(samples_per_s=round(rate, 1), rows_per_s=round(rate * H))
    print(
        json.dumps(
            dict(
                label=args.label,
                mode=args.mode,
                uri=args.uri.split("/")[0] or "local",
                workers=args.workers,
                first_s=round(first, 2),
                **out,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
