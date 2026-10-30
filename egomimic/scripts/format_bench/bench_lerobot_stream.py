"""StreamingLeRobotDataset bench (LeRobot venv): same JSON fields as bench_lerobot.py.

LeRobot 0.6.1's streaming class as shipped, with three bench-only fixes it needs to read this
dataset at all:
  data      `load_dataset` reads `<root>/data/*/*.parquet` (local or s3://) instead of a Hub repo,
            filtered with parquet `filters` (its own `episodes=` is stored but never used)
  timestamp `make_frame` seeks video to `index / fps`, the frame's position in the whole
            dataset, but each concatenated mp4 restarts at 0; `index` is rewritten to the frame's
            position in its own video file (from_timestamp * fps + frame_index)
  workers   every worker builds the same shard iterators, and inside a worker HF hands each
            1-shard iterator to worker 0 only (the rest yield nothing); each worker instead
            streams its own episodes (episode % workers, a parquet filter that skips the other
            row groups) through the stock iterator with HF's own worker split off
Shuffling is LeRobot's: per-worker shard sampling plus a `--buffer` sample shuffle buffer, filled
before the first sample comes out (counted in startup).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from egomimic.scripts.format_bench import hooks
from egomimic.scripts.format_bench.bench_lerobot import (
    delta_timestamps,
    ds_features,
    mirror_meta,
)
from egomimic.scripts.format_bench.common import emit, evict, run_loader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--pattern", choices=["chunk1", "numeric", "frames8"], default="chunk1"
    )
    ap.add_argument(
        "--episodes", type=Path, help="episode indices to stream (whitespace separated)"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--context", default="fork")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--buffer",
        type=int,
        default=None,
        help="shuffle buffer (default 1000; 100 for frames8)",
    )
    ap.add_argument("--evict", nargs="*", default=[])
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    buffer = args.buffer or (100 if args.pattern == "frames8" else 1000)

    import fsspec
    import torch
    from lerobot.datasets import streaming_dataset as sd

    evicted = evict(*args.evict) if args.evict else 0
    remote = args.root.startswith("s3://")
    root = args.root.rstrip("/")
    episodes = (
        [int(x) for x in args.episodes.read_text().split()] if args.episodes else None
    )

    t0 = time.time()
    if remote:
        # Same s3fs settings as bench_lerobot.patch_s3: 256 KB readahead for video seeks, the
        # default 50 MB block for parquet.
        fsspec.config.conf.setdefault("s3", {}).update(
            default_cache_type="readahead", default_block_size=256 * 2**10
        )
        fs = fsspec.filesystem("s3")
        files = [f"s3://{p}" for p in sorted(fs.glob(f"{root}/data/*/*.parquet"))]
        storage_options = {
            "s3": {**fs.storage_options, "default_block_size": 50 * 2**20}
        }
        meta_root = Path(os.environ["FB_META_MIRROR"])
        mirror_meta(root, meta_root)
    else:
        files = sorted(str(p) for p in Path(root).glob("data/*/*.parquet"))
        storage_options = {}
        meta_root = Path(root)
    pattern = args.pattern
    orig_load = sd.load_dataset

    class Stream(sd.StreamingLeRobotDataset):
        def stream(self, eps):
            """The parquet rows of `eps`, `index` made relative to each row's video file."""
            hf = orig_load(
                "parquet",
                data_files=files,
                split="train",
                streaming=True,
                filters=None if eps is None else [("episode_index", "in", list(eps))],
                storage_options=storage_options,
            )
            if pattern == "numeric":
                return hf
            (vk,) = self.meta.video_keys
            e = self.meta.episodes
            ep_idx = np.asarray(e["episode_index"])
            off = np.zeros(ep_idx.max() + 1, np.int64)
            off[ep_idx] = np.round(
                np.asarray(e[f"videos/{vk}/from_timestamp"]) * self.fps
            ).astype(np.int64)
            return hf.map(
                lambda ep, fi: {"index": int(off[ep] + fi)},
                input_columns=["episode_index", "frame_index"],
            )

        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            if info is not None:
                eps = (
                    episodes
                    if episodes is not None
                    else range(self.meta.total_episodes)
                )
                self.hf_dataset = self.stream(
                    [e for e in eps if e % info.num_workers == info.id]
                )
                self.num_shards = min(self.hf_dataset.num_shards, self._max_shards)
                # HF splits shards across torch workers by itself; this worker's stream is
                # already its own, so let HF iterate it whole.
                torch.utils.data.get_worker_info = lambda: None
            yield from super().__iter__()

    sd.load_dataset = lambda *a, **kw: orig_load(
        "parquet",
        data_files=files[:1],
        split="train",
        streaming=True,
        storage_options=storage_options,
    )
    ds = Stream(
        "egoverse/aria_pilot",
        root=meta_root,
        delta_timestamps=delta_timestamps(pattern, "action" in ds_features(meta_root)),
        tolerance_s=1e-3,
        buffer_size=buffer,
    )
    ds._max_shards = 16
    if remote:
        ds.root = root  # video paths are f"{root}/{relpath}", opened through fsspec
    if pattern == "numeric":
        type(ds.meta).video_keys = property(lambda self: [])
    ds.hf_dataset = ds.stream(episodes)
    init_s = time.time() - t0

    res = run_loader(
        ds,
        workers=args.workers,
        context=args.context,
        batch_size=args.batch_size,
        batches=args.batches,
        warmup=args.warmup,
        hook=None if args.pattern == "numeric" else hooks.lerobot_stream_decoders,
    )
    n_eps = len(episodes) if episodes is not None else ds.meta.total_episodes
    emit(
        dict(
            label=args.label,
            format="lerobot_stream",
            location="s3" if remote else "local",
            root=args.root,
            pattern=args.pattern,
            workers=args.workers,
            context=args.context,
            batch_size=args.batch_size,
            repeat=args.repeat,
            episodes=n_eps,
            buffer=buffer,
            files=len(files),
            evicted_gb=round(evicted / 1e9, 2),
            resolve_s=round(init_s, 3),
            construct_s=0.0,
            **res,
            startup_total_s=round(init_s + res["first_batch_s"], 3),
        ),
        args.out,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
