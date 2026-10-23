"""Loader benchmark: zarr vs Lance on the same Aria recordings, through the real keymap and transforms.

Prints one JSON line: resolve time, time to first batch, samples/s, worker startup, RSS.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader, Dataset

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.utils.gpu_orphans import orphan_guarded


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


class PerSample(Dataset):
    """Hides MultiDataset.__getitems__, so the DataLoader fetches one sample at a time."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


def rss_mb(proc: psutil.Process) -> float:
    total = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 2**20


def build(args):
    key_map = Human.get_keymap(keymap_mode="keypoints", include_ee_pose=True)
    if args.no_images:
        key_map = {
            k: v for k, v in key_map.items() if v.get("key_type") != "camera_keys"
        }
    transforms = Human.get_transform_list(
        mode="keypoints_wristframe_6d",
        include_ee_pose=True,
        pad_proprio_gripper=True,
        stride=3,
    )
    hashes = (
        [h.strip() for h in Path(args.hashes).read_text().split() if h.strip()]
        if args.hashes
        else None
    )
    filters = DatasetFilter(episode_hashes=hashes) if hashes else None
    if args.format == "zarr":
        resolver = LocalEpisodeResolver(
            args.zarr_dir, key_map=key_map, transform_list=transforms
        )
        leaves = resolver.resolve(filters=filters, expected_embodiment="human_bimanual")
    else:
        from egomimic.rldb.lance.dataset import (
            LanceEpisodeResolver,
            LanceStreamingResolver,
        )

        kw = dict(
            key_map=key_map,
            transform_list=transforms,
            cameras={} if args.no_images else {"images.front_1": args.camera},
        )
        resolver = (
            LanceStreamingResolver(
                args.lance_root,
                fetch_split=args.fetch_split or None,
                overlap_reads=not args.no_overlap,
                **kw,
            )
            if args.stream
            else LanceEpisodeResolver(args.lance_root, **kw)
        )
        leaves = resolver.resolve(filters=filters, expected_embodiment="human_bimanual")
    ds = MultiDataset(datasets=leaves, mode="total")
    return (PerSample(ds) if args.per_sample else ds), len(leaves)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=["zarr", "lance"], required=True)
    ap.add_argument(
        "--zarr-dir", type=Path, default=Path("/coc/flash7/scratch/lance_pilot/zarr")
    )
    ap.add_argument(
        "--lance-root",
        default="/coc/flash7/scratch/lance_pilot/lance_v2",
        help="local path or s3:// URI",
    )
    ap.add_argument("--camera", default="front_1", help="front_1 (360p) or front_1_480")
    ap.add_argument(
        "--no-images", action="store_true", help="drop camera keys: numerics only"
    )
    ap.add_argument(
        "--hashes", type=Path, help="whitespace-separated episode hashes to pin"
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="read through LanceStreamingResolver: batched fetches, MP4 byte ranges",
    )
    ap.add_argument(
        "--per-sample", action="store_true", help="disable the batched __getitems__"
    )
    ap.add_argument(
        "--fetch-split",
        type=int,
        default=0,
        help="samples per concurrent part; 0 = one",
    )
    ap.add_argument(
        "--no-overlap", action="store_true", help="steps take, then chunk reads"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--context",
        default=None,
        help="fork | forkserver | spawn (default: fork for zarr, forkserver for lance)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    context = args.context or ("fork" if args.format == "zarr" else "forkserver")
    if context == "forkserver":
        mp.set_forkserver_preload(
            ["torch", "egomimic.rldb.lance.dataset", "egomimic.rldb.embodiment.human"]
        )

    t0 = time.perf_counter()
    ds, n_leaves = build(args)
    t_build = time.perf_counter() - t0

    params = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        prefetch_factor=2 if args.workers else None,
        persistent_workers=args.workers > 0,
        multiprocessing_context=context if args.workers else None,
    )
    loader = DataLoader(ds, **orphan_guarded(params))
    t1 = time.perf_counter()
    it = iter(loader)
    next(it)
    t_first = time.perf_counter() - t1
    for _ in range(args.warmup):
        next(it)
    proc = psutil.Process(os.getpid())
    t2 = time.perf_counter()
    for _ in range(args.batches):
        next(it)
    dt = time.perf_counter() - t2
    print(
        json.dumps(
            dict(
                label=args.label,
                format=args.format,
                camera="none"
                if args.no_images
                else (args.camera if args.format == "lance" else "jpeg"),
                context=context,
                stream=args.stream,
                fetch="per_sample" if args.per_sample else "batched",
                fetch_split=args.fetch_split,
                overlap=not args.no_overlap,
                root="r2" if str(args.lance_root).startswith("s3://") else "local",
                workers=args.workers,
                batch_size=args.batch_size,
                recordings=n_leaves,
                samples=len(ds),
                build_s=round(t_build, 2),
                first_batch_s=round(t_first, 2),
                samples_per_s=round(args.batches * args.batch_size / dt, 1),
                rss_mb=round(rss_mb(proc)),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
