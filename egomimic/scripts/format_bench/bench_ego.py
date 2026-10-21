"""zarr / Lance loader bench: startup phases and steady-state throughput, one JSON line per run.

Native loaders without transforms (LeRobot has no equivalent of our transforms, so all three
formats are measured on read + decode only):
  zarr   LocalEpisodeResolver over a directory of stores (local), or one ZarrDataset per store
         under an s3:// prefix opened through obstore (s3patch)
  zarr_fast  the same stores through format_bench/zarr_fast.py: arrays opened once per worker,
         shard indexes cached, a sample's (or a batch's) inner-chunk ranges read concurrently
  lance  LanceEpisodeResolver + MP4 decode (local), LanceStreamingResolver + MP4 ranges (s3://)
Patterns:
  chunk1   Aria keypoints keymap: 30-step action chunks, proprio, 1 front frame
  numeric  same keymap without the camera
  frames8  chunk1 with 8 consecutive front frames (camera history 8)
  rdt      RDT base's sample: chunk1 plus the front frame 0.1 s back (image_history_gap_s)
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

from egomimic.scripts.format_bench import (  # noqa: F401  (s3patch patches on import)
    hooks,
    s3patch,
)
from egomimic.scripts.format_bench.common import emit, evict, run_loader

CONTEXT = {"zarr": "fork", "zarr_fast": "fork", "lance": "forkserver"}


def keymap(pattern: str) -> dict:
    from egomimic.rldb.embodiment.human import Human

    km = Human.get_keymap(
        keymap_mode="keypoints",
        include_ee_pose=True,
        image_history_gap_s=0.1 if pattern == "rdt" else None,
    )
    if pattern == "numeric":
        return {k: v for k, v in km.items() if v.get("key_type") != "camera_keys"}
    if pattern == "frames8":
        return {
            k: {**v, "history": 8} if v.get("key_type") == "camera_keys" else v
            for k, v in km.items()
        }
    return km


def zarr_s3_leaves(root: str, key_map: dict, embodiment: str | None) -> dict:
    """One LIST for the store prefixes, then a ZarrDataset per store (the s3 analogue of the
    local resolver's directory listing)."""
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

    store = s3patch.obstore_s3(root)
    names = sorted(
        p.rstrip("/").split("/")[-1]
        for p in store.list_with_delimiter()["common_prefixes"]
    )
    leaves = {n: ZarrDataset(f"{root.rstrip('/')}/{n}", key_map, None) for n in names}
    if embodiment:
        leaves = {n: d for n, d in leaves.items() if d.embodiment.lower() == embodiment}
    return leaves


def build(args, key_map):
    from egomimic.rldb.filters import DatasetFilter

    # The training configs' filter; expected_embodiment alone only warns in LocalEpisodeResolver.
    filters = (
        DatasetFilter(
            filter_lambdas=[f"lambda row: row['embodiment'] == '{args.embodiment}'"]
        )
        if args.embodiment
        else None
    )
    remote = args.root.startswith("s3://")
    t0 = time.time()
    if args.format in ("zarr", "zarr_fast"):
        from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver
        from egomimic.scripts.format_bench import zarr_fast

        fast = args.format == "zarr_fast"
        if remote:
            leaves = (zarr_fast.s3_leaves if fast else zarr_s3_leaves)(
                args.root, key_map, args.embodiment
            )
        else:
            resolver = (
                zarr_fast.FastLocalEpisodeResolver if fast else LocalEpisodeResolver
            )
            leaves = resolver(
                Path(args.root), key_map=key_map, transform_list=None
            ).resolve(filters=filters, expected_embodiment=args.embodiment)
    else:
        from egomimic.rldb.embodiment.human import Human
        from egomimic.rldb.lance.dataset import (
            LanceEpisodeResolver,
            LanceStreamingResolver,
        )

        has_cam = any(v.get("key_type") == "camera_keys" for v in key_map.values())
        transforms = (
            Human.get_transform_list(mode="keypoints_headframe_6d", stride=3)
            if args.transforms
            else None
        )
        cams = {"images.front_1": args.camera} if has_cam else {}
        if remote:
            r = LanceStreamingResolver(
                args.root,
                key_map=key_map,
                transform_list=transforms,
                cameras=cams,
            )
        else:
            r = LanceEpisodeResolver(
                args.root, key_map=key_map, transform_list=transforms, cameras=cams
            )
        leaves = r.resolve(filters=filters, expected_embodiment=args.embodiment)
    t1 = time.time()
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    ds = MultiDataset(datasets=leaves, mode="total")
    t2 = time.time()
    return ds, leaves, t1 - t0, t2 - t1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=["zarr", "zarr_fast", "lance"], required=True)
    ap.add_argument(
        "--root", required=True, help="zarr store dir / Lance root, local or s3://"
    )
    ap.add_argument(
        "--pattern", choices=["chunk1", "numeric", "frames8", "rdt"], default="chunk1"
    )
    ap.add_argument(
        "--transforms",
        action="store_true",
        help="Lance only: apply the training transforms (keypoints_headframe_6d)",
    )
    ap.add_argument("--camera", default="front_1_480")
    ap.add_argument(
        "--embodiment", default="human_bimanual", help="'' for every recording"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--context", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--real",
        type=Path,
        help="first batch samples only these leaves (startup at scale)",
    )
    ap.add_argument(
        "--evict", nargs="*", default=[], help="paths to drop from the page cache first"
    )
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    args.embodiment = args.embodiment or None

    context = args.context or (
        "forkserver" if args.root.startswith("s3://") else CONTEXT[args.format]
    )
    if context == "forkserver":
        mp.set_forkserver_preload(
            [
                "torch",
                "egomimic.rldb.lance.dataset",
                "egomimic.rldb.zarr.zarr_dataset_multi",
                "egomimic.scripts.format_bench.s3patch",
                "egomimic.scripts.format_bench.hooks",
                "egomimic.scripts.format_bench.zarr_fast",
            ]
        )
    evicted = evict(*args.evict) if args.evict else 0
    km = keymap(args.pattern)
    ds, leaves, resolve_s, construct_s = build(args, km)
    indices = None
    if args.real:
        real = set(args.real.read_text().split())
        ranges = ds.index_map.ranges()
        indices = [i for n in leaves if n in real for i in ranges[n]]
    hook = None
    if (
        args.format == "lance"
        and not args.root.startswith("s3://")
        and args.pattern != "numeric"
    ):
        hook = hooks.lance_decoders
    res = run_loader(
        ds,
        workers=args.workers,
        context=context,
        batch_size=args.batch_size,
        batches=args.batches,
        warmup=args.warmup,
        hook=hook,
        indices=indices,
    )
    emit(
        dict(
            label=args.label,
            format=args.format,
            location="s3" if args.root.startswith("s3://") else "local",
            root=args.root,
            pattern=args.pattern,
            transforms=args.transforms,
            workers=args.workers,
            context=context,
            batch_size=args.batch_size,
            repeat=args.repeat,
            episodes=len(leaves),
            samples=len(ds),
            evicted_gb=round(evicted / 1e9, 2),
            resolve_s=round(resolve_s, 3),
            construct_s=round(construct_s, 3),
            **res,
            startup_total_s=round(resolve_s + construct_s + res["first_batch_s"], 3),
        ),
        args.out,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
