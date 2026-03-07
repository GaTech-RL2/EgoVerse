"""
Sync EgoVerse data from S3/R2 to a local directory.

Progress (tqdm) and benchmarking live only here; the resolver does plain sync.
Works when s5cmd runs with many workers (parallel downloads): we poll the
filesystem for completed episode dirs and update the bar.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _count_present(local_dir: Path, to_sync: list[tuple[str, str]]) -> int:
    return sum(
        1
        for _, episode_hash in to_sync
        if S3EpisodeResolver.episode_already_present_for_progress_bar(local_dir, episode_hash)
    )


def run_resolver_sync(
    *,
    bucket_name: str,
    local_dir: str | Path,
    filters: dict,
    numworkers: int = 128,
    show_progress: bool = True,
    debug: bool = False,
):
    load_env()
    local_dir = Path(local_dir)

    to_sync, already = S3EpisodeResolver.get_to_sync_paths_for_progress_bar(
        filters, local_dir, debug=debug
    )
    if already:
        print(f"Skipping {len(already)} episodes already present locally.")
    if not to_sync:
        print("Nothing to sync (all episodes already present or none matched).")
        return []

    n = len(to_sync)
    sync_error: list[Exception] = []

    def do_sync() -> None:
        try:
            S3EpisodeResolver.run_sync_for_progress_bar(
                bucket_name=bucket_name,
                to_sync=to_sync,
                local_dir=local_dir,
                numworkers=numworkers,
            )
        except Exception as e:
            sync_error.append(e)

    t0 = time.perf_counter()
    thread = threading.Thread(target=do_sync, daemon=False)
    thread.start()

    if show_progress and tqdm is not None:
        with tqdm(
            total=n,
            unit="ep",
            desc="sync",
            dynamic_ncols=True,
        ) as pbar:
            while True:
                pbar.n = min(n, _count_present(local_dir, to_sync))
                pbar.refresh()
                if not thread.is_alive():
                    pbar.n = min(n, _count_present(local_dir, to_sync))
                    pbar.refresh()
                    break
                time.sleep(0.25)
    thread.join()

    if sync_error:
        raise sync_error[0]

    elapsed = time.perf_counter() - t0
    print(
        f"[sync_s3] Benchmark: {elapsed:.1f} s total for {n} episode(s) "
        f"(~{elapsed / n:.2f} s per episode)"
    )
    return [(p, h) for p, h in to_sync]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync EgoVerse data from S3/R2 to a local directory."
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        required=False,
        default="rldb",
        help="S3 bucket name (default: rldb).",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        required=False,
        default="/coc/flash7/scratch/egoverseS3ZarrDataset",
        help="Local directory to sync into.",
    )
    parser.add_argument(
        "--numworkers",
        type=int,
        required=False,
        default=128,
        help="s5cmd parallel workers.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show tqdm progress bar (default). Kept for backward compatibility.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (benchmark still printed).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Limit to 10 matched episodes (resolver debug mode).",
    )
    parser.add_argument(
        "--filters",
        type=str,
        required=False,
        default="{}",
        help='JSON dict of SQL filters, e.g. \'{"episode_hash": ["h1","h2"], "robot_name": "eva"}\'.',
    )
    args = parser.parse_args()

    filters = json.loads(args.filters)
    if not isinstance(filters, dict):
        raise ValueError("--filters must be a JSON object (dict).")

    run_resolver_sync(
        bucket_name=args.bucket_name,
        local_dir=args.local_dir,
        filters=filters,
        numworkers=args.numworkers,
        show_progress=args.show_progress or not args.no_progress,
        debug=args.debug,
    )
