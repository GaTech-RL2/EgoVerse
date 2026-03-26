"""
Zero-row scanner for local Zarr episodes.

For every numeric array in each episode, finds frame indices where all values
are exactly zero.  Zero quaternions (0,0,0,0) embedded in a pose vec are the
classic sign of data corruption this was designed to catch.

Uses LocalEpisodeResolver to discover only valid, readable zarr stores.

Fast by design:
  - ThreadPool parallel reads (mostly I/O-bound, no pickling overhead).
  - Reads each array as a single numpy slice then checks with .all(axis=1).
  - tqdm bar advances per episode.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import random
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr import LocalEpisodeResolver

# ── per-episode worker ────────────────────────────────────────────────────────


def _scan_episode(ep_path: Path, episode_hash: str) -> dict:
    """
    Open the zarr store, check every numeric (T, ...) array for all-zero rows.
    Returns {episode_hash, zero_rows: {key: [frame_indices]}, error: str|None}.
    """
    eh = episode_hash
    try:
        g = zarr.open_group(str(ep_path), mode="r")
        total_frames = int(g.attrs.get("total_frames", 0) or 0)

        zero_rows: dict[str, list[int]] = {}
        for key in g.keys():
            arr = g[key]
            # Skip non-numeric or 1-D-only (annotations, jpeg stores)
            if arr.ndim < 2 or not np.issubdtype(arr.dtype, np.number):
                continue
            data: np.ndarray = arr[:]  # read whole array once
            T = data.shape[0]
            flat = data.reshape(T, -1)  # (T, features)
            zero_mask = (flat == 0).all(axis=1)
            bad = np.where(zero_mask)[0].tolist()
            if bad:
                zero_rows[key] = bad

        return {
            "episode_hash": eh,
            "total_frames": total_frames,
            "zero_rows": zero_rows,
            "error": None,
        }

    except Exception as e:
        return {"episode_hash": eh, "total_frames": 0, "zero_rows": {}, "error": str(e)}


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan local Zarr episodes for all-zero rows in numeric arrays."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=str,
        help="Directory containing episode dirs.",
    )
    parser.add_argument(
        "--pct",
        default=100.0,
        type=float,
        help="Percentage of episodes to scan (default 100 = all).",
    )
    parser.add_argument(
        "--workers",
        default=32,
        type=int,
        help="Parallel threads (default 32).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        print(f"Error: not a directory: {dataset_root}")
        return 2

    # Use the resolver to enumerate only valid, readable zarr stores.
    print("Resolving valid episodes...")
    raw = LocalEpisodeResolver._get_local_filtered_paths(
        dataset_root,
        filters=DatasetFilter(),
    )
    # raw is a list of (path_str, episode_hash)
    if not raw:
        print("No valid zarr episodes found.")
        return 2

    # Optional episode sampling
    if args.pct < 100.0:
        k = max(1, int(round(len(raw) * args.pct / 100.0)))
        rng = random.Random(args.seed)
        raw = sorted(rng.sample(raw, k))
        print(f"Sampling {k} / {len(raw)} episodes ({args.pct:.1f}%).")

    eps = [(Path(path_str), eh) for path_str, eh in raw]
    print(f"Scanning {len(eps)} episodes with {args.workers} threads...")

    # ── parallel scan ─────────────────────────────────────────────────────────
    total_episodes_with_zeros = 0
    total_zero_frames = 0
    scan_errors: list[str] = []

    # {episode_hash: {key: [bad_indices]}}
    results_with_zeros: dict[str, dict[str, list[int]]] = {}

    workers = max(1, args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_scan_episode, ep_path, eh): eh for ep_path, eh in eps}
        pbar = tqdm(
            total=len(eps),
            desc="scanning",
            unit="ep",
            dynamic_ncols=True,
        )
        for fut in concurrent.futures.as_completed(future_map):
            pbar.update(1)
            res = fut.result()
            eh = res["episode_hash"]

            if res["error"]:
                scan_errors.append(f"{eh}: {res['error']}")
                tqdm.write(f"[ERROR] {eh}: {res['error']}")
                continue

            if res["zero_rows"]:
                total_episodes_with_zeros += 1
                results_with_zeros[eh] = res["zero_rows"]
                # Count unique bad frame indices across all keys
                all_bad = set()
                for bad_idx in res["zero_rows"].values():
                    all_bad.update(bad_idx)
                n_bad = len(all_bad)
                total_zero_frames += n_bad
                keys_str = ", ".join(
                    f"{k}({len(v)})" for k, v in res["zero_rows"].items()
                )
                tqdm.write(
                    f"[ZEROS] ep={eh}  "
                    f"{n_bad} frame(s) with all-zero rows  [{keys_str}]"
                )
    pbar.close()

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Episodes scanned       : {len(eps)}")
    print(f"Scan errors            : {len(scan_errors)}")
    print(f"Episodes with zeros    : {total_episodes_with_zeros}")
    print(f"Total bad frame slots  : {total_zero_frames}")

    if results_with_zeros:
        print()
        print("Episodes with zero rows (sorted by bad-frame count):")
        sorted_eps = sorted(
            results_with_zeros.items(),
            key=lambda x: -len(set().union(*x[1].values())),
        )
        for eh, zero_rows in sorted_eps:
            all_bad = sorted(set().union(*zero_rows.values()))
            keys_str = ", ".join(zero_rows.keys())
            print(f"  {eh}")
            print(f"    keys   : {keys_str}")
            print(
                f"    frames : {len(all_bad)}  {all_bad[:20]}{'...' if len(all_bad) > 20 else ''}"
            )
    print("=" * 60)

    return 1 if (results_with_zeros or scan_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
