#!/usr/bin/env python3
"""
Benchmark ZarrDataset/ZarrEpisode initialization time.

Measures the time to initialize 100 ZarrDataset instances (which each create a ZarrEpisode).
"""

import argparse
import time
from pathlib import Path

from egomimic.rldb.zarr.zarr_utils import ZarrDataset, ZarrEpisode


def benchmark_episode_init(episode_paths: list[Path], num_iterations: int = 100) -> dict:
    """
    Benchmark ZarrEpisode initialization.

    Args:
        episode_paths: List of paths to .zarr episode directories
        num_iterations: Number of initialization iterations to run

    Returns:
        Dictionary with benchmark results
    """
    # Repeat episodes if we don't have enough
    paths_to_test = (episode_paths * ((num_iterations // len(episode_paths)) + 1))[:num_iterations]

    print(f"Benchmarking {num_iterations} ZarrEpisode initializations...")
    print(f"Using {len(episode_paths)} unique episode(s)")

    start_time = time.perf_counter()

    for i, path in enumerate(paths_to_test):
        episode = ZarrEpisode(path)

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / (i + 1)
            print(f"  {i + 1}/{num_iterations}: {avg_time*1000:.2f}ms avg per init")

    total_time = time.perf_counter() - start_time
    avg_time = total_time / num_iterations

    return {
        "total_time_sec": total_time,
        "avg_time_ms": avg_time * 1000,
        "iterations": num_iterations,
        "unique_episodes": len(episode_paths),
    }


def benchmark_dataset_init(episode_paths: list[Path], num_iterations: int = 100) -> dict:
    """
    Benchmark ZarrDataset initialization.

    Args:
        episode_paths: List of paths to .zarr episode directories
        num_iterations: Number of initialization iterations to run

    Returns:
        Dictionary with benchmark results
    """
    # Repeat episodes if we don't have enough
    paths_to_test = (episode_paths * ((num_iterations // len(episode_paths)) + 1))[:num_iterations]

    print(f"\nBenchmarking {num_iterations} ZarrDataset initializations...")
    print(f"Using {len(episode_paths)} unique episode(s)")

    start_time = time.perf_counter()

    for i, path in enumerate(paths_to_test):
        dataset = ZarrDataset(path)

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / (i + 1)
            print(f"  {i + 1}/{num_iterations}: {avg_time*1000:.2f}ms avg per init")

    total_time = time.perf_counter() - start_time
    avg_time = total_time / num_iterations

    return {
        "total_time_sec": total_time,
        "avg_time_ms": avg_time * 1000,
        "iterations": num_iterations,
        "unique_episodes": len(episode_paths),
    }


def find_zarr_episodes(root: Path, max_episodes: int | None = None) -> list[Path]:
    """
    Find all .zarr episode directories under root.

    Args:
        root: Root directory to search
        max_episodes: Maximum number of episodes to return (None for all)

    Returns:
        List of paths to .zarr episode directories
    """
    episodes = sorted(root.glob("episode_*.zarr"))

    if not episodes:
        raise FileNotFoundError(f"No episode_*.zarr directories found in {root}")

    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ZarrDataset/ZarrEpisode initialization time"
    )

    parser.add_argument(
        "--zarr-path",
        type=str,
        required=True,
        help="Path to directory containing .zarr episode folders",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of initialization iterations (default: 100)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of unique episodes to use (default: all)",
    )
    parser.add_argument(
        "--test",
        choices=["episode", "dataset", "both"],
        default="both",
        help="What to benchmark: 'episode' (ZarrEpisode), 'dataset' (ZarrDataset), or 'both' (default: both)",
    )

    args = parser.parse_args()

    zarr_path = Path(args.zarr_path).resolve()

    print("=" * 70)
    print("ZarrDataset/ZarrEpisode Initialization Benchmark")
    print("=" * 70)
    print(f"Zarr path: {zarr_path}")
    print(f"Iterations: {args.num_iterations}")

    # Find episode directories
    episode_paths = find_zarr_episodes(zarr_path, max_episodes=args.max_episodes)
    print(f"Found {len(episode_paths)} episode(s)")

    results = {}

    # Benchmark ZarrDataset FIRST to test cache effects
    if args.test in ["dataset", "both"]:
        results["dataset"] = benchmark_dataset_init(episode_paths, args.num_iterations)

    # Benchmark ZarrEpisode SECOND
    if args.test in ["episode", "both"]:
        results["episode"] = benchmark_episode_init(episode_paths, args.num_iterations)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if "episode" in results:
        r = results["episode"]
        print(f"\nZarrEpisode Initialization:")
        print(f"  Total time: {r['total_time_sec']:.3f}s")
        print(f"  Avg per init: {r['avg_time_ms']:.2f}ms")
        print(f"  Iterations: {r['iterations']}")
        print(f"  Unique episodes: {r['unique_episodes']}")

    if "dataset" in results:
        r = results["dataset"]
        print(f"\nZarrDataset Initialization:")
        print(f"  Total time: {r['total_time_sec']:.3f}s")
        print(f"  Avg per init: {r['avg_time_ms']:.2f}ms")
        print(f"  Iterations: {r['iterations']}")
        print(f"  Unique episodes: {r['unique_episodes']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
