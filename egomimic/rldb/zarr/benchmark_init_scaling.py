#!/usr/bin/env python3
"""
Benchmark Zarr initialization time scaling with number of episodes.

Directly measures initialization overhead to understand memory and time
costs per episode without running full training benchmarks.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
from torch.utils.data import ConcatDataset

from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_init(
    zarr_path: Path,
    max_episodes: int,
    action_horizon: int | None = None,
) -> dict:
    """
    Benchmark initialization time and memory for loading N episodes.

    Args:
        zarr_path: Path to directory containing .zarr episodes
        max_episodes: Number of episodes to load
        action_horizon: Optional action chunking horizon

    Returns:
        dict with metrics
    """
    # Find episode directories
    episode_dirs = sorted([
        path for path in zarr_path.iterdir()
        if path.is_dir() and path.suffix == ".zarr"
    ])

    if not episode_dirs:
        raise FileNotFoundError(f"No .zarr episodes found in {zarr_path}")

    episode_dirs = episode_dirs[:max_episodes]

    print(f"\n  Loading {len(episode_dirs)} episodes...")

    # Measure memory before
    memory_start = get_memory_mb()

    # Time the initialization
    start = time.perf_counter()

    datasets = []
    total_frames = 0
    for i, episode_dir in enumerate(episode_dirs):
        ds = ZarrDataset(str(episode_dir), action_horizon=action_horizon)
        datasets.append(ds)
        total_frames += len(ds)  # Triggers init_episode

        # Progress every 10%
        if (i + 1) % max(1, len(episode_dirs) // 10) == 0:
            print(f"    Progress: {i + 1}/{len(episode_dirs)} ({(i + 1) / len(episode_dirs):.0%})")

    dataset = ConcatDataset(datasets)

    init_time = time.perf_counter() - start

    # Measure memory after
    memory_end = get_memory_mb()
    memory_delta = memory_end - memory_start

    print(f"  Init time: {init_time:.3f}s")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Memory delta: {memory_delta:+.1f} MB")

    return {
        "num_episodes": len(episode_dirs),
        "init_time_sec": init_time,
        "total_frames": total_frames,
        "memory_start_mb": memory_start,
        "memory_end_mb": memory_end,
        "memory_delta_mb": memory_delta,
        "time_per_episode_ms": (init_time / len(episode_dirs)) * 1000,
        "memory_per_episode_mb": memory_delta / len(episode_dirs) if len(episode_dirs) > 0 else 0,
    }


def plot_results(results: list[dict], output_path: str = "init_scaling.png"):
    """Plot initialization time and memory scaling."""

    if not results:
        print("No results to plot!")
        return

    # Ensure output path is absolute and in current working directory
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    num_episodes = np.array([r["num_episodes"] for r in results])
    init_times = np.array([r["init_time_sec"] for r in results])
    memory_end = np.array([r["memory_end_mb"] for r in results])

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Initialization time
    ax1.plot(num_episodes, init_times, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel("Number of Episodes", fontsize=12)
    ax1.set_ylabel("Initialization Time (seconds)", fontsize=12)
    ax1.set_title("Initialization Time vs Episodes", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory usage (should be roughly constant)
    ax2.plot(num_episodes, memory_end, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel("Number of Episodes", fontsize=12)
    ax2.set_ylabel("Memory Usage (MB)", fontsize=12)
    ax2.set_title("Memory Usage vs Episodes", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path.resolve()}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Episodes tested: {num_episodes.tolist()}")
    print(f"\nTime:")
    print(f"  Init times (s): {[f'{t:.2f}' for t in init_times]}")
    time_per_episode_ms = (init_times / num_episodes) * 1000
    print(f"  Mean time per episode: {np.mean(time_per_episode_ms):.2f}ms")
    print(f"  Std time per episode: {np.std(time_per_episode_ms):.2f}ms")

    print(f"\nMemory:")
    print(f"  Memory usage (MB): {[f'{m:.1f}' for m in memory_end]}")
    print(f"  Mean memory usage: {np.mean(memory_end):.1f}MB")
    print(f"  Std memory usage: {np.std(memory_end):.1f}MB")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Zarr initialization time scaling with episode count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small episode counts
  %(prog)s --zarr-path /data/zarr_episodes --episode-counts 1 10 50 100

  # Full scaling test
  %(prog)s --zarr-path /data/zarr_episodes --episode-counts 1 10 100 500 1000 5000

  # Test with action chunking enabled
  %(prog)s --zarr-path /data/zarr_episodes --action-horizon 10
        """,
    )

    parser.add_argument(
        "--zarr-path",
        type=str,
        required=True,
        help="Path to directory containing .zarr episodes",
    )
    parser.add_argument(
        "--episode-counts",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500, 1000, 5000, 9990],
        help="List of episode counts to test (default: 1 10 50 100 500 1000)",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Action chunking horizon (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="init_scaling.png",
        help="Output plot filename (default: init_scaling.png)",
    )

    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"ERROR: Zarr path does not exist: {zarr_path}")
        return 1

    if not zarr_path.is_dir():
        print(f"ERROR: Zarr path is not a directory: {zarr_path}")
        return 1

    print("=== Zarr Initialization Scaling Benchmark ===")
    print(f"Dataset: {zarr_path}")
    print(f"Testing episode counts: {args.episode_counts}")
    if args.action_horizon:
        print(f"Action horizon: {args.action_horizon}")

    # Run benchmarks
    results = []
    for max_episodes in sorted(args.episode_counts):
        print(f"\n== Benchmark: {max_episodes} episodes ==")
        try:
            result = benchmark_init(
                zarr_path,
                max_episodes=max_episodes,
                action_horizon=args.action_horizon,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\nERROR: No successful benchmark runs!")
        return 1

    # Plot results
    plot_results(results, output_path=args.output)

    print("\n=== Benchmark Complete ===")
    return 0


if __name__ == "__main__":
    exit(main())
