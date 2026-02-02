#!/usr/bin/env python3
"""
Zarr Chunk Size Benchmark Script

Benchmarks zarr dataset loading performance across different chunk sizes and action horizons.
"""

import argparse
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_benchmark(
    benchmark_script: Path,
    zarr_path: Path,
    action_horizon: int,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    warmup: int,
    simulated_compute: int,
) -> dict:
    """Run benchmark_forward_pass.py and parse the results."""
    cmd = [
        "python",
        str(benchmark_script),
        "--zarr-path",
        str(zarr_path),
        "--skip-lerobot",
        "--dynamic-chunks",
        "--action-horizon",
        str(action_horizon),
        "--num-samples",
        str(num_samples),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--warmup",
        str(warmup),
        "--simulated-compute",
        str(simulated_compute),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    metrics = {}

    # Parse batches/sec
    match = re.search(r"Batches/sec\s*\|\s*([\d.]+)", output)
    if match:
        metrics["batches_per_sec"] = float(match.group(1))

    # Parse avg load time per batch (ms)
    match = re.search(r"Avg load time/batch \(ms\)\s*\|\s*([\d.]+)", output)
    if match:
        metrics["avg_load_time_ms"] = float(match.group(1))

    # Parse avg overhead per batch (ms)
    match = re.search(r"Avg overhead/batch \(ms\)\s*\|\s*([\d.]+)", output)
    if match:
        metrics["avg_overhead_ms"] = float(match.group(1))

    # Parse total time (s)
    match = re.search(r"Total time \(s\)\s*\|\s*([\d.]+)", output)
    if match:
        metrics["total_time_sec"] = float(match.group(1))

    # Parse samples processed
    match = re.search(r"Samples processed\s*\|\s*([\d,]+)", output)
    if match:
        metrics["samples_processed"] = int(match.group(1).replace(",", ""))

    return metrics


def create_heatmap(df: pd.DataFrame, output_path: Path, chunk_sizes: list, action_horizons: list):
    """Create heatmap of batches/sec."""
    pivot = df.pivot(index="chunk_size", columns="action_horizon", values="batches_per_sec")

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    # Set labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Action Horizon (timesteps)")
    ax.set_ylabel("Chunk Size (timesteps)")
    ax.set_title("DataLoader Throughput (batches/sec)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Batches/sec")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < pivot.values.max() * 0.5 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def create_line_plots(df: pd.DataFrame, output_path: Path, chunk_sizes: list):
    """Create line plots for throughput and load time vs action horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Batches/sec vs action horizon
    ax1 = axes[0]
    for chunk_size in chunk_sizes:
        subset = df[df["chunk_size"] == chunk_size]
        if not subset.empty:
            ax1.plot(
                subset["action_horizon"],
                subset["batches_per_sec"],
                marker="o",
                label=f"{chunk_size}ts",
            )

    ax1.set_xlabel("Action Horizon (timesteps)")
    ax1.set_ylabel("Batches/sec")
    ax1.set_title("Throughput vs Action Horizon")
    ax1.legend(title="Chunk Size", loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Avg load time vs action horizon
    ax2 = axes[1]
    for chunk_size in chunk_sizes:
        subset = df[df["chunk_size"] == chunk_size]
        if not subset.empty:
            ax2.plot(
                subset["action_horizon"],
                subset["avg_load_time_ms"],
                marker="o",
                label=f"{chunk_size}ts",
            )

    ax2.set_xlabel("Action Horizon (timesteps)")
    ax2.set_ylabel("Avg Load Time per Batch (ms)")
    ax2.set_title("Load Time vs Action Horizon")
    ax2.legend(title="Chunk Size", loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved line plots to: {output_path}")


def print_summary(df: pd.DataFrame, action_horizons: list):
    """Print summary statistics."""
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Best overall configuration
    best_idx = df["batches_per_sec"].idxmax()
    best = df.loc[best_idx]
    print(f"\nBest overall configuration:")
    print(f"  Chunk size: {int(best['chunk_size'])} timesteps")
    print(f"  Action horizon: {int(best['action_horizon'])} timesteps")
    print(f"  Batches/sec: {best['batches_per_sec']:.1f}")
    print(f"  Avg load time: {best['avg_load_time_ms']:.1f} ms")

    # Worst overall configuration
    worst_idx = df["batches_per_sec"].idxmin()
    worst = df.loc[worst_idx]
    print(f"\nWorst overall configuration:")
    print(f"  Chunk size: {int(worst['chunk_size'])} timesteps")
    print(f"  Action horizon: {int(worst['action_horizon'])} timesteps")
    print(f"  Batches/sec: {worst['batches_per_sec']:.1f}")
    print(f"  Avg load time: {worst['avg_load_time_ms']:.1f} ms")

    # Speedup
    speedup = best["batches_per_sec"] / worst["batches_per_sec"]
    print(f"\nSpeedup (best vs worst): {speedup:.1f}x")

    # Best chunk size for each action horizon
    print(f"\nBest chunk size for each action horizon:")
    for ah in action_horizons:
        subset = df[df["action_horizon"] == ah]
        if not subset.empty:
            best_for_ah = subset.loc[subset["batches_per_sec"].idxmax()]
            print(
                f"  Action horizon {ah:>3}: chunk_size={int(best_for_ah['chunk_size']):>4}ts "
                f"-> {best_for_ah['batches_per_sec']:.1f} batches/sec"
            )


def next_available_path(path: Path) -> Path:
    """Return the next available path (appending _1, _2, etc. if exists)."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def main():
    parser = argparse.ArgumentParser(description="Benchmark zarr chunk sizes")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("/coc/flash7/rco3/EgoVerse/egomimic/rldb/zarr/zarr/dynamic"),
        help="Base path containing zarr datasets",
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path("/coc/flash7/rco3/EgoVerse/egomimic/rldb/zarr/benchmark_forward_pass.py"),
        help="Path to benchmark_forward_pass.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: same as base-path)",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[1, 100, 250, 500],
        help="Chunk sizes to test",
    )
    parser.add_argument(
        "--action-horizons",
        type=int,
        nargs="+",
        default=[1, 100, 250, 500],
        help="Action horizons to test",
    )
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup batches")
    parser.add_argument(
        "--simulated-compute",
        type=int,
        default=0,
        help="Simulated compute time in ms (0 for pure loading speed)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.base_path

    # Run all benchmarks
    results = []
    total_runs = len(args.chunk_sizes) * len(args.action_horizons)
    current_run = 0

    for chunk_size in args.chunk_sizes:
        zarr_path = args.base_path / f"{chunk_size}ts"

        if not zarr_path.exists():
            print(f"Warning: {zarr_path} does not exist, skipping...")
            continue

        for action_horizon in args.action_horizons:
            current_run += 1
            print(
                f"[{current_run}/{total_runs}] Running benchmark: "
                f"chunk_size={chunk_size}, action_horizon={action_horizon}"
            )

            metrics = run_benchmark(
                benchmark_script=args.benchmark_script,
                zarr_path=zarr_path,
                action_horizon=action_horizon,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                warmup=args.warmup,
                simulated_compute=args.simulated_compute,
            )

            if metrics:
                metrics["chunk_size"] = chunk_size
                metrics["action_horizon"] = action_horizon
                results.append(metrics)
                print(f"  -> batches/sec: {metrics.get('batches_per_sec', 'N/A'):.1f}")
            else:
                print(f"  -> Failed to parse results")

    print(f"\nCompleted {len(results)} benchmark runs.")

    if not results:
        print("No results to process. Exiting.")
        return

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(["chunk_size", "action_horizon"])
    print(f"\nResults DataFrame shape: {df.shape}")
    print(df.to_string())

    # Create and save figures
    heatmap_path = next_available_path(output_dir / "benchmark_heatmap.png")
    create_heatmap(df, heatmap_path, args.chunk_sizes, args.action_horizons)

    lineplot_path = next_available_path(output_dir / "benchmark_lineplots.png")
    create_line_plots(df, lineplot_path, args.chunk_sizes)

    # Print summary
    print_summary(df, args.action_horizons)

    # Save CSV
    csv_path = next_available_path(output_dir / "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
