#!/usr/bin/env python3
"""
Benchmark Zarr data loading speed for forward pass simulation.

Uses ZarrDataset from zarr_dataset_multi.py for per-episode Zarr loading.

Measures:
1. DataLoader throughput: Shuffled DataLoader simulating training
2. Per-sample loading breakdown with profiling
"""

import argparse
import time
from pathlib import Path

import psutil
import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

# Local imports
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset, ZarrEpisode


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def safe_collate(batch: list[dict]) -> dict:
    """Collate dict batches while tolerating missing keys across samples."""
    if not batch:
        return {}
    common_keys = set(batch[0].keys())
    for item in batch[1:]:
        common_keys &= set(item.keys())
    return {key: default_collate([item[key] for item in batch]) for key in common_keys}


def _infer_batch_size(batch) -> int:
    """Best-effort batch size inference for different dataset outputs."""
    if isinstance(batch, dict):
        if "frame_index" in batch:
            return int(batch["frame_index"].shape[0])
        for value in batch.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
    if isinstance(batch, (list, tuple)):
        return len(batch)
    return 0


def benchmark_dataloader(
    dataset: Dataset,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    warmup: int,
    prefetch_factor: int = 2,
    collate_fn=None,
    simulated_compute_sec: float = 0.0,
    pytorch_profile: bool = False,
    profile_output: str | None = None,
) -> dict:
    """Benchmark DataLoader throughput with shuffling."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Calculate iterations needed
    warmup_batches = warmup
    benchmark_batches = (num_samples + batch_size - 1) // batch_size

    # Warmup
    print(f"  Warming up with {warmup_batches} batches...")
    batch_iter = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            _ = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            _ = next(batch_iter)

    # Record memory before benchmark
    memory_start_mb = get_memory_mb()

    # Benchmark
    compute_info = f", simulated_compute={simulated_compute_sec}s" if simulated_compute_sec > 0 else ""
    print(f"  Benchmarking {benchmark_batches} batches (batch_size={batch_size}, workers={num_workers}, prefetch={prefetch_factor}{compute_info})...")
    samples_processed = 0
    progress_step = max(1, benchmark_batches // 10)
    total_loading_time = 0.0
    peak_memory_mb = memory_start_mb
    memory_samples = []

    def run_benchmark_loop():
        nonlocal samples_processed, total_loading_time, batch_iter, peak_memory_mb
        for i in range(benchmark_batches):
            load_start = time.perf_counter()
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)
            total_loading_time += time.perf_counter() - load_start

            # Track memory usage
            current_memory_mb = get_memory_mb()
            peak_memory_mb = max(peak_memory_mb, current_memory_mb)
            if i % max(1, benchmark_batches // 20) == 0:  # Sample memory periodically
                memory_samples.append(current_memory_mb)

            # Count actual samples in batch (last batch may be smaller)
            batch_samples = _infer_batch_size(batch)
            samples_processed += batch_samples

            # Simulate forward/backward pass (allows prefetching to overlap)
            if simulated_compute_sec > 0:
                time.sleep(simulated_compute_sec)

            if (i + 1) % progress_step == 0 or (i + 1) == benchmark_batches:
                print(f"    Progress: {i + 1}/{benchmark_batches} ({(i + 1) / benchmark_batches:.0%})")

    total_start = time.perf_counter()
    if pytorch_profile:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            run_benchmark_loop()
        print("\n  PyTorch Profiler Summary (top 20 by CPU time):")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        if profile_output:
            prof.export_chrome_trace(profile_output)
            print(f"  Trace saved to: {profile_output}")
    else:
        run_benchmark_loop()
    total_elapsed = time.perf_counter() - total_start

    batches_per_sec = benchmark_batches / total_elapsed if total_elapsed > 0 else 0
    avg_loading_time_per_batch = total_loading_time / benchmark_batches if benchmark_batches > 0 else 0

    # Actual overhead: how much longer did training take vs pure compute?
    expected_compute_time = simulated_compute_sec * benchmark_batches
    actual_overhead = total_elapsed - expected_compute_time
    avg_overhead_per_batch = actual_overhead / benchmark_batches if benchmark_batches > 0 else 0

    # Memory statistics
    memory_end_mb = get_memory_mb()
    memory_delta_mb = memory_end_mb - memory_start_mb
    avg_memory_mb = sum(memory_samples) / len(memory_samples) if memory_samples else memory_end_mb

    return {
        "batches_per_sec": batches_per_sec,
        "samples_processed": samples_processed,
        "avg_loading_time_per_batch": avg_loading_time_per_batch,
        "avg_overhead_per_batch": avg_overhead_per_batch,
        "total_elapsed_sec": total_elapsed,
        "total_loading_time_sec": total_loading_time,
        "simulated_compute_sec": simulated_compute_sec,
        "memory_start_mb": memory_start_mb,
        "memory_end_mb": memory_end_mb,
        "peak_memory_mb": peak_memory_mb,
        "memory_delta_mb": memory_delta_mb,
        "avg_memory_mb": avg_memory_mb,
    }


def run_benchmarks(
    name: str,
    dataset: Dataset,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    warmup: int,
    prefetch_factor: int,
    collate_fn,
    simulated_compute_sec: float = 0.0,
    pytorch_profile: bool = False,
    profile_output: str | None = None,
) -> dict:
    """Run dataloader benchmark and return results for side-by-side comparison."""
    total_frames = len(dataset)
    print(f"\n== {name} ==")
    print(f"Total frames: {total_frames}")
    if total_frames == 0:
        print("Error: Dataset has no frames!")
        return {
            "name": name,
            "total_frames": 0,
            "dataloader": None,
        }

    num_samples = min(num_samples, total_frames)

    dataloader_results = benchmark_dataloader(
        dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        warmup=warmup,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        simulated_compute_sec=simulated_compute_sec,
        pytorch_profile=pytorch_profile,
        profile_output=profile_output,
    )

    return {
        "name": name,
        "total_frames": total_frames,
        "dataloader": dataloader_results,
    }


def print_results_table(results: list[dict]) -> None:
    """Print benchmark results in a table format."""
    if not results:
        return

    # Filter out results with no dataloader data
    valid_results = [r for r in results if r.get("dataloader")]
    if not valid_results:
        return

    # Build table data
    headers = ["Metric"] + [r["name"] for r in valid_results]
    rows = []

    # Total frames
    rows.append(["Total frames"] + [f"{r['total_frames']:,}" for r in valid_results])

    # Init time
    if all(r.get("init_time_sec") is not None for r in valid_results):
        rows.append(["Init time (s)"] + [f"{r['init_time_sec']:.3f}" for r in valid_results])

    # Dataloader metrics
    rows.append(["Batches/sec"] + [f"{r['dataloader']['batches_per_sec']:.1f}" for r in valid_results])
    rows.append(["Avg load time/batch (ms)"] + [f"{r['dataloader']['avg_loading_time_per_batch']*1000:.1f}" for r in valid_results])
    rows.append(["Avg overhead/batch (ms)"] + [f"{r['dataloader']['avg_overhead_per_batch']*1000:.1f}" for r in valid_results])
    rows.append(["Total time (s)"] + [f"{r['dataloader']['total_elapsed_sec']:.2f}" for r in valid_results])
    rows.append(["Samples processed"] + [f"{r['dataloader']['samples_processed']:,}" for r in valid_results])

    # Memory metrics
    rows.append(["Memory start (MB)"] + [f"{r['dataloader']['memory_start_mb']:.1f}" for r in valid_results])
    rows.append(["Memory end (MB)"] + [f"{r['dataloader']['memory_end_mb']:.1f}" for r in valid_results])
    rows.append(["Peak memory (MB)"] + [f"{r['dataloader']['peak_memory_mb']:.1f}" for r in valid_results])
    rows.append(["Memory delta (MB)"] + [f"{r['dataloader']['memory_delta_mb']:+.1f}" for r in valid_results])
    rows.append(["Avg memory (MB)"] + [f"{r['dataloader']['avg_memory_mb']:.1f}" for r in valid_results])

    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Print table
    print("\n" + "=" * (sum(col_widths) + 3 * len(col_widths) + 1))
    print("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for row in rows:
        print("| " + " | ".join(str(v).rjust(col_widths[i]) for i, v in enumerate(row)) + " |")
    print("=" * (sum(col_widths) + 3 * len(col_widths) + 1))


def build_zarr_dataset(
    root: Path,
    max_episodes: int | None = None,
    action_horizon: int | None = None,
) -> tuple[Dataset, str, int]:
    """
    Load Zarr dataset from root directory.

    Supports two structures:
    1. Single episode: root is a .zarr directory
    2. Multiple episodes: root contains multiple .zarr directories

    Args:
        root: Path to dataset root
        max_episodes: Maximum number of episodes to load (for debugging)
        action_horizon: Number of future action timesteps to load per sample

    Returns:
        Tuple of (dataset, name, num_episodes)
    """
    root = root.resolve()

    # Check if root itself is a .zarr episode
    if root.suffix == ".zarr":
        print(f"  Loading single episode: {root.name}")
        dataset = ZarrDataset(str(root), action_horizon=action_horizon)
        return dataset, f"Zarr ({root.name})", 1

    # Look for .zarr episode directories
    episode_dirs = sorted([
        path for path in root.iterdir()
        if path.is_dir() and path.suffix == ".zarr"
    ])

    if not episode_dirs:
        raise FileNotFoundError(
            f"No .zarr episode directories found in {root}"
        )

    # Limit episodes if requested
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    print(f"  Found {len(episode_dirs)} episodes")

    datasets = []
    total_frames = 0
    for i, episode_dir in enumerate(episode_dirs):
        if (i + 1) % max(1, len(episode_dirs) // 10) == 0 or i == 0:
            print(f"    Loading episode {i + 1}/{len(episode_dirs)}: {episode_dir.name}")
        ds = ZarrDataset(str(episode_dir), action_horizon=action_horizon)
        datasets.append(ds)
        total_frames += len(ds)  # Triggers init_episode during loop with progress output

    print(f"  Total frames: {total_frames:,}")
    print(f"  Creating ConcatDataset...")
    dataset = ConcatDataset(datasets)  # Now fast - lengths already cached
    return dataset, f"Zarr ({root.name}, {len(datasets)} episodes)", len(datasets)


def profile_single_episode(
    episode_path: Path,
    num_samples: int = 100,
) -> None:
    """Profile loading performance for a single episode."""
    print(f"\n== Profiling Single Episode ==")
    print(f"Episode: {episode_path.name}")

    dataset = ZarrDataset(str(episode_path))
    total_frames = len(dataset)
    num_samples = min(num_samples, total_frames)

    print(f"Total frames: {total_frames}")
    print(f"Profiling {num_samples} samples...")

    # Warmup
    for i in range(min(10, num_samples)):
        _ = dataset[i]

    # Profile
    times = []
    for i in range(num_samples):
        start = time.perf_counter()
        _ = dataset[i]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = sorted(times)
    print(f"\nLoading time per sample:")
    print(f"  Min:    {times[0]*1000:.2f}ms")
    print(f"  Median: {times[len(times)//2]*1000:.2f}ms")
    print(f"  Mean:   {sum(times)/len(times)*1000:.2f}ms")
    print(f"  P95:    {times[int(len(times)*0.95)]*1000:.2f}ms")
    print(f"  Max:    {times[-1]*1000:.2f}ms")


def main():
    # Check multiprocessing context
    import torch.multiprocessing as mp
    print(f"Multiprocessing start method: {mp.get_start_method()}")

    parser = argparse.ArgumentParser(
        description="Benchmark Zarr data loading speed for forward pass simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark single episode
  %(prog)s --zarr-path /data/episode_000000.zarr

  # Benchmark multiple episodes
  %(prog)s --zarr-path /data/zarr_episodes --max-episodes 10

  # Profile single episode
  %(prog)s --zarr-path /data/episode_000000.zarr --profile-episode

  # Benchmark with simulated compute
  %(prog)s --zarr-path /data/zarr_episodes --simulated-compute 0.05
        """
    )

    # Dataset source
    parser.add_argument(
        "--zarr-path",
        type=str,
        required=True,
        help="Path to Zarr dataset (single .zarr episode or directory containing episodes)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples to load via DataLoader (default: 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader test (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="DataLoader workers (default: 10)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor (default: 2)",
    )
    parser.add_argument(
        "--simulated-compute",
        type=float,
        default=0.0,
        help="Simulated forward/backward pass time in seconds per batch (default: 0.0). "
             "Use this to test whether data loading can keep up with GPU compute.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before timing (default: 10)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episode directories to load (for debugging)",
    )
    parser.add_argument(
        "--pytorch-profile",
        action="store_true",
        help="Use PyTorch profiler for detailed performance breakdown",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        help="Path to save PyTorch profiler trace (for Chrome trace viewer)",
    )
    parser.add_argument(
        "--profile-episode",
        action="store_true",
        help="Profile single-episode loading performance (requires single .zarr path)",
    )
    parser.add_argument(
        "--profile-samples",
        type=int,
        default=100,
        help="Number of samples to profile for single-episode profiling",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Number of future action timesteps to load per sample (enables dynamic chunking)",
    )

    args = parser.parse_args()

    # Display action chunking info if enabled
    if args.action_horizon is not None:
        print(f"NOTE: Action chunking enabled (horizon={args.action_horizon})")
        print("  - actions_base_cartesian and actions_joints will be loaded as sequences")
        print(f"  - Shape: (action_horizon={args.action_horizon}, action_dim)\n")

    print("=== Zarr Forward Pass Benchmark ===\n")

    zarr_path = Path(args.zarr_path).resolve()

    # Profile single episode if requested
    if args.profile_episode:
        if not zarr_path.suffix == ".zarr":
            parser.error("--profile-episode requires a single .zarr episode path")
        profile_single_episode(zarr_path, num_samples=args.profile_samples)
        return

    # Load dataset
    print("Loading Zarr dataset...")
    zarr_init_start = time.perf_counter()
    zarr_dataset, zarr_name, num_episodes = build_zarr_dataset(
        zarr_path,
        max_episodes=args.max_episodes,
        action_horizon=args.action_horizon,
    )
    zarr_init_time = time.perf_counter() - zarr_init_start
    print(f"  Initialization time: {zarr_init_time:.3f}s")

    # Run benchmark
    result = run_benchmarks(
        name=zarr_name,
        dataset=zarr_dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup=args.warmup,
        prefetch_factor=args.prefetch_factor,
        collate_fn=safe_collate,
        simulated_compute_sec=args.simulated_compute,
        pytorch_profile=args.pytorch_profile,
        profile_output=args.profile_output,
    )
    result["init_time_sec"] = zarr_init_time

    print_results_table([result])
    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
