#!/usr/bin/env python3
"""
Benchmark Zarr vs LeRobot data loading speed for forward pass simulation.

Measures:
1. DataLoader throughput: Shuffled DataLoader simulating training
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

# Local imports
from egomimic.rldb.zarr.zarr_dataset import ZarrDataset


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
) -> dict:
    """Benchmark DataLoader throughput with shuffling."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
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

    # Benchmark
    print(f"  Benchmarking {benchmark_batches} batches (batch_size={batch_size}, workers={num_workers}, prefetch={prefetch_factor})...")
    samples_processed = 0
    start = time.perf_counter()
    progress_step = max(1, benchmark_batches // 10)

    for i in range(benchmark_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        # Count actual samples in batch (last batch may be smaller)
        batch_samples = _infer_batch_size(batch)
        samples_processed += batch_samples
        if (i + 1) % progress_step == 0 or (i + 1) == benchmark_batches:
            print(f"    Progress: {i + 1}/{benchmark_batches} ({(i + 1) / benchmark_batches:.0%})")

    end = time.perf_counter()
    total_time = end - start

    throughput = samples_processed / total_time if total_time > 0 else 0
    batches_per_sec = benchmark_batches / total_time if total_time > 0 else 0

    return {
        "throughput_samples_sec": throughput,
        "batches_per_sec": batches_per_sec,
        "samples_processed": samples_processed,
        "total_time_sec": total_time,
    }


def build_lerobot_dataset(root: Path) -> tuple[Dataset, str]:
    """Load a LeRobot dataset from a root, supporting per-episode subdirs."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = root.resolve()
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        dataset = LeRobotDataset(
            repo_id=root.name,
            root=root,
            local_files_only=True,
        )
        return dataset, f"LeRobot ({root})"

    episode_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "meta" / "info.json").exists()
    ]
    if not episode_dirs:
        raise FileNotFoundError(
            f"Could not find meta/info.json in {root} or any immediate subdirectories."
        )

    datasets = []
    for episode_dir in episode_dirs:
        datasets.append(
            LeRobotDataset(
                repo_id=f"{root.name}/{episode_dir.name}",
                root=episode_dir,
                local_files_only=True,
            )
        )
    return ConcatDataset(datasets), f"LeRobot ({root}/*, {len(datasets)} episodes)"


def run_benchmarks(
    name: str,
    dataset: Dataset,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    warmup: int,
    prefetch_factor: int,
    collate_fn,
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

    print(f"\nDataLoader (batch={batch_size}, workers={num_workers}, prefetch={prefetch_factor}):")
    dataloader_results = benchmark_dataloader(
        dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        warmup=warmup,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )
    print(f"  Throughput: {dataloader_results['throughput_samples_sec']:.1f} samples/sec")
    print(f"  Batches/sec: {dataloader_results['batches_per_sec']:.1f}")
    print(
        f"  Total time: {dataloader_results['total_time_sec']:.2f}s "
        f"for {dataloader_results['samples_processed']} samples"
    )

    return {
        "name": name,
        "total_frames": total_frames,
        "dataloader": dataloader_results,
    }


def print_side_by_side(results: list[dict]) -> None:
    """Print a compact side-by-side summary for two datasets."""
    if len(results) != 2:
        return
    left, right = results
    print("\n=== Side-by-side summary ===")
    print(f"Left:  {left['name']}")
    print(f"Right: {right['name']}")
    print("")
    print(f"Total frames: {left['total_frames']} vs {right['total_frames']}")

    if left["dataloader"] and right["dataloader"]:
        print("DataLoader:")
        print(
            "  Throughput (samples/sec): "
            f"{left['dataloader']['throughput_samples_sec']:.1f} vs "
            f"{right['dataloader']['throughput_samples_sec']:.1f}"
        )
        print(
            "  Batches/sec: "
            f"{left['dataloader']['batches_per_sec']:.1f} vs {right['dataloader']['batches_per_sec']:.1f}"
        )
        print(
            "  Total time (s): "
            f"{left['dataloader']['total_time_sec']:.2f} vs {right['dataloader']['total_time_sec']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Zarr vs LeRobot data loading speed for forward pass simulation"
    )

    # Dataset sources (both required for side-by-side)
    parser.add_argument(
        "--zarr-path",
        type=str,
        required=True,
        help="Path to local zarr dataset",
    )
    parser.add_argument(
        "--lerobot-path",
        type=str,
        required=True,
        help="Path to a local LeRobot dataset root (e.g., /path/to/dataset)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to load via DataLoader (default: 1000)",
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
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor (default: 2)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before timing (default: 10)",
    )

    args = parser.parse_args()

    print("=== Zarr vs LeRobot Forward Pass Benchmark ===")

    zarr_path = Path(args.zarr_path).resolve()
    lerobot_path = Path(args.lerobot_path).resolve()

    print("\nLoading Zarr dataset...")
    zarr_dataset = ZarrDataset(
        repo_id=zarr_path.name,
        root=zarr_path,
    )

    print("\nLoading LeRobot dataset...")
    lerobot_dataset, lerobot_name = build_lerobot_dataset(lerobot_path)

    results = []
    results.append(
        run_benchmarks(
            name=f"Zarr ({zarr_path})",
            dataset=zarr_dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup=args.warmup,
            prefetch_factor=args.prefetch_factor,
            collate_fn=safe_collate,
        )
    )
    results.append(
        run_benchmarks(
            name=lerobot_name,
            dataset=lerobot_dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup=args.warmup,
            prefetch_factor=args.prefetch_factor,
            collate_fn=None,
        )
    )

    print_side_by_side(results)

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
