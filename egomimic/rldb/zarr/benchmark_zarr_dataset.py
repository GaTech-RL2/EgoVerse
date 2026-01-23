"""
Benchmark script for ZarrDataset training performance.

Tests dataset loading, DataLoader throughput, and delta_timestamps overhead.
Supports both mock datasets and real Zarr datasets.

Usage:
    # With mock dataset (default)
    python benchmark_zarr_dataset.py

    # With custom parameters
    python benchmark_zarr_dataset.py --num-episodes 20 --batch-size 128

    # With real dataset
    python benchmark_zarr_dataset.py --dataset-path /path/to/zarr/dataset
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import zarr

from egomimic.rldb.zarr import ZarrDataset, ZarrDatasetMetadata, RLDBZarrDataset


# =============================================================================
# Utilities
# =============================================================================


def monitor_memory() -> float:
    """Monitor memory usage of the current process in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def create_benchmark_dataset(
    root: Path,
    num_episodes: int = 10,
    frames_per_episode: int = 100,
    fps: int = 30,
    action_dim: int = 7,
    state_dim: int = 14,
) -> Path:
    """
    Create a mock Zarr dataset for benchmarking.

    Args:
        root: Root directory for the dataset
        num_episodes: Number of episodes to create
        frames_per_episode: Number of frames per episode
        fps: Frames per second
        action_dim: Action vector dimension
        state_dim: State vector dimension

    Returns:
        Path to the created dataset
    """
    root = Path(root)

    # Create directory structure
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

    total_frames = num_episodes * frames_per_episode

    # Create info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": "EVA_BIMANUAL",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": list(range(num_episodes - 1)),
            "valid": [num_episodes - 1],
        },
        "data_path": "data/episode_{episode_index:06d}/chunk-{chunk_idx:03d}.zarr",
        "video_path": None,
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [f"a{i}" for i in range(action_dim)],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [f"s{i}" for i in range(state_dim)],
            },
        },
    }

    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Create stats.json (empty)
    with open(root / "meta" / "stats.json", "w") as f:
        json.dump({}, f)

    # Create tasks.jsonl
    with open(root / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "Benchmark task"}) + "\n")

    # Create episodes.jsonl
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for ep_idx in range(num_episodes):
            ep_data = {
                "episode_index": ep_idx,
                "tasks": ["Benchmark task"],
                "length": frames_per_episode,
            }
            f.write(json.dumps(ep_data) + "\n")

    # Create episode data
    global_idx = 0
    for ep_idx in range(num_episodes):
        ep_dir = root / "data" / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Create episode meta
        (ep_dir / "meta").mkdir(exist_ok=True)
        with open(ep_dir / "meta" / "episode_info.json", "w") as f:
            json.dump({"episode_index": ep_idx, "length": frames_per_episode}, f)

        # Create chunk-000.zarr
        chunk_path = ep_dir / "chunk-000.zarr"
        store = zarr.open(str(chunk_path), mode="w")

        # Generate mock data
        timestamps = np.arange(frames_per_episode, dtype=np.float32) / fps
        frame_indices = np.arange(frames_per_episode, dtype=np.int64)
        episode_indices = np.full(frames_per_episode, ep_idx, dtype=np.int64)
        indices = np.arange(global_idx, global_idx + frames_per_episode, dtype=np.int64)
        task_indices = np.zeros(frames_per_episode, dtype=np.int64)
        actions = np.random.randn(frames_per_episode, action_dim).astype(np.float32)
        states = np.random.randn(frames_per_episode, state_dim).astype(np.float32)

        # Write arrays (compatible with zarr v2 and v3)
        store.create_dataset("timestamp", shape=timestamps.shape, dtype=timestamps.dtype, data=timestamps)
        store.create_dataset("frame_index", shape=frame_indices.shape, dtype=frame_indices.dtype, data=frame_indices)
        store.create_dataset("episode_index", shape=episode_indices.shape, dtype=episode_indices.dtype, data=episode_indices)
        store.create_dataset("index", shape=indices.shape, dtype=indices.dtype, data=indices)
        store.create_dataset("task_index", shape=task_indices.shape, dtype=task_indices.dtype, data=task_indices)
        store.create_dataset("action", shape=actions.shape, dtype=actions.dtype, data=actions)
        store.create_dataset("observation.state", shape=states.shape, dtype=states.dtype, data=states)

        global_idx += frames_per_episode

    return root


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_dataset_loading(root: Path, is_real_dataset: bool = False) -> dict:
    """
    Benchmark dataset loading time and memory.

    Args:
        root: Path to dataset
        is_real_dataset: Whether this is a real dataset (affects repo_id)

    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 60)
    print("Benchmark: Dataset Loading")
    print("=" * 60)

    results = {}
    repo_id = "benchmark" if not is_real_dataset else root.name

    # Benchmark ZarrDataset
    mem_before = monitor_memory()
    with Timer("ZarrDataset") as t:
        dataset = ZarrDataset(repo_id=repo_id, root=root)
    mem_after = monitor_memory()

    results["zarr_dataset"] = {
        "load_time_s": t.elapsed,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_delta_mb": mem_after - mem_before,
        "num_frames": len(dataset),
        "num_episodes": dataset.num_episodes,
    }

    print(f"\nZarrDataset:")
    print(f"  Load time: {t.elapsed:.3f}s")
    print(f"  Memory: {mem_before:.1f} -> {mem_after:.1f} MB (+{mem_after - mem_before:.1f} MB)")
    print(f"  Frames: {len(dataset)}, Episodes: {dataset.num_episodes}")

    # Benchmark RLDBZarrDataset
    mem_before = monitor_memory()
    with Timer("RLDBZarrDataset") as t:
        rldb_dataset = RLDBZarrDataset(repo_id=repo_id, root=root, mode="train")
    mem_after = monitor_memory()

    results["rldb_zarr_dataset"] = {
        "load_time_s": t.elapsed,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_delta_mb": mem_after - mem_before,
        "num_frames": len(rldb_dataset),
        "num_episodes": rldb_dataset.num_episodes,
    }

    print(f"\nRLDBZarrDataset (mode='train'):")
    print(f"  Load time: {t.elapsed:.3f}s")
    print(f"  Memory: {mem_before:.1f} -> {mem_after:.1f} MB (+{mem_after - mem_before:.1f} MB)")
    print(f"  Frames: {len(rldb_dataset)}, Episodes: {rldb_dataset.num_episodes}")

    return results


def benchmark_delta_timestamps(root: Path, chunk_size: int = 64) -> dict:
    """
    Benchmark delta_timestamps overhead (ACT-style config).

    Args:
        root: Path to dataset
        chunk_size: Number of future action frames to load

    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 60)
    print(f"Benchmark: Delta Timestamps (ACT-style, chunk_size={chunk_size})")
    print("=" * 60)

    results = {}
    repo_id = "benchmark"

    # Load dataset without delta_timestamps (baseline)
    dataset_baseline = ZarrDataset(repo_id=repo_id, root=root)
    fps = dataset_baseline.fps

    # Load dataset with delta_timestamps
    delta_timestamps = {
        "action": [i / fps for i in range(chunk_size)],
    }

    mem_before = monitor_memory()
    with Timer("with_delta") as t:
        dataset_delta = ZarrDataset(
            repo_id=repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
        )
    mem_after = monitor_memory()

    results["load_with_delta"] = {
        "load_time_s": t.elapsed,
        "memory_delta_mb": mem_after - mem_before,
    }

    print(f"\nDataset with delta_timestamps:")
    print(f"  Load time: {t.elapsed:.3f}s")
    print(f"  Memory delta: +{mem_after - mem_before:.1f} MB")

    # Benchmark single item access
    num_samples = min(100, len(dataset_baseline))

    # Baseline access
    with Timer("baseline_access") as t:
        for i in range(num_samples):
            _ = dataset_baseline[i]
    baseline_per_item = t.elapsed / num_samples

    # Delta access
    with Timer("delta_access") as t:
        for i in range(num_samples):
            _ = dataset_delta[i]
    delta_per_item = t.elapsed / num_samples

    results["item_access"] = {
        "baseline_per_item_ms": baseline_per_item * 1000,
        "delta_per_item_ms": delta_per_item * 1000,
        "overhead_ms": (delta_per_item - baseline_per_item) * 1000,
        "overhead_ratio": delta_per_item / baseline_per_item if baseline_per_item > 0 else 0,
    }

    print(f"\nItem access time ({num_samples} samples):")
    print(f"  Baseline: {baseline_per_item * 1000:.3f} ms/item")
    print(f"  With delta: {delta_per_item * 1000:.3f} ms/item")
    print(f"  Overhead: {(delta_per_item - baseline_per_item) * 1000:.3f} ms ({results['item_access']['overhead_ratio']:.2f}x)")

    # Check output shape
    sample = dataset_delta[0]
    if "action" in sample:
        action_shape = sample["action"].shape
        print(f"\nAction shape with delta_timestamps: {action_shape}")
        results["action_shape"] = list(action_shape)

    return results


def benchmark_dataloader(
    root: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    max_batches: int = 100,
) -> dict:
    """
    Benchmark PyTorch DataLoader throughput.

    Args:
        root: Path to dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        max_batches: Maximum batches to iterate

    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 60)
    print(f"Benchmark: DataLoader (batch_size={batch_size}, num_workers={num_workers})")
    print("=" * 60)

    results = {}
    repo_id = "benchmark"

    dataset = ZarrDataset(repo_id=repo_id, root=root)
    total_frames = len(dataset)

    # Adjust max_batches if dataset is small
    actual_max_batches = min(max_batches, total_frames // batch_size)
    if actual_max_batches < 1:
        actual_max_batches = 1
        batch_size = min(batch_size, total_frames)

    print(f"\nDataset: {total_frames} frames")
    print(f"Running {actual_max_batches} batches...")

    mem_before = monitor_memory()
    with Timer("dataloader_init") as t:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
    mem_after_init = monitor_memory()

    results["init"] = {
        "time_s": t.elapsed,
        "memory_delta_mb": mem_after_init - mem_before,
    }

    print(f"\nDataLoader init: {t.elapsed:.3f}s, +{mem_after_init - mem_before:.1f} MB")

    # Iterate through batches
    batch_times = []
    batch_memories = []

    with Timer("total_iteration") as t_total:
        for i, batch in enumerate(dataloader):
            if i >= actual_max_batches:
                break

            t_start = time.perf_counter()
            # Simulate minimal processing (move to CPU is already done)
            _ = batch["action"].shape
            t_end = time.perf_counter()

            batch_times.append(t_end - t_start)
            batch_memories.append(monitor_memory())

            if i == 0:
                print(f"  First batch shapes:")
                for key in ["action", "observation.state", "episode_index"]:
                    if key in batch:
                        print(f"    {key}: {batch[key].shape}")

    total_samples = actual_max_batches * batch_size
    throughput = total_samples / t_total.elapsed if t_total.elapsed > 0 else 0

    results["iteration"] = {
        "total_time_s": t_total.elapsed,
        "batches": actual_max_batches,
        "samples": total_samples,
        "throughput_samples_per_s": throughput,
        "avg_batch_time_ms": np.mean(batch_times) * 1000 if batch_times else 0,
        "std_batch_time_ms": np.std(batch_times) * 1000 if batch_times else 0,
        "peak_memory_mb": max(batch_memories) if batch_memories else 0,
        "avg_memory_mb": np.mean(batch_memories) if batch_memories else 0,
    }

    print(f"\nIteration results:")
    print(f"  Total time: {t_total.elapsed:.3f}s for {actual_max_batches} batches")
    print(f"  Throughput: {throughput:.1f} samples/s")
    print(f"  Avg batch time: {results['iteration']['avg_batch_time_ms']:.3f} +/- {results['iteration']['std_batch_time_ms']:.3f} ms")
    print(f"  Peak memory: {results['iteration']['peak_memory_mb']:.1f} MB")

    return results


def benchmark_rldb_modes(root: Path) -> dict:
    """
    Benchmark different RLDBZarrDataset modes.

    Args:
        root: Path to dataset

    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 60)
    print("Benchmark: RLDB Dataset Modes")
    print("=" * 60)

    results = {}
    repo_id = "benchmark"

    modes = [
        ("train", {}),
        ("valid", {}),
        ("percent", {"percent": 0.5}),
    ]

    for mode, kwargs in modes:
        try:
            with Timer(mode) as t:
                dataset = RLDBZarrDataset(
                    repo_id=repo_id,
                    root=root,
                    mode=mode,
                    **kwargs,
                )
            results[mode] = {
                "load_time_s": t.elapsed,
                "num_frames": len(dataset),
                "num_episodes": dataset.num_episodes,
            }
            print(f"\n{mode}: {len(dataset)} frames, {dataset.num_episodes} episodes, {t.elapsed:.3f}s")
        except Exception as e:
            print(f"\n{mode}: Error - {e}")
            results[mode] = {"error": str(e)}

    return results


# =============================================================================
# Main
# =============================================================================


def run_all_benchmarks(args) -> dict:
    """Run all benchmarks and return results."""
    all_results = {
        "config": {
            "num_episodes": args.num_episodes,
            "frames_per_episode": args.frames_per_episode,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_batches": args.max_batches,
            "dataset_path": args.dataset_path,
        }
    }

    if args.dataset_path:
        # Use real dataset
        root = Path(args.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")
        print(f"\nUsing real dataset: {root}")
        is_real = True
        cleanup = False
    else:
        # Create mock dataset
        tmpdir = tempfile.mkdtemp(prefix="zarr_benchmark_")
        root = create_benchmark_dataset(
            Path(tmpdir) / "benchmark_dataset",
            num_episodes=args.num_episodes,
            frames_per_episode=args.frames_per_episode,
        )
        print(f"\nCreated mock dataset: {root}")
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Frames per episode: {args.frames_per_episode}")
        print(f"  Total frames: {args.num_episodes * args.frames_per_episode}")
        is_real = False
        cleanup = True

    try:
        # Run benchmarks
        all_results["dataset_loading"] = benchmark_dataset_loading(root, is_real)
        all_results["delta_timestamps"] = benchmark_delta_timestamps(root)
        all_results["dataloader"] = benchmark_dataloader(
            root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
        )
        all_results["rldb_modes"] = benchmark_rldb_modes(root)

    finally:
        if cleanup:
            import shutil
            shutil.rmtree(Path(root).parent, ignore_errors=True)
            print(f"\nCleaned up temporary dataset")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset load time: {all_results['dataset_loading']['zarr_dataset']['load_time_s']:.3f}s")
    print(f"Delta timestamps overhead: {all_results['delta_timestamps']['item_access']['overhead_ratio']:.2f}x")
    print(f"DataLoader throughput: {all_results['dataloader']['iteration']['throughput_samples_per_s']:.1f} samples/s")

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ZarrDataset for training performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to real Zarr dataset (if not provided, creates mock dataset)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes for mock dataset",
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=100,
        help="Frames per episode for mock dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for DataLoader benchmark",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum batches to iterate",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output results to JSON file",
    )

    args = parser.parse_args()
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
