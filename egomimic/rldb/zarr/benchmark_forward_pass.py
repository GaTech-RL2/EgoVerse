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
from torch.profiler import profile, ProfilerActivity
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


class DropKeysDataset(Dataset):
    """Wrapper dataset that removes keys from dict samples."""

    def __init__(self, base: Dataset, drop_predicate):
        self.base = base
        self.drop_predicate = drop_predicate

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if not isinstance(item, dict):
            return item
        return {k: v for k, v in item.items() if not self.drop_predicate(k)}


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
    compute_info = f", simulated_compute={simulated_compute_sec}s" if simulated_compute_sec > 0 else ""
    print(f"  Benchmarking {benchmark_batches} batches (batch_size={batch_size}, workers={num_workers}, prefetch={prefetch_factor}{compute_info})...")
    samples_processed = 0
    progress_step = max(1, benchmark_batches // 10)
    total_loading_time = 0.0
    profiler_result = None

    def run_benchmark_loop():
        nonlocal samples_processed, total_loading_time, batch_iter
        for i in range(benchmark_batches):
            load_start = time.perf_counter()
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)
            total_loading_time += time.perf_counter() - load_start

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
        profiler_result = prof
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

    return {
        "batches_per_sec": batches_per_sec,
        "samples_processed": samples_processed,
        "avg_loading_time_per_batch": avg_loading_time_per_batch,
        "avg_overhead_per_batch": avg_overhead_per_batch,
        "total_elapsed_sec": total_elapsed,
        "total_loading_time_sec": total_loading_time,
        "simulated_compute_sec": simulated_compute_sec,
    }


def build_lerobot_dataset(root: Path, max_episodes: int | None = None) -> tuple[Dataset, str]:
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

    # Limit episodes if requested
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

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
    headers = ["Metric"] + ["Zarr" if "Zarr" in r["name"] else "LeRobot" for r in valid_results]
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

    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Print table
    print("\n" + "=" * (sum(col_widths) + 3 * len(col_widths) + 1))
    print("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
    for row in rows:
        print("| " + " | ".join(str(v).rjust(col_widths[i]) for i, v in enumerate(row)) + " |")
    print("=" * (sum(col_widths) + 3 * len(col_widths) + 1))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Zarr vs LeRobot data loading speed for forward pass simulation"
    )

    # Dataset sources
    parser.add_argument(
        "--zarr-path",
        type=str,
        required=True,
        help="Path to local zarr dataset",
    )
    parser.add_argument(
        "--lerobot-path",
        type=str,
        default=None,
        help="Path to a local LeRobot dataset root (e.g., /path/to/dataset)",
    )
    parser.add_argument(
        "--skip-lerobot",
        action="store_true",
        help="Skip LeRobot dataset loading and benchmarking (Zarr only)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to load via DataLoader (default: 10000)",
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
    default=0.5,
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
        "--skip-images",
        action="store_true",
        help="Drop image keys from samples to avoid decode overhead",
    )
    parser.add_argument(
        "--profile-zarr",
        action="store_true",
        help="Profile Zarr __getitem__ time breakdown in-process (num_workers=0)",
    )
    parser.add_argument(
        "--profile-samples",
        type=int,
        default=100,
        help="Number of samples to profile for Zarr breakdown",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episode folders to load (for debugging)",
    )
    parser.add_argument(
        "--dynamic-chunks",
        action="store_true",
        help="Enable dynamic action chunking (load action sequences at runtime). "
             "Use this when benchmarking datasets with prestack=False.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=100,
        help="Number of future action timesteps to load when --dynamic-chunks is enabled",
    )
    parser.add_argument(
        "--action-keys",
        type=str,
        nargs="+",
        default=["actions_base_cartesian", "actions_cartesian", "actions_eef_cartesian", "actions_joints"],
        help="Action keys to apply dynamic chunking to",
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

    args = parser.parse_args()

    # Validate arguments
    if not args.skip_lerobot and args.lerobot_path is None:
        parser.error("--lerobot-path is required unless --skip-lerobot is specified")

    if args.skip_lerobot:
        print("=== Zarr Forward Pass Benchmark ===")
    else:
        print("=== Zarr vs LeRobot Forward Pass Benchmark ===")
    if args.skip_images:
        print("\nNOTE: --skip-images enabled")
        print("  - Zarr: truly skips image loading (no I/O, no decode)")
        print("  - LeRobot: still decodes video frames, then drops them (no skip_keys support)")

    if args.dynamic_chunks:
        print("\nNOTE: --dynamic-chunks enabled")
        print(f"  - Loading {args.action_horizon} future actions per sample")
        print(f"  - Action keys: {', '.join(args.action_keys)}")
        print("  - This simulates runtime action chunking for prestack=False datasets")

    zarr_path = Path(args.zarr_path).resolve()
    lerobot_path = Path(args.lerobot_path).resolve() if args.lerobot_path else None

    # Determine which keys to skip for Zarr (need to peek at metadata first)
    zarr_skip_keys: set[str] | None = None
    if args.skip_images:
        from egomimic.rldb.zarr.zarr_dataset import ZarrDatasetMetadata
        meta = ZarrDatasetMetadata(repo_id=zarr_path.name, root=zarr_path)
        zarr_skip_keys = set(meta.camera_keys)

    # Determine episodes to load if max_episodes is set
    zarr_episodes = None
    if args.max_episodes is not None:
        zarr_episodes = list(range(args.max_episodes))
        print(f"\nNOTE: --max-episodes={args.max_episodes} (loading limited episodes for debugging)")

    # Configure temporal windowing for dynamic chunks
    delta_timestamps = None
    if args.dynamic_chunks:
        # Get FPS from dataset metadata for accurate time offsets
        from egomimic.rldb.zarr.zarr_dataset import ZarrDatasetMetadata
        meta = ZarrDatasetMetadata(repo_id=zarr_path.name, root=zarr_path)
        fps = meta.fps if hasattr(meta, 'fps') else 30

        # Build delta_timestamps dict for action keys
        # Load actions from current frame (t=0) to future frames
        delta_timestamps = {}
        for action_key in args.action_keys:
            # Create time offsets: [0.0, 1/fps, 2/fps, ..., (horizon-1)/fps]
            time_offsets = [i / fps for i in range(args.action_horizon)]
            delta_timestamps[action_key] = time_offsets

        print(f"\nDynamic chunks configuration:")
        print(f"  Action horizon: {args.action_horizon} timesteps")
        print(f"  Action keys: {args.action_keys}")
        print(f"  FPS: {fps}")
        print(f"  Time offsets: {len(time_offsets)} points over {time_offsets[-1]:.2f}s")

    print("\nLoading Zarr dataset...")
    zarr_init_start = time.perf_counter()
    zarr_dataset = ZarrDataset(
        repo_id=zarr_path.name,
        root=zarr_path,
        profile=args.profile_zarr,
        skip_keys=zarr_skip_keys,
        episodes=zarr_episodes,
        delta_timestamps=delta_timestamps,
    )
    zarr_init_time = time.perf_counter() - zarr_init_start
    print(f"  Initialization time: {zarr_init_time:.3f}s")

    lerobot_dataset = None
    lerobot_name = None
    lerobot_init_time = None
    if not args.skip_lerobot:
        print("\nLoading LeRobot dataset...")
        lerobot_init_start = time.perf_counter()
        lerobot_dataset, lerobot_name = build_lerobot_dataset(lerobot_path, max_episodes=args.max_episodes)
        lerobot_init_time = time.perf_counter() - lerobot_init_start
        print(f"  Initialization time: {lerobot_init_time:.3f}s")

        if args.skip_images:
            lerobot_dataset = DropKeysDataset(
                lerobot_dataset,
                drop_predicate=lambda k: k.startswith("observation.images."),
            )

    if args.profile_zarr:
        num_profile = min(args.profile_samples, len(zarr_dataset))
        print(f"\nProfiling Zarr __getitem__ on {num_profile} samples (num_workers=0)...")
        zarr_dataset.reset_profile()
        start = time.perf_counter()
        for i in range(num_profile):
            _ = zarr_dataset[i]
        elapsed = time.perf_counter() - start
        profile = zarr_dataset.get_profile_summary()
        totals = profile.get("totals", {})
        counts = profile.get("counts", {})
        total_getitem = totals.get("getitem_total_sec", 0.0)
        print(f"  Total wall time: {elapsed:.3f}s")
        print(f"  __getitem__ total: {total_getitem:.3f}s over {counts.get('getitem_count', 0)} samples")
        if total_getitem > 0:
            def pct(val: float) -> float:
                return 100.0 * val / total_getitem

            for key in [
                "zarr_read_sec",
                "image_decode_sec",
                "image_to_tensor_sec",
                "non_image_to_tensor_sec",
                "open_store_sec",
            ]:
                if key in totals:
                    print(f"  {key}: {totals[key]:.3f}s ({pct(totals[key]):.1f}%)")

    results = []
    zarr_result = run_benchmarks(
        name=f"Zarr ({zarr_path})",
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
    zarr_result["init_time_sec"] = zarr_init_time
    results.append(zarr_result)

    if not args.skip_lerobot:
        lerobot_result = run_benchmarks(
            name=lerobot_name,
            dataset=lerobot_dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup=args.warmup,
            prefetch_factor=args.prefetch_factor,
            collate_fn=None,
            simulated_compute_sec=args.simulated_compute,
        )
        lerobot_result["init_time_sec"] = lerobot_init_time
        results.append(lerobot_result)

    print_results_table(results)
    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
