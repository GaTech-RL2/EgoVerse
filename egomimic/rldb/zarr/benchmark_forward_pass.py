#!/usr/bin/env python3
"""
Benchmark Zarr data loading speed for forward pass simulation.

Measures:
1. Random single-frame access: Direct dataset[random_idx] calls
2. DataLoader throughput: Shuffled DataLoader simulating training

Supports both local filesystem and S3 backends.
"""

import argparse
import random
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Local imports
from egomimic.rldb.zarr.zarr_dataset import ZarrDataset, decode_jxl, load_info


def percentile(data: list[float], p: float) -> float:
    """Calculate the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class S3ZarrDataset(Dataset):
    """
    Zarr dataset that reads from S3 using fsspec.
    Simplified version focused on benchmarking read performance.

    Expects per-episode structure:
        s3://bucket/dataset/
        └── episode_{idx}.zarr/
            ├── meta/
            │   └── info.json
            ├── observation.images.{cam}
            └── ...
    """

    def __init__(
        self,
        s3_path: str,
        storage_options: dict | None = None,
    ):
        super().__init__()
        import s3fs
        import zarr

        self.s3_path = s3_path.rstrip("/")
        self.storage_options = storage_options or {}

        # Initialize S3 filesystem
        self.fs = s3fs.S3FileSystem(**self.storage_options)

        # Discover episodes first
        self._episode_paths: dict[int, str] = {}
        self._episode_stores: dict[int, zarr.Group] = {}
        self._episode_info: dict[int, dict] = {}
        self._discover_episodes()

        # Load metadata from first episode
        if self._episode_paths:
            first_ep = min(self._episode_paths.keys())
            self.info = self._episode_info.get(first_ep, {})
        else:
            self.info = {}

        self.fps = self.info.get("fps", 30)
        self.features = self.info.get("features", {})
        self.camera_keys = [k for k, v in self.features.items() if v.get("dtype") == "jxl"]

        # Build frame index
        self.episode_data_index = self._build_episode_data_index()
        self._num_frames = int(self.episode_data_index["to"][-1]) if len(self.episode_data_index["to"]) > 0 else 0

    def _discover_episodes(self):
        """Find all episode zarr stores."""
        import json

        try:
            entries = self.fs.ls(self.s3_path, detail=False)
        except FileNotFoundError:
            return

        for entry in sorted(entries):
            name = entry.split("/")[-1]
            if not name.startswith("episode_") or not name.endswith(".zarr"):
                continue
            try:
                ep_idx = int(name.replace("episode_", "").replace(".zarr", ""))
            except ValueError:
                continue

            self._episode_paths[ep_idx] = entry

            # Load per-episode metadata
            metadata_path = f"{entry}/metadata.json"
            try:
                with self.fs.open(metadata_path, "r") as f:
                    self._episode_info[ep_idx] = json.load(f)
            except FileNotFoundError:
                self._episode_info[ep_idx] = {}

    def _get_episode_store(self, ep_idx: int):
        """Open/cache episode Zarr store on demand."""
        import zarr

        if ep_idx in self._episode_stores:
            return self._episode_stores[ep_idx]

        ep_path = self._episode_paths.get(ep_idx)
        if ep_path is None:
            return None

        # Open zarr store via fsspec
        store = zarr.open(
            f"s3://{ep_path}",
            mode="r",
            storage_options=self.storage_options,
        )
        self._episode_stores[ep_idx] = store
        return store

    def _build_episode_data_index(self) -> dict[str, torch.Tensor]:
        """Build from/to indices for each episode."""
        episode_lengths = {}

        for ep_idx in sorted(self._episode_paths.keys()):
            store = self._get_episode_store(ep_idx)
            if store is not None:
                keys = list(store.keys())
                if keys:
                    episode_lengths[ep_idx] = store[keys[0]].shape[0]
                else:
                    episode_lengths[ep_idx] = 0
            else:
                episode_lengths[ep_idx] = 0

        from_indices = []
        to_indices = []
        cumulative = 0

        for ep_idx in sorted(episode_lengths.keys()):
            from_indices.append(cumulative)
            cumulative += episode_lengths[ep_idx]
            to_indices.append(cumulative)

        return {
            "from": torch.LongTensor(from_indices),
            "to": torch.LongTensor(to_indices),
        }

    def _global_idx_to_episode_local(self, global_idx: int) -> tuple[int, int]:
        """Map global frame index to (episode_idx, local_idx)."""
        ep_list = sorted(self._episode_paths.keys())

        for i, ep_idx in enumerate(ep_list):
            ep_start = int(self.episode_data_index["from"][i])
            ep_end = int(self.episode_data_index["to"][i])

            if ep_start <= global_idx < ep_end:
                local_idx = global_idx - ep_start
                return ep_idx, local_idx

        raise IndexError(f"Global index {global_idx} out of range")

    def __len__(self):
        return self._num_frames

    def __getitem__(self, idx: int) -> dict:
        """Get a single frame from S3 Zarr store."""
        ep_idx, local_idx = self._global_idx_to_episode_local(idx)

        store = self._get_episode_store(ep_idx)
        if store is None:
            raise ValueError(f"Could not load store for episode {ep_idx}")

        item = {}
        for key in store.keys():
            arr = store[key]
            if local_idx >= arr.shape[0]:
                continue

            if key in self.camera_keys:
                compressed_bytes = arr[local_idx]
                img_np = decode_jxl(compressed_bytes)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                item[key] = img_tensor
            else:
                item[key] = torch.from_numpy(np.array(arr[local_idx]))

        item["episode_index"] = torch.tensor(ep_idx)
        item["frame_index"] = torch.tensor(local_idx)
        item["timestamp"] = torch.tensor(local_idx / self.fps, dtype=torch.float32)

        return item


def benchmark_single_frame_access(
    dataset: Dataset,
    num_samples: int,
    warmup: int,
) -> dict:
    """Benchmark random single-frame access."""
    total_frames = len(dataset)

    # Generate random indices
    random.seed(42)
    indices = [random.randint(0, total_frames - 1) for _ in range(warmup + num_samples)]

    # Warmup
    print(f"  Warming up with {warmup} samples...")
    for i in range(warmup):
        _ = dataset[indices[i]]

    # Benchmark
    print(f"  Benchmarking {num_samples} random frame accesses...")
    latencies = []
    for i in range(warmup, warmup + num_samples):
        start = time.perf_counter()
        _ = dataset[indices[i]]
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    total_time = sum(latencies) / 1000  # Back to seconds
    throughput = num_samples / total_time if total_time > 0 else 0

    return {
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "throughput_samples_sec": throughput,
    }


def benchmark_dataloader(
    dataset: Dataset,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    warmup: int,
    prefetch_factor: int = 2,
) -> dict:
    """Benchmark DataLoader throughput with shuffling."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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

    for i in range(benchmark_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        # Count actual samples in batch (last batch may be smaller)
        batch_samples = batch["frame_index"].shape[0]
        samples_processed += batch_samples

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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Zarr data loading speed for forward pass simulation"
    )

    # Dataset source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset-path",
        type=str,
        help="Path to local zarr dataset",
    )
    source_group.add_argument(
        "--s3-path",
        type=str,
        help="S3 path to zarr dataset (e.g., s3://bucket/dataset)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of random frames to sample (default: 1000)",
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
        default=None,
        help="DataLoader workers (default: 4 for local, 8 for S3)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="DataLoader prefetch factor (default: 2 for local, 4 for S3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before timing (default: 10)",
    )

    # S3 options
    parser.add_argument(
        "--s3-profile",
        type=str,
        default=None,
        help="AWS profile name for S3 access",
    )
    parser.add_argument(
        "--s3-region",
        type=str,
        default=None,
        help="AWS region for S3 access",
    )
    parser.add_argument(
        "--s3-anon",
        action="store_true",
        help="Use anonymous S3 access (for public buckets)",
    )

    # Benchmark modes
    parser.add_argument(
        "--skip-single-frame",
        action="store_true",
        help="Skip single-frame benchmark (slow for S3)",
    )
    parser.add_argument(
        "--single-frame-samples",
        type=int,
        default=None,
        help="Override num-samples for single-frame test (default: same as num-samples, or 50 for S3)",
    )

    args = parser.parse_args()

    # Determine if using S3
    is_s3 = args.s3_path is not None

    # Set defaults based on backend
    if args.num_workers is None:
        args.num_workers = 8 if is_s3 else 4
    if args.prefetch_factor is None:
        args.prefetch_factor = 4 if is_s3 else 2
    if args.single_frame_samples is None:
        args.single_frame_samples = 50 if is_s3 else args.num_samples

    print("=== Zarr Forward Pass Benchmark ===")

    if is_s3:
        print(f"Backend: S3")
        print(f"Dataset: {args.s3_path}")

        # Build storage options
        storage_options = {}
        if args.s3_anon:
            storage_options["anon"] = True
        if args.s3_profile:
            storage_options["profile"] = args.s3_profile
        if args.s3_region:
            storage_options["client_kwargs"] = {"region_name": args.s3_region}

        # Load S3 dataset
        print("\nLoading dataset from S3...")
        dataset = S3ZarrDataset(
            s3_path=args.s3_path,
            storage_options=storage_options,
        )
    else:
        print(f"Backend: Local filesystem")
        dataset_path = Path(args.dataset_path).resolve()
        print(f"Dataset: {dataset_path}")

        # Load local dataset
        print("\nLoading dataset...")
        dataset = ZarrDataset(
            repo_id=dataset_path.name,
            root=dataset_path,
        )

    total_frames = len(dataset)
    print(f"Total frames: {total_frames}")

    if total_frames == 0:
        print("Error: Dataset has no frames!")
        return

    # Adjust num_samples if larger than dataset
    num_samples = min(args.num_samples, total_frames)
    single_frame_samples = min(args.single_frame_samples, total_frames)

    if num_samples < args.num_samples:
        print(f"Note: Reduced num_samples to {num_samples} (dataset size)")

    # Single-frame random access benchmark
    if not args.skip_single_frame:
        print(f"\nSingle-frame random access ({single_frame_samples} samples):")
        if is_s3:
            print("  (Note: This is expected to be slow due to S3 per-request latency)")
        single_results = benchmark_single_frame_access(
            dataset,
            num_samples=single_frame_samples,
            warmup=min(args.warmup, 3) if is_s3 else args.warmup,
        )
        print(f"  Mean latency: {single_results['mean_latency_ms']:.1f} ms")
        print(f"  Median: {single_results['median_latency_ms']:.1f} ms")
        print(f"  P95: {single_results['p95_latency_ms']:.1f} ms")
        print(f"  Throughput: {single_results['throughput_samples_sec']:.1f} samples/sec")
    else:
        print("\nSkipping single-frame benchmark (--skip-single-frame)")

    # DataLoader throughput benchmark
    print(f"\nDataLoader (batch={args.batch_size}, workers={args.num_workers}, prefetch={args.prefetch_factor}):")
    dataloader_results = benchmark_dataloader(
        dataset,
        num_samples=num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup=args.warmup,
        prefetch_factor=args.prefetch_factor,
    )
    print(f"  Throughput: {dataloader_results['throughput_samples_sec']:.1f} samples/sec")
    print(f"  Batches/sec: {dataloader_results['batches_per_sec']:.1f}")
    print(f"  Total time: {dataloader_results['total_time_sec']:.2f}s for {dataloader_results['samples_processed']} samples")

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
