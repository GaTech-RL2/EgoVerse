#!/usr/bin/env python3
"""
Benchmark JPEG-XL compression quality vs reconstruction error.

Measures compression ratio, PSNR, SSIM, MSE, and encode/decode times
across multiple quality levels for Zarr or HDF5 datasets.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import imagecodecs
import numpy as np
import zarr
from tqdm import tqdm

# Lazy imports for scikit-image metrics (checked at runtime)
_skimage_metrics = None


def _get_skimage_metrics():
    """Lazy import of scikit-image metrics."""
    global _skimage_metrics
    if _skimage_metrics is None:
        try:
            from skimage.metrics import (
                mean_squared_error,
                peak_signal_noise_ratio,
                structural_similarity,
            )
            _skimage_metrics = {
                "mse": mean_squared_error,
                "psnr": peak_signal_noise_ratio,
                "ssim": structural_similarity,
            }
        except ImportError:
            raise ImportError(
                "scikit-image is required for quality metrics. "
                "Install with: pip install scikit-image"
            )
    return _skimage_metrics


def decode_jxl(data) -> np.ndarray:
    """Decode JPEG-XL image to numpy array.

    Args:
        data: Compressed image bytes (with padding byte from encoder),
              may be wrapped in numpy array from zarr

    Returns:
        RGB image array of shape (H, W, C) with dtype uint8
    """
    # Extract bytes from numpy array wrapper (zarr VariableLengthBytes returns 0-dim array)
    while isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    # Strip padding byte added by encoder to prevent null-byte stripping
    if data and data[-1:] == b'\x01':
        data = data[:-1]
    return imagecodecs.jpegxl_decode(data)


def encode_jxl(image: np.ndarray, quality: int) -> bytes:
    """Encode image to JPEG-XL format without padding byte.

    Args:
        image: RGB image array of shape (H, W, C) with dtype uint8
        quality: Compression quality (0-100), higher is better quality

    Returns:
        Compressed image bytes (no padding, for accurate size measurement)
    """
    return imagecodecs.jpegxl_encode(image, level=quality)


def load_images_from_zarr(
    zarr_path: Path,
    num_samples: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Load and decode images from a Zarr dataset.

    Args:
        zarr_path: Path to Zarr dataset root
        num_samples: Number of images to sample
        seed: Random seed for reproducibility

    Returns:
        List of RGB images as numpy arrays (H, W, C), uint8
    """
    images = []

    # Find episode directories
    episode_dirs = sorted([
        p for p in zarr_path.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and p.name.endswith(".zarr")
    ])

    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found in {zarr_path}")

    # Collect all image array references
    image_refs = []  # List of (episode_path, image_key, frame_idx)

    for ep_dir in episode_dirs:
        store = zarr.open(str(ep_dir), mode="r")

        # Find image keys (variable-length byte arrays with "images" in name)
        for key in store.keys():
            arr = store[key]
            if not hasattr(arr, "dtype"):
                continue
            if "images" in key and arr.dtype.kind == "O":
                for frame_idx in range(arr.shape[0]):
                    image_refs.append((ep_dir, key, frame_idx))

    if not image_refs:
        raise ValueError(f"No image arrays found in {zarr_path}")

    # Sample random images
    random.seed(seed)
    sampled_refs = random.sample(image_refs, min(num_samples, len(image_refs)))

    print(f"Loading {len(sampled_refs)} images from Zarr dataset...")
    for ep_dir, key, frame_idx in tqdm(sampled_refs, desc="Loading images"):
        store = zarr.open(str(ep_dir), mode="r")
        compressed_bytes = store[key][frame_idx]
        img = decode_jxl(compressed_bytes)
        images.append(img)

    return images


def load_images_from_hdf5(
    hdf5_path: Path,
    num_samples: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Load and decode images from an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        num_samples: Number of images to sample
        seed: Random seed for reproducibility

    Returns:
        List of RGB images as numpy arrays (H, W, C), uint8
    """
    import h5py

    images = []

    with h5py.File(hdf5_path, "r") as f:
        # Collect all image array references
        image_refs = []  # List of (dataset_path, frame_idx)

        def find_image_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Look for image datasets (typically have "image" in name or shape suggests images)
                if "image" in name.lower() or "rgb" in name.lower():
                    for i in range(obj.shape[0]):
                        image_refs.append((name, i))

        f.visititems(find_image_datasets)

        if not image_refs:
            raise ValueError(f"No image datasets found in {hdf5_path}")

        # Sample random images
        random.seed(seed)
        sampled_refs = random.sample(image_refs, min(num_samples, len(image_refs)))

        print(f"Loading {len(sampled_refs)} images from HDF5 file...")
        for dataset_path, frame_idx in tqdm(sampled_refs, desc="Loading images"):
            img_data = f[dataset_path][frame_idx]

            # Handle compressed images (JPEG/PNG stored as bytes)
            if img_data.dtype == np.uint8 and img_data.ndim == 1:
                # Compressed image bytes
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img_data.ndim == 3:
                # Already decoded image
                if img_data.shape[0] in [1, 3, 4]:  # CHW format
                    img = np.transpose(img_data, (1, 2, 0))
                else:
                    img = img_data
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
            else:
                continue

            images.append(img)

    return images


def benchmark_quality_level(
    images: list[np.ndarray],
    quality: int,
    verbose: bool = False,
) -> dict:
    """Benchmark a single quality level across all images.

    Args:
        images: List of RGB images (H, W, C), uint8
        quality: JPEG-XL quality level (0-100)
        verbose: Print per-image details

    Returns:
        Dict with aggregated metrics
    """
    results = {
        "quality": quality,
        "compression_ratios": [],
        "psnr_values": [],
        "ssim_values": [],
        "mse_values": [],
        "encode_times": [],
        "decode_times": [],
    }

    for img in images:
        raw_size = img.nbytes

        # Encode
        t0 = time.perf_counter()
        compressed = encode_jxl(img, quality)
        encode_time = time.perf_counter() - t0

        compressed_size = len(compressed)
        compression_ratio = compressed_size / raw_size

        # Decode
        t0 = time.perf_counter()
        decoded = imagecodecs.jpegxl_decode(compressed)
        decode_time = time.perf_counter() - t0

        # Quality metrics
        metrics = _get_skimage_metrics()
        psnr = metrics["psnr"](img, decoded, data_range=255)
        ssim = metrics["ssim"](img, decoded, channel_axis=2, data_range=255)
        mse = metrics["mse"](img, decoded)

        results["compression_ratios"].append(compression_ratio)
        results["psnr_values"].append(psnr)
        results["ssim_values"].append(ssim)
        results["mse_values"].append(mse)
        results["encode_times"].append(encode_time * 1000)  # Convert to ms
        results["decode_times"].append(decode_time * 1000)

        if verbose:
            print(
                f"  Quality {quality}: ratio={compression_ratio:.4f}, "
                f"PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, MSE={mse:.6f}"
            )

    # Compute aggregates
    return {
        "quality": quality,
        "ratio_mean": np.mean(results["compression_ratios"]),
        "ratio_std": np.std(results["compression_ratios"]),
        "psnr_mean": np.mean(results["psnr_values"]),
        "psnr_std": np.std(results["psnr_values"]),
        "ssim_mean": np.mean(results["ssim_values"]),
        "ssim_std": np.std(results["ssim_values"]),
        "mse_mean": np.mean(results["mse_values"]),
        "mse_std": np.std(results["mse_values"]),
        "encode_ms_mean": np.mean(results["encode_times"]),
        "encode_ms_std": np.std(results["encode_times"]),
        "decode_ms_mean": np.mean(results["decode_times"]),
        "decode_ms_std": np.std(results["decode_times"]),
    }


def print_results_table(results: list[dict]) -> None:
    """Print results as a formatted ASCII table."""
    print()
    header = " Quality |    Ratio | PSNR (dB) |     SSIM |          MSE |   Enc (ms) |   Dec (ms)"
    separator = "-" * len(header)

    print(header)
    print(separator)

    for r in results:
        print(
            f"{r['quality']:>8} | "
            f"{r['ratio_mean']:>8.4f} | "
            f"{r['psnr_mean']:>9.2f} | "
            f"{r['ssim_mean']:>8.4f} | "
            f"{r['mse_mean']:>12.6f} | "
            f"{r['encode_ms_mean']:>10.2f} | "
            f"{r['decode_ms_mean']:>10.2f}"
        )

    print(separator)


def print_observations(results: list[dict]) -> None:
    """Print summary observations from the results."""
    print("\n=== Observations ===")

    # Find lowest quality with SSIM >= 0.95
    for r in results:
        if r["ssim_mean"] >= 0.95:
            print(f"Lowest quality with SSIM >= 0.95: {r['quality']}")
            break
    else:
        print("No quality level achieved SSIM >= 0.95")

    # Find lowest quality with PSNR >= 40 dB
    for r in results:
        if r["psnr_mean"] >= 40:
            print(f"Lowest quality with PSNR >= 40 dB: {r['quality']}")
            break
    else:
        print("No quality level achieved PSNR >= 40 dB")

    # Best compression ratio with SSIM >= 0.99
    high_quality = [r for r in results if r["ssim_mean"] >= 0.99]
    if high_quality:
        best = min(high_quality, key=lambda x: x["ratio_mean"])
        print(
            f"Best compression ratio with SSIM >= 0.99: "
            f"quality={best['quality']}, ratio={best['ratio_mean']:.4f}"
        )


def save_results_csv(results: list[dict], output_path: Path) -> None:
    """Save results to a CSV file."""
    fieldnames = [
        "quality",
        "ratio_mean", "ratio_std",
        "psnr_mean", "psnr_std",
        "ssim_mean", "ssim_std",
        "mse_mean", "mse_std",
        "encode_ms_mean", "encode_ms_std",
        "decode_ms_mean", "decode_ms_std",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark JPEG-XL compression quality vs reconstruction error"
    )

    # Data source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--zarr-path",
        type=Path,
        help="Path to Zarr dataset root",
    )
    source_group.add_argument(
        "--hdf5-path",
        type=Path,
        help="Path to HDF5 file",
    )

    # Benchmark parameters
    parser.add_argument(
        "--quality-levels",
        type=int,
        nargs="+",
        default=[10, 30, 50, 70, 90, 100],
        help="Quality levels to test (default: 10 30 50 70 90 100)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of images to sample (default: 100)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Path to save CSV results (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-quality details",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=== JPEG-XL Compression Quality Benchmark ===\n")

    # Load images
    if args.zarr_path:
        print(f"Source: Zarr dataset at {args.zarr_path}")
        images = load_images_from_zarr(args.zarr_path, args.num_samples, args.seed)
    else:
        print(f"Source: HDF5 file at {args.hdf5_path}")
        images = load_images_from_hdf5(args.hdf5_path, args.num_samples, args.seed)

    print(f"Loaded {len(images)} images")
    if images:
        print(f"Image shape: {images[0].shape}, dtype: {images[0].dtype}")

    # Run benchmarks for each quality level
    print(f"\nBenchmarking quality levels: {args.quality_levels}")
    results = []

    for quality in tqdm(args.quality_levels, desc="Quality levels"):
        result = benchmark_quality_level(images, quality, args.verbose)
        results.append(result)

    # Sort results by quality level
    results.sort(key=lambda x: x["quality"])

    # Print results table
    print_results_table(results)

    # Print observations
    print_observations(results)

    # Save to CSV if requested
    if args.output_csv:
        save_results_csv(results, args.output_csv)

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
