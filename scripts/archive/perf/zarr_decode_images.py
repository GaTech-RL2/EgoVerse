"""Rewrite zarr episodes with pre-decoded images (skip JPEG at training time).
Copies all arrays as-is, but decodes observations.images.front_img_1 from JPEG
to float32 CHW [0,1] — exactly what simplejpeg.decode_jpeg + transpose + /255 produces.

Usage:
    python zarr_decode_images.py <src_zarr_dir> <dst_dir>
"""
import sys
import json
import numpy as np
import zarr
from pathlib import Path
import io
import simplejpeg

def process_episode(src_path, dst_path):
    """Decode JPEG images in one episode, copy everything else."""
    src = zarr.open_group(str(src_path), mode="r")
    dst = zarr.open_group(str(dst_path), mode="w")

    # Copy attributes
    dst.attrs.update(dict(src.attrs))

    # Update features metadata to mark images as float32 instead of jpeg
    if "features" in dst.attrs:
        features = json.loads(dst.attrs["features"]) if isinstance(dst.attrs["features"], str) else dict(dst.attrs["features"])
        if "observations.images.front_img_1" in features:
            features["observations.images.front_img_1"]["dtype"] = "float32"
            features["observations.images.front_img_1"]["shape"] = [3, 96, 96]
        dst.attrs["features"] = features

    for key in src.keys():
        arr = src[key]
        if key == "observations.images.front_img_1":
            # Decode JPEG → float32 CHW [0,1]
            jpeg_data = arr[:]
            n_frames = len(jpeg_data)
            decoded = np.zeros((n_frames, 3, 96, 96), dtype=np.float32)
            for i, jpeg_bytes in enumerate(jpeg_data):
                img = simplejpeg.decode_jpeg(bytes(jpeg_bytes), colorspace="RGB")
                decoded[i] = np.transpose(img, (2, 0, 1)) / 255.0
            dst.create_dataset(key, data=decoded, shape=decoded.shape, dtype=decoded.dtype, chunks=(1, 3, 96, 96))
        elif arr.dtype == object:
            # Skip object arrays (annotations, etc) — not needed for training
            continue
        else:
            # Copy numeric arrays directly
            data = arr[:]
            dst.create_dataset(key, data=data, shape=data.shape, dtype=data.dtype)

def main():
    if len(sys.argv) < 3:
        print("Usage: python zarr_decode_images.py <src_zarr_dir> <dst_dir>")
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted([p for p in src_dir.iterdir() if p.name.endswith(".zarr")])
    print(f"Found {len(episodes)} episodes in {src_dir}")

    for i, ep in enumerate(episodes):
        dst_path = dst_dir / ep.name
        process_episode(ep, dst_path)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(episodes)}] decoded")

    # Copy norm_stats if present
    ns = src_dir / "norm_stats.json"
    if ns.exists():
        import shutil
        shutil.copy(str(ns), str(dst_dir / "norm_stats.json"))
        print("Copied norm_stats.json")

    print(f"\nDone! {len(episodes)} episodes decoded to {dst_dir}")


if __name__ == "__main__":
    main()
