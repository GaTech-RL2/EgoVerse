"""Pre-decode JPEG images from zarr dataset into a single numpy memmap file.
This eliminates JPEG decode overhead during training — loads become simple memcpy.

Usage:
    python predecode_dataset.py <zarr_dir> <output_dir>

Creates:
    <output_dir>/images.npy    — memmap of shape (N_total_frames, 96, 96, 3) uint8
    <output_dir>/metadata.json — episode boundaries, frame counts, episode names

The training DataLoader can then load frames by index directly from the memmap.
"""
import sys
import json
import numpy as np
import zarr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from PIL import Image

def decode_episode(ep_path):
    """Decode all JPEG images in one episode, return as numpy array."""
    z = zarr.open_group(str(ep_path), mode="r")
    jpeg_bytes_arr = z["observations.images.front_img_1"][:]
    n_frames = len(jpeg_bytes_arr)
    frames = np.zeros((n_frames, 96, 96, 3), dtype=np.uint8)
    for i, jpeg_bytes in enumerate(jpeg_bytes_arr):
        img = Image.open(io.BytesIO(bytes(jpeg_bytes)))
        frames[i] = np.array(img)
    return frames

def main():
    if len(sys.argv) < 3:
        print("Usage: python predecode_dataset.py <zarr_dir> <output_dir>")
        sys.exit(1)

    zarr_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted([p for p in zarr_dir.iterdir() if p.name.endswith(".zarr")])
    print(f"Found {len(episodes)} episodes in {zarr_dir}")

    # First pass: count total frames
    print("Counting frames...")
    frame_counts = []
    for i, ep in enumerate(episodes):
        z = zarr.open_group(str(ep), mode="r")
        n = z["observations.images.front_img_1"].shape[0]
        frame_counts.append(n)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(episodes)} episodes scanned, {sum(frame_counts)} frames so far")

    total_frames = sum(frame_counts)
    print(f"Total: {total_frames} frames across {len(episodes)} episodes")
    print(f"Memmap size: {total_frames * 96 * 96 * 3 / 1e9:.2f} GB")

    # Create memmap
    mmap_path = output_dir / "images.npy"
    print(f"Creating memmap at {mmap_path}...")
    mmap = np.memmap(str(mmap_path), dtype=np.uint8, mode="w+",
                     shape=(total_frames, 96, 96, 3))

    # Decode all episodes
    offset = 0
    metadata = {
        "episodes": [],
        "total_frames": total_frames,
        "image_shape": [96, 96, 3],
        "dtype": "uint8",
    }

    for i, ep in enumerate(episodes):
        n = frame_counts[i]
        frames = decode_episode(ep)
        mmap[offset:offset + n] = frames

        metadata["episodes"].append({
            "name": ep.name,
            "start_idx": offset,
            "n_frames": n,
        })
        offset += n

        if (i + 1) % 50 == 0:
            mmap.flush()
            print(f"  [{i+1}/{len(episodes)}] {offset}/{total_frames} frames decoded ({100*offset/total_frames:.1f}%)")

    mmap.flush()
    del mmap

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  images.npy: {total_frames * 96 * 96 * 3 / 1e9:.2f} GB")
    print(f"  metadata.json: episode boundaries")


if __name__ == "__main__":
    main()
