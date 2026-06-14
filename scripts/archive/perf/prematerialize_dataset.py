"""Pre-materialize ALL per-frame training samples into .pt tensors.
Decodes all images, extracts action chunks, saves as tensors.
Training then uses TensorDataset — zero overhead per __getitem__.

Usage: python prematerialize_dataset.py <zarr_dir> <output_dir> [--action_horizon 32]
"""
import sys
import json
import numpy as np
import torch
import zarr
import simplejpeg
from pathlib import Path
import time

def main():
    zarr_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    action_horizon = 32
    if "--action_horizon" in sys.argv:
        action_horizon = int(sys.argv[sys.argv.index("--action_horizon") + 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = sorted([p for p in zarr_dir.iterdir() if p.name.endswith(".zarr")])
    print(f"Found {len(episodes)} episodes, action_horizon={action_horizon}")

    # First pass: count total frames
    total = 0
    ep_frames = []
    for ep in episodes:
        z = zarr.open_group(str(ep), mode="r")
        n = z["observations.images.front_img_1"].shape[0]
        ep_frames.append(n)
        total += n
    print(f"Total frames: {total}")

    # Pre-allocate tensors
    images = torch.zeros(total, 3, 96, 96, dtype=torch.float32)
    states = torch.zeros(total, 5, dtype=torch.float32)
    actions = torch.zeros(total, action_horizon, 2, dtype=torch.float32)
    pusher_cmd = torch.zeros(total, 2, dtype=torch.float32)

    offset = 0
    t0 = time.time()
    for i, ep in enumerate(episodes):
        z = zarr.open_group(str(ep), mode="r")
        n = ep_frames[i]

        # Decode images
        jpeg_arr = z["observations.images.front_img_1"][:]
        for j, jpeg_bytes in enumerate(jpeg_arr):
            img = simplejpeg.decode_jpeg(bytes(jpeg_bytes), colorspace="RGB")
            images[offset + j] = torch.from_numpy(np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0)

        # States
        state_arr = np.asarray(z["observations.state"][:])
        states[offset:offset + n] = torch.from_numpy(state_arr.astype(np.float32))

        # Pusher cmd
        if "observations.pusher_cmd_pose" in z:
            cmd_arr = np.asarray(z["observations.pusher_cmd_pose"][:])
            pusher_cmd[offset:offset + n] = torch.from_numpy(cmd_arr.astype(np.float32))

        # Action chunks (with padding at episode end)
        act_arr = np.asarray(z["actions"][:])
        for j in range(n):
            end = min(j + action_horizon, n)
            chunk = act_arr[j:end]
            if len(chunk) < action_horizon:
                pad = np.repeat(chunk[-1:], action_horizon - len(chunk), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            actions[offset + j] = torch.from_numpy(chunk.astype(np.float32))

        offset += n
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(episodes)}] {offset}/{total} frames, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Saving tensors...")

    torch.save({
        "images": images,
        "states": states,
        "actions": actions,
        "pusher_cmd": pusher_cmd,
        "total_frames": total,
        "action_horizon": action_horizon,
        "n_episodes": len(episodes),
    }, str(output_dir / "dataset.pt"))

    size_gb = (images.nbytes + states.nbytes + actions.nbytes + pusher_cmd.nbytes) / 1e9
    print(f"Saved: {output_dir}/dataset.pt ({size_gb:.2f} GB)")
    print(f"  images: {images.shape}")
    print(f"  states: {states.shape}")
    print(f"  actions: {actions.shape}")


if __name__ == "__main__":
    main()
