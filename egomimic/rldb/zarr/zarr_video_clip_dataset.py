"""Fixed-length video clip dataset for DFoT training.

Loads pushT zarr episodes, samples random consecutive clips of
``n_frames`` frames each. Returns standard padded batches
``(B, n_frames, C, H, W)`` — no packing, no variable lengths.

Matches the reference DFoT repo's data pipeline:
  - Each sample is exactly ``n_frames`` consecutive frames
  - Random start index within the episode
  - Images in float [0, 1], shape (n_frames, C, H, W)
  - Episodes shorter than n_frames are skipped
"""

from typing import Optional, List, Tuple
import glob
import os
import random

import numpy as np
import simplejpeg
import torch
import zarr
from torch.utils.data import Dataset


class ZarrVideoClipDataset(Dataset):
    """Fixed-length video clip dataset from pushT zarr episodes.

    Args:
        folder_path: path to zarr episode directory.
        n_frames: number of consecutive frames per clip.
        image_key: zarr key for images.
        split: 'train' or 'valid'. Uses last 10% for valid.
        image_size: expected image size (for validation only).
    """

    def __init__(
        self,
        folder_path: str,
        n_frames: int = 9,
        image_key: str = "observations.images.front_img_1",
        split: str = "train",
        image_size: int = 96,
        train_frac: float = 0.9,
        **kwargs,
    ):
        super().__init__()
        self.n_frames = int(n_frames)
        self.image_key = str(image_key)
        self.image_size = int(image_size)

        # Discover all zarr episodes
        all_paths = sorted(glob.glob(os.path.join(folder_path, "*.zarr")))
        if not all_paths:
            raise ValueError(f"No zarr files found in {folder_path}")

        # Train/valid split (deterministic)
        random.seed(42)
        indices = list(range(len(all_paths)))
        random.shuffle(indices)
        n_train = int(len(indices) * train_frac)

        if split == "train":
            selected = [all_paths[i] for i in indices[:n_train]]
        else:
            selected = [all_paths[i] for i in indices[n_train:]]

        # Pre-scan episode lengths and filter short ones
        self.episodes: List[Tuple[str, int]] = []
        for path in selected:
            try:
                z = zarr.open_group(path, mode="r")
                ep_len = z[self.image_key].shape[0]
                if ep_len >= self.n_frames:
                    self.episodes.append((path, ep_len))
            except Exception:
                continue

        # Build index: (episode_idx, num_possible_clips)
        self.clips: List[Tuple[int, int]] = []
        for ep_idx, (_, ep_len) in enumerate(self.episodes):
            n_clips = ep_len - self.n_frames + 1
            self.clips.append((ep_idx, n_clips))

        self.total_clips = sum(n for _, n in self.clips)

    def __len__(self) -> int:
        return self.total_clips

    def _decode_frames(self, zarr_path: str, start: int, end: int) -> torch.Tensor:
        """Load and decode JPEG frames from zarr. Returns (T, 3, H, W) float [0,1]."""
        z = zarr.open_group(zarr_path, mode="r")
        raw = z[self.image_key][start:end]
        frames = []
        for j in range(len(raw)):
            val = raw[j]
            while isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            if isinstance(val, bytes):
                img = simplejpeg.decode_jpeg(val, colorspace="RGB")
            elif isinstance(val, np.ndarray) and val.dtype == np.uint8:
                img = val
            else:
                continue
            frames.append(img)
        if len(frames) < self.n_frames:
            return None
        imgs = np.stack(frames)  # (T, H, W, 3)
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0

    def set_norm_stats_from(self, norm_stats):
        """Stub for trainHydra compatibility."""
        pass

    def get_embodiment(self):
        """Stub for trainHydra compatibility."""
        return "pushshapes_sim"

    @property
    def embodiment_name(self):
        return "pushshapes_sim"

    def __getitem__(self, idx: int):
        # Map flat index to (episode, clip_start)
        for ep_idx, n_clips in self.clips:
            if idx < n_clips:
                path, ep_len = self.episodes[ep_idx]
                start = idx
                frames = self._decode_frames(path, start, start + self.n_frames)
                if frames is None:
                    # Fallback: random valid clip from same episode
                    start = random.randint(0, ep_len - self.n_frames)
                    frames = self._decode_frames(path, start, start + self.n_frames)
                return {"front_img_1": frames, "embodiment": 15}  # (n_frames, 3, H, W)
            idx -= n_clips
        raise IndexError("clip index out of range")
