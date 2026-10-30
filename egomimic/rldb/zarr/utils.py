import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42):
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def jpeg_pixels_to_float(decoded: np.ndarray) -> np.ndarray:
    """``(..., H, W, 3)`` uint8 RGB -> ``(..., 3, H, W)`` float32 in [0, 1]."""
    # float32: float64 doubles collate/pin/H2D bytes and is bit-identical here
    # (see tests/unit/test_image_decode_dtype.py).
    return np.moveaxis(decoded, -1, -3).astype(np.float32) / np.float32(255.0)
