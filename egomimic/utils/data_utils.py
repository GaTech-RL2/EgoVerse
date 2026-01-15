import numpy as np
import torch
import torch.nn.functional as F

def downsample_hwc_uint8_in_chunks(
  images: np.ndarray,  # (T,H,W,3) uint8
  out_hw=(240, 320),
  chunk: int = 256,
) -> np.ndarray:
    assert images.dtype == np.uint8 and images.ndim == 4 and images.shape[-1] == 3
    T, H, W, C = images.shape
    outH, outW = out_hw

    out = np.empty((T, outH, outW, 3), dtype=np.uint8)

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        x = torch.from_numpy(images[s:e]).permute(0, 3, 1, 2).to(torch.float32) / 255.0  # (B,3,H,W)
        x = F.interpolate(x, size=(outH, outW), mode="bilinear", align_corners=False)
        x = (x * 255.0).clamp(0, 255).to(torch.uint8)  # (B,3,outH,outW)
        out[s:e] = x.permute(0, 2, 3, 1).cpu().numpy()
        del x

    return out