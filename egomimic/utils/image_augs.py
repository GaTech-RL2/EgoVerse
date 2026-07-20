import math

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


class BatchedRandomCropRotate:
    """Per-image random square crop + small rotation, vectorized as ONE batched
    affine grid_sample — the fast replacement for PerImageCompose's python loop
    (which cost ~2x training throughput in the pp2 sweep).

    scale: area-fraction range of the crop (RandomResizedCrop semantics,
    ratio fixed to 1). degrees: rotation sampled U(-degrees, +degrees).
    Output size equals input size. Expects float tensors (B,C,H,W) or (C,H,W).
    """

    def __init__(self, scale=(0.55, 1.0), degrees=5.0):
        self.scale = scale
        self.degrees = degrees

    def __call__(self, x):
        single = x.ndim == 3
        if single:
            x = x[None]
        B = x.shape[0]
        dev = x.device
        area = torch.empty(B, device=dev).uniform_(self.scale[0], self.scale[1])
        s = area.sqrt()                              # crop side fraction
        th = torch.empty(B, device=dev).uniform_(
            -self.degrees, self.degrees) * math.pi / 180.0
        # translation keeps the (unrotated) crop inside the image
        lim = (1.0 - s).clamp(min=0)
        tx = (torch.rand(B, device=dev) * 2 - 1) * lim
        ty = (torch.rand(B, device=dev) * 2 - 1) * lim
        cos, sin = th.cos() * s, th.sin() * s
        theta = torch.stack([
            torch.stack([cos, -sin, tx], 1),
            torch.stack([sin, cos, ty], 1),
        ], 1)                                        # (B,2,3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        out = F.grid_sample(x, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        return out[0] if single else out


class PerImageCompose:
    """Compose that draws augmentation params independently PER IMAGE when given
    a batched (B,C,H,W) tensor.

    Stock torchvision transforms sample ONE set of random params per call, so
    applying them to a batch shares a single crop/rotation across all images in
    that batch (how train_image_augs has always been applied here). This wrapper
    restores per-sample diversity at the cost of a python loop over the batch.
    """

    def __init__(self, transforms):
        self.transform = Compose(transforms)

    def __call__(self, x):
        if torch.is_tensor(x) and x.ndim == 4:
            return torch.stack([self.transform(img) for img in x])
        return self.transform(x)
