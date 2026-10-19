"""Per-sample image augmentation.

torchvision's random transforms draw one parameter set per call, so applying a
``ColorJitter`` to a stacked (B, 3, H, W) batch jitters every image in the batch
by the same factors (and in the same op order).  ``PerSampleAugs`` gives every
sample its own parameters.

The obvious way to do that -- call the transform once per image -- costs one
Python round trip and a few dozen tiny CUDA launches *per image*: measured at
89 ms of a 196 ms HPT train step at batch 64 (H100, 2026-09-19), i.e. 46% of
the step, because every factor is a CPU scalar that has to be copied to the
device. ``BatchedColorJitter`` draws the factors as one on-device (B,) tensor
and runs each adjustment once over the whole batch instead.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import _functional_tensor as _ft

_GRAY_W = (0.2989, 0.587, 0.114)


def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(_GRAY_W, dtype=img.dtype, device=img.device).view(3, 1, 1)
    return (img * w).sum(dim=-3, keepdim=True)


def _blend(img1: torch.Tensor, img2: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0.0, 1.0)


class BatchedColorJitter(nn.Module):
    """``torchvision.transforms.ColorJitter`` with per-sample factors, batched.

    Same adjustments and same factor distributions as torchvision, drawn once
    per image instead of once per call. The op ORDER is drawn once per call
    (torchvision's own behaviour), not once per image as the per-image loop
    did; each adjustment clamps to [0, 1], so the order is not quite inert, but
    it is a permutation of the same four factors either way.

    Input is (B, 3, H, W) float in [0, 1]. With ``frames`` > 1 it is
    (B * frames, 3, H, W) with a sample's frames adjacent: they share one draw,
    and every adjustment (contrast's mean included) still runs per frame.
    """

    def __init__(
        self,
        brightness: Optional[Tuple[float, float]] = None,
        contrast: Optional[Tuple[float, float]] = None,
        saturation: Optional[Tuple[float, float]] = None,
        hue: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @classmethod
    def from_color_jitter(cls, jitter: T.ColorJitter) -> "BatchedColorJitter":
        return cls(jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)

    @staticmethod
    def _factors(bounds, n, img) -> torch.Tensor:
        # Drawn in float32: uniform_ into bfloat16 yields ~40 distinct values,
        # and into uint8 it raises.
        lo, hi = bounds
        f = torch.empty(n, 1, 1, 1, dtype=torch.float32, device=img.device)
        return f.uniform_(lo, hi).to(img.dtype)

    def _apply_brightness(self, img, f):
        return (img * f).clamp(0.0, 1.0)

    def _apply_contrast(self, img, f):
        mean = _rgb_to_gray(img).mean(dim=(-3, -2, -1), keepdim=True)
        return _blend(img, mean, f)

    def _apply_saturation(self, img, f):
        return _blend(img, _rgb_to_gray(img), f)

    def _apply_hue(self, img, f):
        hsv = _ft._rgb2hsv(img)
        h, s, v = hsv.unbind(dim=-3)
        h = (h + f.squeeze(-3)) % 1.0
        return _ft._hsv2rgb(torch.stack((h, s, v), dim=-3)).clamp(0.0, 1.0)

    def forward(self, img: torch.Tensor, frames: int = 1) -> torch.Tensor:
        n = img.shape[0] // frames
        ops = [
            (self.brightness, self._apply_brightness),
            (self.contrast, self._apply_contrast),
            (self.saturation, self._apply_saturation),
            (self.hue, self._apply_hue),
        ]
        for idx in torch.randperm(len(ops)).tolist():
            bounds, fn = ops[idx]
            if bounds is None:
                continue
            img = fn(img, self._factors(bounds, n, img).repeat_interleave(frames, 0))
        return img

    def extra_repr(self) -> str:
        return (
            f"brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue}"
        )


# Transforms that are already batch-safe AND sample no parameters, so running
# them on the stacked batch is identical to running them per image.
_DETERMINISTIC = (T.Normalize, T.Resize, T.CenterCrop, T.ConvertImageDtype)


def _vectorize(augs) -> Optional[List[nn.Module]]:
    """The batched equivalent of ``augs``, or None if any member has no batched
    form (then the caller keeps the per-image loop)."""
    members = augs.transforms if isinstance(augs, T.Compose) else [augs]
    out: List[nn.Module] = []
    for t in members:
        if isinstance(t, T.ColorJitter):
            out.append(BatchedColorJitter.from_color_jitter(t))
        elif isinstance(t, _DETERMINISTIC):
            out.append(t)
        else:
            return None
    return out


class PerSampleAugs(nn.Module):
    """Apply ``augs`` to each image of a (B, C, H, W) batch independently.

    ``augs`` is the configured callable (typically a
    ``torchvision.transforms.Compose``); a deterministic one is unaffected by
    the wrapping, so this is safe for any existing aug list.

    ``frames`` > 1 takes (B * frames, C, H, W), each sample's frames adjacent:
    the frames of a sample share one parameter draw, and every transform still
    sees one frame at a time, so a crop, resize or contrast mean never spans
    two frames.
    """

    def __init__(self, augs):
        super().__init__()
        self.augs = augs
        vectorized = _vectorize(augs)
        self.vectorized = nn.ModuleList(vectorized) if vectorized is not None else None

    def forward(self, images, frames: int = 1):
        if self.vectorized is None:
            if frames == 1:
                return torch.stack([self.augs(image) for image in images])
            # torchvision draws once per call and works per image of a stack
            return torch.cat([self.augs(clip) for clip in images.split(frames)])
        for t in self.vectorized:
            images = (
                t(images, frames=frames)
                if isinstance(t, BatchedColorJitter)
                else t(images)
            )
        return images

    def extra_repr(self):
        return repr(self.augs)
