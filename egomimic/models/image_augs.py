"""Per-sample image augmentation.

torchvision's random transforms draw one parameter set per call, so applying a
``ColorJitter`` to a stacked (B, 3, H, W) batch jitters every image in the batch
by the same factors (and in the same op order).  ``PerSampleAugs`` wraps the
configured transform and calls it once per image, so every sample draws its own
parameters.
"""

import torch
from torch import nn


class PerSampleAugs(nn.Module):
    """Apply ``augs`` to each image of a (B, C, H, W) batch independently.

    ``augs`` is the configured callable (typically a
    ``torchvision.transforms.Compose``); a deterministic one is unaffected by
    the wrapping, so this is safe for any existing aug list.
    """

    def __init__(self, augs):
        super().__init__()
        self.augs = augs

    def forward(self, images):
        return torch.stack([self.augs(image) for image in images])

    def extra_repr(self):
        return repr(self.augs)
