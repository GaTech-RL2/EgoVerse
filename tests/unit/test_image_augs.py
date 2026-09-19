"""BatchedColorJitter must match torchvision's per-image adjustments, keep
per-sample randomness, and hand anything it cannot batch back to the loop."""

import pytest
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T

from egomimic.models.image_augs import BatchedColorJitter, PerSampleAugs

IMAGENET = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@pytest.fixture
def images():
    torch.manual_seed(0)
    return torch.rand(8, 3, 16, 16)


@pytest.mark.parametrize(
    "kwargs, reference",
    [
        ({"brightness": (0.7, 0.7)}, lambda x: F.adjust_brightness(x, 0.7)),
        ({"contrast": (0.6, 0.6)}, lambda x: F.adjust_contrast(x, 0.6)),
        ({"saturation": (1.3, 1.3)}, lambda x: F.adjust_saturation(x, 1.3)),
        ({"hue": (0.04, 0.04)}, lambda x: F.adjust_hue(x, 0.04)),
    ],
)
def test_matches_torchvision(images, kwargs, reference):
    """A degenerate factor range pins every sample to one factor, so the batched
    adjustment must equal torchvision's scalar one elementwise."""
    out = BatchedColorJitter(**kwargs)(images)
    assert torch.allclose(out, reference(images), atol=1e-5)


def test_factors_are_per_sample(images):
    jitter = BatchedColorJitter(brightness=(0.5, 1.5))
    ratios = (jitter(images) / images.clamp(min=1e-6)).flatten(1).median(dim=1).values
    assert ratios.std() > 1e-3, "every sample got the same brightness factor"


def test_vectorizes_the_shipped_aug_list():
    augs = PerSampleAugs(
        T.Compose(
            [
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                IMAGENET,
            ]
        )
    )
    assert augs.vectorized is not None
    assert isinstance(augs.vectorized[0], BatchedColorJitter)


def test_falls_back_for_unbatchable_transforms(images):
    augs = PerSampleAugs(T.Compose([T.RandomHorizontalFlip(), IMAGENET]))
    assert augs.vectorized is None
    assert augs(images).shape == images.shape


def test_deterministic_aug_list_is_unchanged(images):
    """Eval augs have no random member, so the batched path must be exact."""
    augs = PerSampleAugs(T.Compose([IMAGENET]))
    assert augs.vectorized is not None
    assert torch.allclose(augs(images), IMAGENET(images), atol=1e-6)


def test_output_stays_in_range(images):
    augs = PerSampleAugs(
        T.Compose(
            [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)]
        )
    )
    out = augs(images)
    assert out.min() >= 0.0 and out.max() <= 1.0
