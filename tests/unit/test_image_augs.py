"""BatchedColorJitter must match torchvision's per-image adjustments, keep
per-sample randomness, and refuse anything it cannot batch; HPT applies the
per-sample augs in train mode and the batched eval augs otherwise."""

import pytest
import torch
import torchvision.transforms.functional as F
from torch import nn
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
    assert isinstance(augs.vectorized[0], BatchedColorJitter)


def test_rejects_unbatchable_transforms():
    with pytest.raises(ValueError, match="RandomHorizontalFlip"):
        PerSampleAugs(T.Compose([T.RandomHorizontalFlip(), IMAGENET]))


def test_deterministic_aug_list_is_unchanged(images):
    """Eval augs have no random member, so the batched path must be exact."""
    augs = PerSampleAugs(T.Compose([IMAGENET]))
    assert torch.allclose(augs(images), IMAGENET(images), atol=1e-6)


def test_output_stays_in_range(images):
    augs = PerSampleAugs(
        T.Compose(
            [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)]
        )
    )
    out = augs(images)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_factors_are_drawn_in_float32():
    """uniform_ into bfloat16 collapses 4096 draws onto a few dozen values, and
    into uint8 it raises."""
    img = torch.zeros(4096, 3, 1, 1, dtype=torch.bfloat16)
    f = BatchedColorJitter._factors((0.9, 1.1), 4096, img)
    assert f.dtype == torch.bfloat16
    assert f.float().unique().numel() > 30
    BatchedColorJitter._factors(
        (0.9, 1.1), 4, torch.zeros(4, 3, 1, 1, dtype=torch.uint8)
    )


def test_absent_camera_stays_zero_per_sample():
    """An all-zero sample is an absent camera: it must not be normalized to
    -mean/std, even when the rest of its batch is present."""
    from types import SimpleNamespace

    from egomimic.algo.hpt import HPT

    stub = SimpleNamespace(
        nets=SimpleNamespace(training=True),
        train_image_augs=PerSampleAugs(T.Compose([T.ColorJitter(0.4), IMAGENET])),
        eval_image_augs=None,
        encoders={"front_img_1": object()},
    )
    torch.manual_seed(0)
    images = torch.rand(3, 3, 8, 8)
    images[1] = 0.0
    out = HPT._apply_image_augs(stub, images, "front_img_1")
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert not torch.equal(out[0], images[0]), "present samples are augmented"
    assert torch.equal(HPT._apply_image_augs(stub, images, "not_an_encoder"), images)


def _hpt_stub(train_image_augs=None, eval_image_augs=None, training=True):
    """A bare HPT carrying only what ``_apply_image_augs`` reads; constructing
    a real HPT would build the trunk and stems."""
    from egomimic.algo.hpt import HPT

    algo = HPT.__new__(HPT)
    algo.nets = nn.ModuleDict()
    algo.nets.train(training)
    algo.encoders = {"front_img_1": {}}
    algo.train_image_augs = (
        PerSampleAugs(train_image_augs) if train_image_augs is not None else None
    )
    algo.eval_image_augs = eval_image_augs
    return algo


def test_apply_image_augs_is_per_sample_in_train_mode():
    torch.manual_seed(0)
    algo = _hpt_stub(train_image_augs=T.ColorJitter(brightness=0.5), training=True)
    out = algo._apply_image_augs(torch.full((64, 3, 16, 16), 0.5), "front_img_1")
    assert out.flatten(1).mean(dim=1).std().item() > 0.01


def test_apply_image_augs_eval_mode_equals_batched_normalize(images):
    eval_augs = T.Compose([IMAGENET])
    algo = _hpt_stub(
        train_image_augs=T.ColorJitter(brightness=0.5),
        eval_image_augs=eval_augs,
        training=False,
    )
    torch.testing.assert_close(
        algo._apply_image_augs(images, "front_img_1"), eval_augs(images), rtol=0, atol=0
    )
    # a camera with no encoder passes through untouched
    assert torch.equal(algo._apply_image_augs(images, "wrist_img_1"), images)


def test_per_sample_augs_preserves_shape_dtype_device_and_grad():
    images = torch.rand(4, 3, 16, 16, dtype=torch.float32, requires_grad=True)
    out = PerSampleAugs(T.Compose([IMAGENET]))(images)

    assert out.shape == images.shape
    assert out.dtype == images.dtype
    assert out.device == images.device
    out.sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()


def test_missing_train_augs_pass_through():
    """__init__ allows train_image_augs=None; the helper must not call it."""
    algo = _hpt_stub(train_image_augs=None, eval_image_augs=None, training=True)
    images = torch.rand(2, 3, 8, 8)
    assert algo._apply_image_augs(images, "front_img_1") is images
