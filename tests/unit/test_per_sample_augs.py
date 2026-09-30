"""Per-sample image augmentation (ideas 0.10).

torchvision's random transforms draw one parameter set per call, so the old
inline ``self.train_image_augs(_data)`` on the stacked (B, 3, H, W) batch gave
every image in the batch the same jitter.  These tests pin the per-sample
behaviour of ``PerSampleAugs`` and of ``HPT._apply_image_augs``, and pin that
the deterministic eval path is unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import ColorJitter, Compose, Normalize

from egomimic.models.image_augs import PerSampleAugs

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _identical_batch(n=64, mean=0.5):
    """``n`` copies of one image whose per-image mean is exactly ``mean``."""
    return torch.full((n, 3, 16, 16), float(mean))


def _per_image_means(x):
    return x.flatten(1).mean(dim=1)


def test_per_sample_augs_jitters_each_image_independently():
    """A batch of identical images must come out with per-image brightness
    spread; the same ColorJitter called on the whole batch must not."""
    torch.manual_seed(0)
    jitter = ColorJitter(brightness=0.5)
    images = _identical_batch()

    per_sample_std = _per_image_means(PerSampleAugs(jitter)(images)).std().item()
    batched_std = _per_image_means(jitter(images)).std().item()
    print(
        f"per-image brightness std: per-sample={per_sample_std:.6f} "
        f"batched={batched_std:.6f}"
    )

    # brightness=0.5 scales each image by U(0.5, 1.5), so the per-image means
    # of a mean-0.5 batch are U(0.25, 0.75): std = 0.5 / sqrt(12) ~ 0.144.
    # 0.01 is ~14x below that and ~0 for the batched call.
    assert per_sample_std > 0.01
    assert batched_std < 1e-6


def test_per_sample_augs_matches_batched_for_deterministic_transform():
    """A deterministic Compose through PerSampleAugs is exactly the batched
    result, so wrapping a Normalize-only aug list is a no-op."""
    augs = Compose([Normalize(mean=MEAN, std=STD)])
    images = torch.rand(8, 3, 16, 16)
    torch.testing.assert_close(
        PerSampleAugs(augs)(images), augs(images), rtol=0, atol=0
    )


def test_per_sample_augs_preserves_shape_dtype_device_and_grad():
    images = torch.rand(4, 3, 16, 16, dtype=torch.float32, requires_grad=True)
    out = PerSampleAugs(Compose([Normalize(mean=MEAN, std=STD)]))(images)

    assert out.shape == images.shape
    assert out.dtype == images.dtype
    assert out.device == images.device
    out.sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()


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
    algo = _hpt_stub(train_image_augs=ColorJitter(brightness=0.5), training=True)
    out = algo._apply_image_augs(_identical_batch(), "front_img_1")
    assert _per_image_means(out).std().item() > 0.01


def test_apply_image_augs_eval_mode_equals_batched_normalize():
    eval_augs = Compose([Normalize(mean=MEAN, std=STD)])
    algo = _hpt_stub(
        train_image_augs=ColorJitter(brightness=0.5),
        eval_image_augs=eval_augs,
        training=False,
    )
    images = torch.rand(8, 3, 16, 16)
    torch.testing.assert_close(
        algo._apply_image_augs(images, "front_img_1"), eval_augs(images), rtol=0, atol=0
    )


def test_apply_image_augs_passes_through_uncoded_camera():
    """A camera with no encoder is left alone in either mode."""
    images = torch.rand(4, 3, 16, 16)
    for training in (True, False):
        algo = _hpt_stub(
            train_image_augs=ColorJitter(brightness=0.5),
            eval_image_augs=Compose([Normalize(mean=MEAN, std=STD)]),
            training=training,
        )
        out = algo._apply_image_augs(images, "wrist_img_1")
        torch.testing.assert_close(out, images, rtol=0, atol=0)


def test_missing_train_augs_pass_through():
    """__init__ allows train_image_augs=None; the helper must not call it."""
    from egomimic.algo.hpt import HPT

    algo = HPT.__new__(HPT)
    algo.nets = torch.nn.ModuleDict({"policy": torch.nn.Identity()})
    algo.nets.train()
    algo.train_image_augs = None
    algo.eval_image_augs = None
    algo.encoders = {"front_img_1": object()}
    images = torch.rand(2, 3, 8, 8)
    assert algo._apply_image_augs(images, "front_img_1") is images
