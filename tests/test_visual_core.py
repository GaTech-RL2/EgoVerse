import torch
import torch.nn as nn

from egomimic.models.stems import SpatialSoftmax, VisualCore


def test_spatial_softmax_returns_bounded_keypoints():
    pool = SpatialSoftmax(in_channels=3, in_h=4, in_w=5, num_kp=2)
    output = pool(torch.randn(6, 3, 4, 5))

    assert output.shape == (6, 2, 2)
    assert torch.isfinite(output).all()
    assert torch.all(output.abs() <= 1.0)


def test_visual_core_preserves_leading_shape_and_group_norm_contract():
    encoder = VisualCore(
        image_size=40,
        feature_dimension=12,
        num_kp=4,
        pretrained=False,
        crop_aug=True,
        crop_height=32,
        crop_width=32,
        crop_eval_mode="center",
        norm_layer="group",
    ).eval()

    with torch.inference_mode():
        output = encoder(torch.randn(2, 3, 3, 40, 40))

    assert output.shape == (2, 3, 12)
    assert torch.isfinite(output).all()
    assert not any(isinstance(module, nn.BatchNorm2d) for module in encoder.modules())
