import math

import numpy as np
import torch

from egomimic.rldb.embodiment.usocket_arc_velocity import (
    USocketArcLocalVelocityNativeDecoder,
    get_usocket_arc_velocity_transform_list,
)


def test_loader_codec_round_trip_and_requested_identity():
    actions = np.zeros((80, 3), dtype=np.float32)
    actions[:, 0] = np.linspace(4.0, 84.0, 80)
    actions[:, 1] = np.linspace(10.0, 30.0, 80)
    actions[:, 2] = np.linspace(-0.2, 0.2, 80)
    codec = get_usocket_arc_velocity_transform_list(
        min_distance_unit=80.0,
        resampled_vector_length=56,
        rotation_distance_unit=math.radians(26),
    )[0]
    token = codec.transform({"actions": actions.copy()})["actions"]
    assert token.shape == (112, 5)
    decoder = USocketArcLocalVelocityNativeDecoder(56, 16)
    decoded = decoder.decode(token)
    assert decoded.shape == (16, 3)
    assert np.isfinite(decoded).all()
    np.testing.assert_allclose(decoded[0], actions[0], atol=1e-5)


def test_native_decoder_preserves_tensor_device_and_grad_free_shape():
    token = torch.zeros(2, 112, 5)
    token[:, :56, 0] = torch.linspace(0, 55, 56)
    token[:, :56, 4] = 30
    token[:, 56:, 2] = 1
    decoded = USocketArcLocalVelocityNativeDecoder(56, 16).decode(token)
    assert decoded.shape == (2, 16, 3)
    assert decoded.device == token.device
    assert torch.isfinite(decoded).all()
