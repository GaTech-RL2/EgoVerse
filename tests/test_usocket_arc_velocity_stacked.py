import math

import numpy as np
import torch

from egomimic.rldb.embodiment.usocket_arc_velocity import (
    USocketArcLocalVelocityNativeDecoder,
    USocketArcLocalVelocityStackedNativeDecoder,
    get_usocket_arc_velocity_stacked_transform_list,
)
from egomimic.rldb.zarr.planar_arc import (
    TokenizeUSocketArcVelocity,
    TokenizeUSocketArcVelocityStacked,
)

M, D, R, DT, H = 56, 80.0, math.radians(26), 1.0 / 30.0, 16
KW = dict(min_distance_unit=D, resampled_vector_length=M, dt=DT, rotation_distance_unit=R)


def _episode(seed: int, length: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.cumsum(rng.normal(0.04, 0.03, length))
    xy = np.cumsum(rng.normal(0.0, 2.5, (length, 2)), axis=0)
    actions = np.zeros((length, 3))
    actions[:, :2] = xy
    actions[:, 2] = theta
    return actions


def test_stacked_token_shape_and_no_dead_entries():
    token = TokenizeUSocketArcVelocityStacked(**KW).transform(
        {"actions": _episode(0)}
    )["actions"]
    assert token.shape == (M, 6)
    # The row-split layout zero-pads to the common planar width; the stacked
    # layout must have no structurally dead column.
    assert not np.all(token == 0.0, axis=0).any()


def test_stacked_decode_matches_row_split_exactly():
    """Both layouts carry identical information, so they must decode identically."""
    for seed in range(5):
        actions = _episode(seed, length=int(40 + 17 * seed))
        row = TokenizeUSocketArcVelocity(**KW).transform({"actions": actions.copy()})["actions"]
        stacked = TokenizeUSocketArcVelocityStacked(**KW).transform(
            {"actions": actions.copy()}
        )["actions"]
        assert row.shape == (2 * M, 5) and stacked.shape == (M, 6)
        row_native = USocketArcLocalVelocityNativeDecoder(M, H).decode(row)
        stacked_native = USocketArcLocalVelocityStackedNativeDecoder(M, H).decode(stacked)
        np.testing.assert_allclose(stacked_native, row_native, atol=1e-9)


def test_stacked_rotation_budget_is_wired_and_changes_the_token():
    """A dead rotation budget is the failure this codec family already had once."""
    a = _episode(1)
    hot = get_usocket_arc_velocity_stacked_transform_list(
        min_distance_unit=D, resampled_vector_length=M, rotation_distance_unit=math.radians(26)
    )[0]
    cold = get_usocket_arc_velocity_stacked_transform_list(
        min_distance_unit=D, resampled_vector_length=M, rotation_distance_unit=math.radians(14)
    )[0]
    assert hot.rotation_distance is not None and cold.rotation_distance is not None
    t_hot = hot.transform({"actions": a.copy()})["actions"]
    t_cold = cold.transform({"actions": a.copy()})["actions"]
    assert np.abs(t_hot - t_cold).max() > 1e-6


def test_stacked_decoder_preserves_device_and_shape():
    token = torch.zeros(2, M, 6)
    token[:, :, 0] = torch.linspace(0, M - 1, M)
    token[:, :, 2] = 30.0
    token[:, :, 3] = 1.0
    decoded = USocketArcLocalVelocityStackedNativeDecoder(M, H).decode(token)
    assert decoded.shape == (2, H, 3)
    assert decoded.device == token.device
    assert torch.isfinite(decoded).all()


def test_carry_shares_the_translation_clock_and_refuses_a_budget():
    """The shared-clock ablation must have no angular budget at all."""
    import pytest

    from egomimic.rldb.embodiment.usocket_arc_velocity import (
        get_usocket_arc_velocity_carry_transform_list,
    )
    from egomimic.rldb.zarr.planar_arc import TokenizeUSocketArcVelocityCarry

    # A dead knob is the failure this family already shipped once, so the carry
    # variant refuses R rather than silently ignoring it.
    with pytest.raises(ValueError, match="rotation_distance_unit"):
        TokenizeUSocketArcVelocityCarry(
            min_distance_unit=D, resampled_vector_length=M, rotation_distance_unit=R
        )
    with pytest.raises(ValueError, match="angular budget"):
        get_usocket_arc_velocity_carry_transform_list(rotation_distance_unit=R)

    actions = _episode(7)
    carry = get_usocket_arc_velocity_carry_transform_list(
        min_distance_unit=D, resampled_vector_length=M
    )[0]
    token = carry.transform({"actions": actions.copy()})["actions"]
    assert token.shape == (M, 6)
    assert np.isfinite(token).all()

    # Sharing the clock must actually change the rotation stream relative to
    # the independent-clock tokenizer, otherwise the ablation is a no-op.
    hybrid = TokenizeUSocketArcVelocityStacked(**KW).transform(
        {"actions": actions.copy()}
    )["actions"]
    assert np.abs(token[:, 3:] - hybrid[:, 3:]).max() > 1e-6
    # Translation is sampled identically in both.
    np.testing.assert_allclose(token[:, :3], hybrid[:, :3], atol=1e-12)


def test_duration_codec_matches_velocity_and_drops_the_dead_omega_sign():
    """Duration is the primitive the velocity channel is derived from."""
    from egomimic.rldb.embodiment.usocket_arc_velocity import (
        USocketArcDurationNativeDecoder,
    )
    from egomimic.rldb.zarr.planar_arc import TokenizeUSocketArcDuration

    for seed in range(4):
        actions = _episode(seed, length=int(50 + 13 * seed))
        vel = TokenizeUSocketArcVelocityStacked(**KW).transform(
            {"actions": actions.copy()}
        )["actions"]
        dur = TokenizeUSocketArcDuration(**KW).transform(
            {"actions": actions.copy()}
        )["actions"]
        assert dur.shape == (M, 6)
        # Geometry is untouched; only the timing channels differ.
        np.testing.assert_allclose(dur[:, 0:2], vel[:, 0:2], atol=1e-12)
        np.testing.assert_allclose(dur[:, 3:5], vel[:, 3:5], atol=1e-12)
        # Durations are seconds and cannot be negative.
        assert (dur[:, 2] >= 0).all() and (dur[:, 5] >= 0).all()
        # Both timing representations decode to the same native actions.
        v_native = USocketArcLocalVelocityStackedNativeDecoder(M, H).decode(vel)
        d_native = USocketArcDurationNativeDecoder(M, H).decode(dur)
        np.testing.assert_allclose(d_native, v_native, atol=1e-9)


def test_velocity_decoder_ignores_the_sign_of_omega():
    """Documents why a duration channel is not just a cosmetic swap.

    _decode_stream uses rate.abs(), so the predicted sign of omega never
    reaches the output -- direction is already carried by cos/sin. The velocity
    codec therefore spends model capacity on a quantity decoding discards.
    """
    actions = _episode(2)
    token = TokenizeUSocketArcVelocityStacked(**KW).transform(
        {"actions": actions}
    )["actions"]
    decoder = USocketArcLocalVelocityStackedNativeDecoder(M, H)
    base = np.asarray(decoder.decode(token))
    flipped = token.copy()
    flipped[:, 5] = -flipped[:, 5]
    np.testing.assert_array_equal(np.asarray(decoder.decode(flipped)), base)
