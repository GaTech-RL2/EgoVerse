import numpy as np
import pytest
import torch

from egomimic.rldb.embodiment import pushshapes


def test_common5_xy_overlay_uses_only_unnormalized_xy(monkeypatch):
    captured = []

    def fake_draw(frame, pixel_vals, **kwargs):
        captured.append((np.asarray(pixel_vals), kwargs["palette"]))
        return frame

    monkeypatch.setattr(pushshapes, "draw_dot_on_frame", fake_draw)
    images = torch.zeros(1, 3, 8, 8)
    target = torch.tensor([[[64.0, 128.0, 1.0, 0.0, 0.0]]])
    prediction = torch.tensor([[[256.0, 384.0, -1.0, 0.0, 1.0]]])
    frames = pushshapes.viz_gt_preds_xy(
        predictions={"pushshapes_sim_u_socket_actions": prediction},
        batch={
            "embodiment": torch.tensor([19]),
            "front_img_1": images,
            "actions": target,
        },
    )

    assert frames.shape == (1, 32, 32, 3)
    assert [palette for _, palette in captured] == ["Greens", "Reds"]
    np.testing.assert_allclose(captured[0][0], [[4.0, 8.0]])
    np.testing.assert_allclose(captured[1][0], [[16.0, 24.0]])


def test_common5_xy_overlay_rejects_non_finite_xy():
    with pytest.raises(ValueError, match="non-finite"):
        pushshapes._draw_xy_chunk(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.asarray([[np.nan, 1.0, 0.0, 1.0, 0.0]]),
            palette="Reds",
            alpha=1.0,
        )
