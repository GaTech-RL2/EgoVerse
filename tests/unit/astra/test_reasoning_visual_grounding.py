import numpy as np
import pytest

from astra_reversal.demo_skill_agent import png_wire
from astra_reversal.reasoning_learning.visual_grounding import (
    build_request,
    fit_projection,
    parse_proposal,
)


def test_projection_rejects_unexcited_motion_and_bad_correspondences():
    xyz = np.random.default_rng(1).normal(0, 0.1, (12, 3))
    jacobian = np.array([[100, -70, 4], [20, 30, -110]])
    pixels = xyz @ jacobian.T + [112, 112]
    fitted = fit_projection(xyz, pixels)
    assert fitted["accepted"] and fitted["loo_rms_pixels"] < 1e-10
    np.testing.assert_allclose(fitted["jacobian_pixels_per_meter"], jacobian)
    pixels[0] += [100, -100]
    assert not fit_projection(xyz, pixels)["accepted"]
    xyz[:, 2] = 0
    assert fit_projection(xyz, pixels)["reason"] == "insufficient_motion_excitation"


def test_pixel_labels_cannot_hide_missing_or_duplicate_frames_or_receive_poses():
    wire = png_wire(np.zeros((8, 8, 3), dtype=np.uint8))
    frames = [{"step": i, "image": wire} for i in range(6)]
    request = build_request(frames, "put bottle in bowl")
    point = {
        "visible": True,
        "confidence": 0.7,
        "uv": [0.5, 0.6],
        "evidence": "Visible fingers",
    }
    proposal = {
        "frames": [{"step": i, **point} for i in range(6)],
        "destination": point,
    }
    assert parse_proposal(proposal, request)["frames"][0]["visible"]
    proposal["frames"][0]["step"] = 1
    with pytest.raises(ValueError, match="exactly one"):
        parse_proposal(proposal, request)
    proposal["frames"][0]["step"] = 0
    proposal["frames"][0]["visible"] = False
    with pytest.raises(ValueError, match="Invisible"):
        parse_proposal(proposal, request)
    frames[0]["state"] = [1, 2, 3]
    with pytest.raises(ValueError, match="no poses"):
        build_request(frames, "put bottle in bowl")
