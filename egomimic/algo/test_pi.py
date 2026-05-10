from types import SimpleNamespace

import pytest
import torch

import egomimic.algo.pi as pi_module
from egomimic.algo.pi import PI
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class _StubNormStats:
    def __init__(self, viz_img_keys):
        self._viz_img_keys = viz_img_keys

    def viz_img_key(self):
        return self._viz_img_keys


def _make_transform(name):
    return SimpleNamespace(
        extrinsics={"name": f"{name}_extrinsics"},
        intrinsics={"name": f"{name}_intrinsics"},
    )


def _make_pi(camera_transforms, domains):
    pi = object.__new__(PI)
    pi.domains = domains
    pi.camera_transforms = camera_transforms
    pi.is_6dof = False
    pi.ac_keys = {get_embodiment_id(domain): "actions_cartesian" for domain in domains}
    pi.norm_stats = _StubNormStats(
        {get_embodiment_id(domain): "front_img_1" for domain in domains}
    )
    return pi


def _make_batch(embodiment_name):
    embodiment_id = get_embodiment_id(embodiment_name)
    return {
        "embodiment": torch.tensor([embodiment_id]),
        "front_img_1": torch.zeros(1, 3, 4, 4),
        "actions_cartesian": torch.zeros(1, 2, 6),
    }


def _make_predictions(embodiment_name):
    return {f"{embodiment_name}_actions_cartesian": torch.ones(1, 2, 6)}


def test_visualize_preds_supports_single_transform_object(monkeypatch):
    shared_transform = _make_transform("shared")
    pi = _make_pi(shared_transform, ["aria_bimanual"])

    draw_calls = []

    def fake_draw_actions(
        im, ac_type, color, actions, extrinsics, intrinsics, arm="both", **kwargs
    ):
        draw_calls.append((extrinsics, intrinsics))
        return im

    monkeypatch.setattr(pi_module, "draw_actions", fake_draw_actions)

    ims = pi.visualize_preds(
        _make_predictions("aria_bimanual"), _make_batch("aria_bimanual")
    )

    assert ims.shape == (1, 4, 4, 3)
    assert len(draw_calls) == 2
    assert all(
        extrinsics is shared_transform.extrinsics for extrinsics, _ in draw_calls
    )
    assert all(
        intrinsics is shared_transform.intrinsics for _, intrinsics in draw_calls
    )


def test_visualize_preds_raises_clear_error_for_missing_embodiment():
    pi = _make_pi(
        {"aria_bimanual": _make_transform("aria")},
        ["aria_bimanual", "eva_bimanual"],
    )

    with pytest.raises(KeyError) as exc_info:
        pi.visualize_preds(
            _make_predictions("eva_bimanual"), _make_batch("eva_bimanual")
        )

    assert "Missing camera transform for embodiment 'eva_bimanual'" in str(
        exc_info.value
    )
    assert "aria_bimanual" in str(exc_info.value)


def test_visualize_preds_rejects_invalid_camera_transform_shape():
    pi = _make_pi({"aria_bimanual": {"extrinsics": {}}}, ["aria_bimanual"])

    with pytest.raises(TypeError) as exc_info:
        pi.visualize_preds(
            _make_predictions("aria_bimanual"), _make_batch("aria_bimanual")
        )

    assert "camera_transforms must be a CameraTransforms instance or a mapping" in str(
        exc_info.value
    )


def test_visualize_preds_uses_embodiment_specific_camera_transform(monkeypatch):
    aria_transform = _make_transform("aria")
    eva_transform = _make_transform("eva")
    pi = _make_pi(
        {"aria_bimanual": aria_transform, "eva_bimanual": eva_transform},
        ["aria_bimanual", "eva_bimanual"],
    )

    draw_calls = []

    def fake_draw_actions(
        im, ac_type, color, actions, extrinsics, intrinsics, arm="both", **kwargs
    ):
        draw_calls.append(
            {
                "ac_type": ac_type,
                "color": color,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "arm": arm,
                "shape": tuple(actions.shape),
            }
        )
        return im

    monkeypatch.setattr(pi_module, "draw_actions", fake_draw_actions)

    ims = pi.visualize_preds(
        _make_predictions("aria_bimanual"), _make_batch("aria_bimanual")
    )

    assert ims.shape == (1, 4, 4, 3)
    assert len(draw_calls) == 2
    assert all(call["extrinsics"] is aria_transform.extrinsics for call in draw_calls)
    assert all(call["intrinsics"] is aria_transform.intrinsics for call in draw_calls)
    assert all(call["arm"] == "both" for call in draw_calls)
    assert all(call["shape"] == (2, 6) for call in draw_calls)
    assert all(
        call["extrinsics"] is not eva_transform.extrinsics for call in draw_calls
    )
