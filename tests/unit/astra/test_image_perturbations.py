"""Pixel-level checks; no policy, simulator, provider or external images."""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.image_perturbations import (
    CAMERAS,
    ImagePerturbationLimits,
    apply_image_perturbations,
    validate_image_perturbations,
)
from astra_reversal.records import digest


def observation(height=4, width=8):
    return {
        CAMERAS[0]: np.zeros((height, width, 3), dtype=np.uint8),
        CAMERAS[1]: np.full((height, width, 3), 200, dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "task": "Place the original object in its original destination.",
        "metadata": {"nested": np.array([1, 2], dtype=np.int64)},
    }


def donor(pixels, camera=CAMERAS[0], donor_id="std10-e379-f28"):
    return SimpleNamespace(
        donor_id=donor_id,
        camera=camera,
        pixels=pixels,
        provenance={
            "library_id": "a" * 64,
            "sample_sha256": "b" * 64,
            "pixels_sha256": digest(pixels),
            "source": {"episode_index": 379, "frame_index": 28},
        },
    )


def blend(alpha=0.5, camera=CAMERAS[0]):
    return {
        "kind": "demo_blend",
        "camera": camera,
        "donor_id": "std10-e379-f28",
        "alpha": alpha,
    }


def occlusion(strength=1.0, camera=CAMERAS[0], box=None):
    return {
        "kind": "occlusion",
        "camera": camera,
        "box_xyxy": [2, 1, 6, 3] if box is None else box,
        "fill_rgb": [127, 127, 127],
        "strength": strength,
    }


def lookup_for(*donors):
    index = {(row.donor_id, row.camera): row for row in donors}
    return lambda donor_id, camera: index[donor_id, camera]


def test_noop_returns_independent_exact_observation_and_zero_metrics():
    raw = observation()
    original_digest = digest(raw)
    result, audit = apply_image_perturbations(raw, [])
    assert result is not raw and digest(result) == original_digest == digest(raw)
    assert audit["before_observation_sha256"] == audit["after_observation_sha256"]
    assert audit["has_effect"] is False
    for camera in CAMERAS:
        assert not np.shares_memory(raw[camera], result[camera])
        row = audit["cameras"][camera]
        assert row["before_sha256"] == row["after_sha256"] == digest(raw[camera])
        assert row["changed_fraction"] == row["rms"] == row["linf"] == 0
        assert row["mask_pixels"] == 0 and row["operation"] is None
    result["observation/state"][0] = 100
    result["metadata"]["nested"][0] = 100
    assert digest(raw) == original_digest
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("kind", ["demo_blend", "occlusion"])
def test_zero_weight_preserves_every_field_without_weakening_validation(kind):
    raw = observation()
    sample = donor(np.full_like(raw[CAMERAS[0]], 255))
    operation = blend(0) if kind == "demo_blend" else occlusion(0)
    result, audit = apply_image_perturbations(raw, [operation], lookup_for(sample))
    assert digest(result) == digest(raw)
    assert audit["has_effect"] is False
    row = audit["cameras"][CAMERAS[0]]
    assert row["mask_fraction"] > 0 and row["changed_fraction"] == 0
    if kind == "demo_blend":
        assert row["donor_provenance"]["pixels_sha256"] == digest(sample.pixels)
        sample.provenance["pixels_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="pinned digest"):
            apply_image_perturbations(raw, [operation], lookup_for(sample))
    else:
        with pytest.raises(ValueError, match="area"):
            apply_image_perturbations(raw, [occlusion(0, box=[0, 0, 8, 4])])


def test_full_blend_replaces_only_selected_camera_and_binds_donor_pixels():
    raw = observation()
    pixels = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    sample = donor(pixels)
    original = digest(raw)
    result, audit = apply_image_perturbations(raw, [blend(1)], lookup_for(sample))
    np.testing.assert_array_equal(result[CAMERAS[0]], pixels)
    assert not np.shares_memory(result[CAMERAS[0]], pixels)
    assert digest(raw) == original
    assert digest({k: v for k, v in result.items() if k != CAMERAS[0]}) == digest(
        {k: v for k, v in raw.items() if k != CAMERAS[0]}
    )
    row = audit["cameras"][CAMERAS[0]]
    assert row["after_sha256"] == row["donor_provenance"]["pixels_sha256"]
    assert row["mask_fraction"] == 1.0
    assert row["changed_pixels"] == 32
    assert row["changed_channel_values"] == 95
    assert row["linf"] == 95
    assert row["rms"] == math.sqrt(sum(i * i for i in range(96)) / 96)
    sample.provenance["source"]["frame_index"] = 999
    assert row["donor_provenance"]["source"]["frame_index"] == 28


def test_blend_rounds_half_up_without_uint8_overflow_for_entire_byte_range():
    raw = observation(16, 16)
    channel = np.arange(256, dtype=np.uint8).reshape(16, 16)
    raw[CAMERAS[0]] = np.repeat(channel[:, :, None], 3, axis=2)
    sample = donor(255 - raw[CAMERAS[0]])
    result, audit = apply_image_perturbations(raw, [blend(0.5)], lookup_for(sample))
    np.testing.assert_array_equal(
        result[CAMERAS[0]], np.full((16, 16, 3), 128, np.uint8)
    )
    assert result[CAMERAS[0]].dtype == np.uint8
    assert "floor(value+0.5)" in audit["rounding"]
    # 0.5 ties differ from banker's rounding at an even integer.
    raw[CAMERAS[0]].fill(0)
    one = donor(np.ones_like(raw[CAMERAS[0]]))
    result, _ = apply_image_perturbations(raw, [blend(0.5)], lookup_for(one))
    assert np.all(result[CAMERAS[0]] == 1)


def test_nonzero_parameters_do_not_falsely_claim_actual_pixel_effect():
    raw = observation()
    same = donor(raw[CAMERAS[0]].copy())
    result, audit = apply_image_perturbations(raw, [blend(0.73)], lookup_for(same))
    assert digest(result) == digest(raw) and audit["has_effect"] is False
    almost = donor(np.ones_like(raw[CAMERAS[0]]))
    result, audit = apply_image_perturbations(raw, [blend(0.49)], lookup_for(almost))
    assert digest(result) == digest(raw) and audit["has_effect"] is False


def test_half_open_occlusion_changes_exact_interior_with_area_and_rgb_metrics():
    raw = observation()
    operation = occlusion(box=[4, 0, 8, 4])
    result, audit = apply_image_perturbations(raw, [operation])
    expected = np.zeros((4, 8, 3), dtype=np.uint8)
    expected[:, 4:8, :] = 127
    np.testing.assert_array_equal(result[CAMERAS[0]], expected)
    np.testing.assert_array_equal(raw[CAMERAS[0]], np.zeros_like(expected))
    row = audit["cameras"][CAMERAS[0]]
    assert row["mask_pixels"] == row["changed_pixels"] == 16
    assert row["mask_fraction"] == row["changed_fraction"] == 0.5
    assert row["changed_channel_values"] == 48
    assert row["linf"] == 127 and row["rms"] == math.sqrt(127**2 / 2)
    assert audit["cameras"][CAMERAS[1]]["has_effect"] is False


def test_translucent_occlusion_rounds_and_leaves_outside_pixels_exact():
    raw = observation()
    raw[CAMERAS[0]].fill(10)
    result, _ = apply_image_perturbations(raw, [occlusion(0.5)])
    expected = raw[CAMERAS[0]].copy()
    expected[1:3, 2:6] = 69  # 10*0.5 + 127*0.5 = 68.5, round-half-up.
    np.testing.assert_array_equal(result[CAMERAS[0]], expected)


def test_two_camera_edits_commute_and_application_is_stateless_on_fresh_frames():
    raw = observation()
    sample = donor(np.full_like(raw[CAMERAS[0]], 100))
    operations = [blend(0.5), occlusion(1, camera=CAMERAS[1])]
    first, audit = apply_image_perturbations(raw, operations, lookup_for(sample))
    second, second_audit = apply_image_perturbations(
        raw, operations, lookup_for(sample)
    )
    reversed_result, _ = apply_image_perturbations(
        raw, operations[::-1], lookup_for(sample)
    )
    assert digest(first) == digest(second) == digest(reversed_result)
    assert audit == second_audit
    assert np.all(first[CAMERAS[0]] == 50)
    fresh = observation()
    fresh[CAMERAS[0]].fill(20)
    next_result, _ = apply_image_perturbations(fresh, operations, lookup_for(sample))
    assert np.all(next_result[CAMERAS[0]] == 60)  # No accumulation from previous 50.
    assert np.all(raw[CAMERAS[0]] == 0)


def test_wrist_blend_resolves_matching_camera_and_preserves_external_view():
    raw = observation()
    external = donor(np.zeros_like(raw[CAMERAS[0]]))
    wrist = donor(np.ones_like(raw[CAMERAS[1]]), camera=CAMERAS[1])
    result, audit = apply_image_perturbations(
        raw, [blend(1, CAMERAS[1])], lookup_for(external, wrist)
    )
    assert np.all(result[CAMERAS[1]] == 1)
    np.testing.assert_array_equal(result[CAMERAS[0]], raw[CAMERAS[0]])
    assert audit["cameras"][CAMERAS[0]]["has_effect"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"alpha": True},
        {"alpha": "0.5"},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": 10**1000},
        {"extra": 1},
        {"camera": "other"},
        {"kind": "box"},
        {"donor_id": ""},
        {"donor_id": False},
    ],
)
def test_blend_rejects_invalid_json_scalars_keys_and_identifiers(mutation):
    raw = observation()
    with pytest.raises(ValueError):
        apply_image_perturbations(raw, [{**blend(), **mutation}])


@pytest.mark.parametrize(
    "mutation",
    [
        {"box_xyxy": [0.0, 0, 2, 2]},
        {"box_xyxy": [False, 0, 2, 2]},
        {"box_xyxy": [0, 0, 2]},
        {"box_xyxy": (0, 0, 2, 2)},
        {"box_xyxy": [3, 0, 3, 2]},
        {"box_xyxy": [4, 2, 3, 3]},
        {"box_xyxy": [-1, 0, 2, 2]},
        {"box_xyxy": [0, 0, 9, 2]},
        {"box_xyxy": [0, 0, 2, 5]},
        {"box_xyxy": [0, 0, 8, 3]},
        {"fill_rgb": [0, 0, 0]},
        {"fill_rgb": [127.0, 127, 127]},
        {"fill_rgb": [True, 127, 127]},
        {"fill_rgb": (127, 127, 127)},
        {"strength": True},
        {"strength": -0.1},
        {"strength": 1.1},
        {"strength": float("nan")},
        {"extra": "ignored"},
    ],
)
def test_occlusion_rejects_bad_geometry_fill_and_strength(mutation):
    with pytest.raises(ValueError):
        apply_image_perturbations(observation(), [{**occlusion(), **mutation}])


def test_limits_reject_duplicate_camera_disallowed_kind_and_excess_bounds():
    raw = observation()
    shapes = {camera: raw[camera].shape for camera in CAMERAS}
    for operations in [[blend(), blend()], [blend(), occlusion()], [occlusion()] * 3]:
        with pytest.raises(ValueError):
            validate_image_perturbations(operations, shapes)
    limits = ImagePerturbationLimits(
        allowed_kinds=("occlusion",), max_strength=0.5, max_occlusion_fraction=0.25
    )
    for operations in [[blend()], [occlusion(0.6)], [occlusion(0.5, box=[0, 0, 8, 2])]]:
        with pytest.raises(ValueError):
            validate_image_perturbations(operations, shapes, limits=limits)
    with pytest.raises(ValueError, match="alpha"):
        validate_image_perturbations(
            [blend(0.6)], shapes, limits=ImagePerturbationLimits(max_alpha=0.5)
        )
    assert (
        validate_image_perturbations(
            [],
            shapes,
            limits=ImagePerturbationLimits(allowed_kinds=(), max_operations=0),
        )
        == []
    )


def test_validator_returns_copy_and_restricts_catalog_without_loading_images():
    raw = observation()
    shapes = {camera: raw[camera].shape for camera in CAMERAS}
    operations = [blend()]
    checked = validate_image_perturbations(
        operations, shapes, donor_ids={"std10-e379-f28"}
    )
    assert checked == operations and checked is not operations
    checked[0]["alpha"] = 1
    assert operations[0]["alpha"] == 0.5
    for ids in [set(), "std10-e379-f28", [["nested"]], [True], 123]:
        with pytest.raises(ValueError):
            validate_image_perturbations(operations, shapes, donor_ids=ids)


@pytest.mark.parametrize(
    "error", ["id", "camera", "shape", "dtype", "pixels", "hash", "library", "metadata"]
)
def test_donor_binding_fails_closed_without_implicit_image_conversion(error):
    raw = observation()
    sample = donor(np.full_like(raw[CAMERAS[0]], 255))
    if error == "id":
        sample.donor_id = "wrong"
    elif error == "camera":
        sample.camera = CAMERAS[1]
    elif error == "shape":
        sample.pixels = np.zeros((8, 4, 3), dtype=np.uint8)
    elif error == "dtype":
        sample.pixels = sample.pixels.astype(np.float32)
    elif error == "pixels":
        sample.pixels[0, 0] = 0
    elif error == "hash":
        sample.provenance["pixels_sha256"] = "0" * 64
    elif error == "library":
        sample.provenance.pop("library_id")
    else:
        sample.provenance["unsafe"] = float("nan")
    original = digest(raw)
    with pytest.raises(ValueError):
        apply_image_perturbations(raw, [blend()], lambda *_: sample)
    assert digest(raw) == original


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 8, 3)),
        np.zeros((4, 8), dtype=np.uint8),
        np.zeros((0, 8, 3), dtype=np.uint8),
        np.zeros((4, 8, 4), dtype=np.uint8),
    ],
)
def test_raw_frames_require_nonempty_uint8_rgb_even_for_noop(bad):
    raw = observation()
    raw[CAMERAS[1]] = bad
    with pytest.raises(ValueError):
        apply_image_perturbations(raw, [])


def test_noncontiguous_readonly_inputs_and_donors_are_not_mutated():
    raw = observation()
    raw[CAMERAS[0]] = (
        np.arange(8 * 4 * 3, dtype=np.uint8).reshape(8, 4, 3).transpose(1, 0, 2)
    )
    raw[CAMERAS[0]].flags.writeable = False
    sample = donor(np.full_like(raw[CAMERAS[0]], 240))
    sample.pixels.flags.writeable = False
    before = copy.deepcopy(raw)
    result, audit = apply_image_perturbations(raw, [blend()], lookup_for(sample))
    assert digest(raw) == digest(before)
    assert result[CAMERAS[0]].flags.c_contiguous
    assert audit["cameras"][CAMERAS[0]]["has_effect"] is True
    assert digest(sample.pixels) == sample.provenance["pixels_sha256"]
