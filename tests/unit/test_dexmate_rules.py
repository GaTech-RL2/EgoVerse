"""The dexmate_bimanual array rules (egomimic.rldb.zarr.dexmate_rules), run by the
episode validator on top of its general checks."""

import numpy as np
import pytest
import zarr
from fixtures.synthetic_episodes import write_dexmate_episode

from egomimic.rldb.zarr import dexmate_rules as dex
from egomimic.rldb.zarr import validate

T = 16
SMALL = {"T": T, "H": 32, "W": 32}
DIP = dex.HAND_JOINT_NAMES["left"].index("left_index_DIP")
REQUIRED = [
    *(f"{side}.{suffix}" for side in dex.SIDES for suffix in dex.REQUIRED_SIDE_ARRAYS),
    *dex.REQUIRED_ARRAYS,
]


def _report(tmp_path, **kw):
    ep = write_dexmate_episode(tmp_path, **{**SMALL, **kw})
    return validate.validate_episode(ep, allow_missing=True)


def test_fixture_episode_passes(tmp_path):
    report = _report(tmp_path)
    assert report.ok, report.errors
    # Only the waived intrinsics and the fixture's non-hash file name.
    assert len(report.warnings) == 2, report.warnings


@pytest.mark.parametrize("key", REQUIRED)
def test_required_array_missing_is_an_error(tmp_path, key):
    report = _report(tmp_path, arrays={key: None})
    assert report.errors == [f"Missing required key: {key}"]


@pytest.mark.parametrize("wrist_cameras", [True, False])
def test_optional_arrays_may_be_absent(tmp_path, wrist_cameras):
    drop = {f"{side}.cmd_hand_keypoints": None for side in dex.SIDES}
    report = _report(tmp_path, arrays=drop, wrist_cameras=wrist_cameras)
    assert report.ok, report.errors


def _edited(tmp_path, key, edit):
    """Report for a fixture episode whose stored ``key`` (padded rows included)
    was changed in place by ``edit``."""
    ep = write_dexmate_episode(tmp_path, **SMALL)
    store = zarr.open_group(str(ep), mode="r+")
    values = store[key][:]
    edit(values)
    store[key][:] = values
    return validate.validate_episode(ep, allow_missing=True)


def _problems(report, key):
    return (
        [e for e in report.errors if e.startswith(f"{key}:")],
        [w for w in report.warnings if w.startswith(f"{key}:")],
    )


def test_padding_rows_are_never_read(tmp_path):
    ep = write_dexmate_episode(tmp_path, **SMALL)
    store = zarr.open_group(str(ep), mode="r+")
    head = store[dex.HEAD_LINK_POSE]
    assert head.shape[0] > T and not head[T:].any()  # zero quaternions past T
    for key in store.array_keys():
        if key.startswith("images.") or key == "annotations":
            continue
        values = store[key][:]
        values[T:] = np.nan
        store[key][:] = values
    report = validate.validate_episode(ep, allow_missing=True)
    assert report.ok, report.errors


@pytest.mark.parametrize(
    "prefix, excess, error, warning",
    [
        ("obs", dex.ERROR_MARGIN_RAD + 1e-6, True, False),
        ("obs", dex.ERROR_MARGIN_RAD - 1e-6, False, True),
        ("obs", dex.WARN_MARGIN_RAD + 1e-6, False, True),
        ("obs", dex.WARN_MARGIN_RAD - 1e-6, False, False),
        ("obs", -(dex.ERROR_MARGIN_RAD + 1e-6), True, False),  # below the lower
        ("cmd", dex.ERROR_MARGIN_RAD + 0.5, False, True),  # commands only warn
        ("cmd", dex.WARN_MARGIN_RAD - 1e-6, False, False),
    ],
)
def test_hand_limit_thresholds(tmp_path, prefix, excess, error, warning):
    """``excess`` past the upper limit (negative: past the lower) on every frame."""
    lower, upper = dex.HAND_JOINT_LIMITS["left"][DIP]
    value = upper + excess if excess > 0 else lower + excess
    key = f"left.{prefix}_hand_joints"

    def edit(values):
        values[:T, DIP] = value

    errors, warnings = _problems(_edited(tmp_path, key, edit), key)
    assert bool(errors) == error, errors
    assert bool(warnings) == warning, warnings
    for message in errors + warnings:
        assert "worst left_index_DIP by" in message
        assert f"on {T} frame(s)" in message


@pytest.mark.parametrize(
    "key, expected, width",
    [
        ("left.obs_tactile", 30, 60),
        ("right.obs_hand_joints", 22, 21),
        ("left.cmd_hand_keypoints", 63, 62),  # optional, but checked when present
        ("obs_aux_joints", 7, 5),
        (dex.HEAD_LINK_POSE, 7, 6),
    ],
)
def test_wrong_width_is_an_error(tmp_path, key, expected, width):
    report = _report(tmp_path, arrays={key: np.zeros((T, width))})
    # Stored padded to the 100-frame chunk.
    assert report.errors == [
        f"{key}: expected shape (T, {expected}), got (100, {width})"
    ]


@pytest.mark.parametrize(
    "key",
    [
        "left.obs_tactile",
        "right.cmd_hand_keypoints",
        "left.obs_joints",
        "cmd_aux_joints",
        dex.HEAD_LINK_POSE,  # a NaN translation, not just a bad quaternion
    ],
)
def test_nonfinite_values_are_an_error(tmp_path, key):
    def edit(values):
        values[T // 2, 0] = np.nan

    errors, _ = _problems(_edited(tmp_path, key, edit), key)
    assert errors == [f"{key}: contains NaN or inf"]


def test_head_link_quaternion_must_be_unit(tmp_path):
    def edit(values):
        values[3, 3:7] *= 1.01

    errors, _ = _problems(
        _edited(tmp_path, dex.HEAD_LINK_POSE, edit), dex.HEAD_LINK_POSE
    )
    assert len(errors) == 1 and "1 frames with non-unit quaternions" in errors[0]


@pytest.mark.parametrize(
    "key, column, name",
    [
        ("left.obs_joints", 2, "left_arm_joint_3"),
        ("right.cmd_hand_joints", DIP, "right_index_DIP"),
        ("obs_aux_joints", 5, "neck_joint_1"),
    ],
)
def test_one_frame_spike_is_a_warning(tmp_path, key, column, name):
    def edit(values):
        values[T // 2, column] += 1.5

    report = _edited(tmp_path, key, edit)
    errors, warnings = _problems(report, key)
    spikes = [w for w in warnings if "one-frame jump" in w]
    assert not errors
    assert len(spikes) == 1 and f"first at frame {T // 2}, {name} =" in spikes[0]


def test_value_beyond_two_pi_is_a_warning(tmp_path):
    def edit(values):
        values[:T, 0] = 7.0

    errors, warnings = _problems(
        _edited(tmp_path, "right.obs_joints", edit), "right.obs_joints"
    )
    assert not errors
    assert len(warnings) == 1 and f"{T} frame(s)" in warnings[0]


def test_check_dexmate_takes_any_mapping_of_arrays():
    store = {key: np.zeros((T, width)) for key, width in dex.REQUIRED_ARRAYS.items()}
    store[dex.HEAD_LINK_POSE][:, 3] = 1.0
    errors, warnings = [], []
    dex.check_dexmate(store, T, errors, warnings)
    missing = {e.removeprefix("Missing required key: ") for e in errors}
    assert missing == {
        f"{side}.{suffix}" for side in dex.SIDES for suffix in dex.REQUIRED_SIDE_ARRAYS
    }
    assert warnings == []
