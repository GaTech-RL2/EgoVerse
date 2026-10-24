"""The dexmate_bimanual array rules (egomimic.rldb.zarr.dexmate_rules), run by the
episode validator on top of its general checks."""

import pytest
from fixtures.synthetic_episodes import write_dexmate_episode

from egomimic.rldb.zarr import dexmate_rules as dex
from egomimic.rldb.zarr import validate

SMALL = {"T": 16, "H": 32, "W": 32}
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
