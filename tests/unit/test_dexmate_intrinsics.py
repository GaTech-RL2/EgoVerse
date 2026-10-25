"""Intrinsics may be null only for dexmate_bimanual (CONTRIBUTING_DATA.md §6.4).

Writer: intrinsics=MissingMetadata() is accepted only for Dexmate and stored as an
explicit null; every other embodiment, and None, still raises. Validator: an absent
key is always an error; an explicit null is an error for every embodiment except
Dexmate, where --allow-missing makes it a warning; a dict gets the usual checks.
"""

import numpy as np
import pytest
import zarr
from fixtures.synthetic_episodes import write_dexmate_episode, write_episode

from egomimic.rldb.zarr import validate
from egomimic.rldb.zarr.required_metadata import MissingMetadata
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

K = np.array([[100.0, 0, 16, 0], [0, 100.0, 16, 0], [0, 0, 1.0, 0]])
WAIVED = "intrinsics: marked as not available"
NOT_A_DICT = "intrinsics must be a non-empty"
SMALL = {"T": 16, "H": 32, "W": 32}


def _write(path, embodiment, intrinsics, extrinsics=None):
    return ZarrWriter(
        path,
        embodiment=embodiment,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        task_name="fold_clothes",
        operator="op_7",
        lab="rl2",
        scene="kitchen_a",
        objects=["shirt"],
        vendor_recording_id="session_0412",
    ).write(numeric_data={"actions": np.zeros((8, 3), np.float32)})


def _attrs(path):
    return dict(zarr.open_group(str(path), mode="r").attrs)


def _set_attrs(ep, drop=(), **attrs):
    group = zarr.open_group(str(ep), mode="r+")
    meta = {k: v for k, v in dict(group.attrs).items() if k not in drop}
    meta.update(attrs)
    group.attrs.clear()
    group.attrs.update(meta)
    return ep


@pytest.mark.parametrize("embodiment", ["dexmate_bimanual", "DEXMATE_BIMANUAL"])
def test_writer_stores_dexmate_missing_intrinsics_as_null(tmp_path, embodiment):
    attrs = _attrs(_write(tmp_path / "ep.zarr", embodiment, MissingMetadata()))
    assert "intrinsics" in attrs and attrs["intrinsics"] is None


def test_writer_keeps_dexmate_intrinsics_dict(tmp_path):
    attrs = _attrs(_write(tmp_path / "ep.zarr", "dexmate_bimanual", {"front_1": K}))
    assert attrs["intrinsics"] == {"front_1": K.tolist()}


@pytest.mark.parametrize(
    "embodiment, intrinsics",
    [
        ("human_bimanual", MissingMetadata()),
        ("eva_bimanual", MissingMetadata()),
        ("dexmate_bimanual", None),  # only the explicit sentinel waives
        ("dexmate_bimanual", {}),
    ],
)
def test_writer_rejects_other_missing_intrinsics(tmp_path, embodiment, intrinsics):
    with pytest.raises(ValueError, match=NOT_A_DICT):
        _write(tmp_path / "ep.zarr", embodiment, intrinsics)
    assert not (tmp_path / "ep.zarr").exists()


def test_writer_checks_dexmate_extrinsics_without_intrinsics(tmp_path):
    with pytest.raises(ValueError, match="extrinsics"):
        _write(tmp_path / "ep.zarr", "dexmate_bimanual", MissingMetadata(), {})


def test_validator_dexmate_null_intrinsics_needs_allow_missing(tmp_path):
    ep = write_dexmate_episode(tmp_path, **SMALL)
    strict = validate.validate_episode(ep)
    assert [e for e in strict.errors if "intrinsics" in e] == [
        f"{WAIVED} (MissingMetadata). We heavily recommend providing it; pass "
        "--allow-missing to accept the episode anyway."
    ]
    lenient = validate.validate_episode(ep, allow_missing=True)
    assert lenient.ok, lenient.errors
    assert any(WAIVED in w for w in lenient.warnings)
    assert validate.main([str(ep)]) == 1
    assert validate.main([str(ep), "--allow-missing"]) == 0


@pytest.mark.parametrize("allow_missing", [False, True])
def test_validator_absent_intrinsics_is_an_error(tmp_path, allow_missing):
    for ep in (
        write_dexmate_episode(tmp_path, **SMALL),
        write_episode(tmp_path, "aria", **SMALL),
    ):
        report = validate.validate_episode(
            _set_attrs(ep, drop=["intrinsics"]), allow_missing=allow_missing
        )
        assert any(NOT_A_DICT in e for e in report.errors), report.errors
        assert not any(WAIVED in m for m in report.errors + report.warnings)


@pytest.mark.parametrize("vendor", ["aria", "eva"])
@pytest.mark.parametrize("allow_missing", [False, True])
def test_validator_null_intrinsics_is_an_error_off_dexmate(
    tmp_path, vendor, allow_missing
):
    ep = _set_attrs(write_episode(tmp_path, vendor, **SMALL), intrinsics=None)
    report = validate.validate_episode(ep, allow_missing=allow_missing)
    assert any(NOT_A_DICT in e for e in report.errors), report.errors
    assert not any(WAIVED in m for m in report.errors + report.warnings)


def test_validator_checks_a_dexmate_intrinsics_dict_as_usual(tmp_path):
    good = write_dexmate_episode(tmp_path, intrinsics={"front_1": K}, **SMALL)
    report = validate.validate_episode(good)
    assert report.ok, report.errors
    bad = _set_attrs(good, intrinsics={"front_1": [[1, 2], [3]]})
    report = validate.validate_episode(bad, allow_missing=True)
    assert any("must be a 3x4 K matrix" in e for e in report.errors)


def test_validator_checks_dexmate_extrinsics_without_intrinsics(tmp_path):
    ep = write_dexmate_episode(tmp_path, **SMALL)
    _set_attrs(ep, extrinsics={"left": [[float("inf")] * 4] * 4})
    report = validate.validate_episode(ep, allow_missing=True)
    assert any("extrinsics['left'] contains NaN or inf" in e for e in report.errors)
