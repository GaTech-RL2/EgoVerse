import logging
from pathlib import Path

import numpy as np
import pytest
import zarr
from fixtures.synthetic_episodes import write_episode

import egomimic
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr import schema
from egomimic.rldb.zarr import zarr_dataset_multi as zdm
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


def test_missing_version_is_legacy_zero() -> None:
    assert schema.check_format_version({"embodiment": "x"}) == "0.0"


def test_current_version_accepted() -> None:
    assert schema.check_format_version({"format_version": "1.0"}) == "1.0"
    assert schema.check_format_version({"format_version": "1.7"}) == "1.7"


def test_future_major_rejected() -> None:
    with pytest.raises(
        schema.SchemaVersionError, match="episode is v2.0, this egomimic reads v0-v1"
    ):
        schema.check_format_version({"format_version": "2.0"})


def test_malformed_version_rejected() -> None:
    with pytest.raises(schema.SchemaVersionError, match="malformed"):
        schema.check_format_version({"format_version": "one"})


def test_writer_stamps_version_and_provenance(tmp_path) -> None:
    path = write_episode(tmp_path, "aria", seed=0)
    attrs = dict(zarr.open_group(str(path), mode="r").attrs)
    assert attrs["format_version"] == schema.FORMAT_VERSION
    prov = attrs["provenance"]
    assert prov["writer"] == "egomimic.rldb.zarr.zarr_writer.ZarrWriter"
    assert prov["egomimic_version"]
    assert prov["created_at"].endswith("+00:00")
    assert prov["git_sha"] is None or len(prov["git_sha"]) == 40
    assert prov["converter"] is None and prov["source_uri"] is None


def test_writer_records_converter_and_override_cannot_clobber(tmp_path) -> None:
    T = 4
    K = np.array([[1.0, 0, 1, 0], [0, 1.0, 1, 0], [0, 0, 1.0, 0]])
    path = ZarrWriter.create_and_write(
        tmp_path / "ep.zarr",
        numeric_data={"left.obs_ee_pose": np.tile([0, 0, 0, 1, 0, 0, 0.0], (T, 1))},
        image_data={"images.front_1": np.zeros((T, 8, 8, 3), np.uint8)},
        embodiment="human_bimanual",
        intrinsics={"front_1": K},
        converter="tests.fake_converter",
        source_uri="file:///raw/ep",
        metadata_override={
            "format_version": "9.9",
            "provenance": {"writer": "evil"},
            "task_name": "x",
        },
    )
    attrs = dict(zarr.open_group(str(path), mode="r").attrs)
    assert attrs["format_version"] == schema.FORMAT_VERSION
    assert attrs["provenance"]["converter"] == "tests.fake_converter"
    assert attrs["provenance"]["source_uri"] == "file:///raw/ep"
    assert attrs["task_name"] == "x"


def _set_version(path, value) -> None:
    g = zarr.open_group(str(path), mode="r+")
    attrs = dict(g.attrs)
    if value is None:
        attrs.pop("format_version", None)
        g.attrs.clear()
        g.attrs.update(attrs)
    else:
        g.attrs["format_version"] = value


def test_legacy_episode_reads_with_one_warning(tmp_path, caplog) -> None:
    p0 = write_episode(tmp_path, "aria", seed=0)
    p1 = write_episode(tmp_path, "aria", seed=1)
    _set_version(p0, None)
    _set_version(p1, None)
    zdm._warn_legacy_once.cache_clear()
    with caplog.at_level(logging.WARNING):
        e0, e1 = zdm.ZarrEpisode(p0), zdm.ZarrEpisode(p1)
    assert e0.format_version == "0.0" and e1.format_version == "0.0"
    assert sum("no format_version" in r.message for r in caplog.records) == 1


def test_future_episode_fails_resolve(tmp_path) -> None:
    p = write_episode(tmp_path, "aria", seed=0)
    _set_version(p, "2.0")
    with pytest.raises(
        schema.SchemaVersionError, match=r"aria_00\.zarr: episode is v2\.0"
    ):
        zdm.LocalEpisodeResolver(tmp_path).resolve(
            filters=DatasetFilter(episode_hashes=["aria_00"])
        )


def test_validate_synthetic_episode_is_clean(tmp_path) -> None:
    p = write_episode(tmp_path, "eva", seed=0)
    rep = schema.validate_episode(p)
    assert rep.ok, rep.errors
    assert rep.warnings == []


def test_validate_reports_each_defect(tmp_path) -> None:
    p = write_episode(tmp_path, "aria", seed=0)
    g = zarr.open_group(str(p), mode="r+")
    attrs = dict(g.attrs)
    attrs["fps"] = 25
    attrs["total_frames"] = attrs["total_frames"] + 1000
    attrs["embodiment"] = "robot_from_mars"
    del attrs["task_name"]
    del attrs["intrinsics"]
    g.attrs.clear()
    g.attrs.update(attrs)
    rep = schema.validate_episode(p)
    msgs = "\n".join(rep.errors)
    assert "missing attr: task_name" in msgs
    assert "fps=25" in msgs
    assert "robot_from_mars" in msgs
    assert "intrinsics" in msgs
    assert "images.front_1: array length 100 < total_frames 1048" in msgs


def test_validate_legacy_is_warning_not_error(tmp_path) -> None:
    p = write_episode(tmp_path, "aria", seed=0)
    _set_version(p, None)
    rep = schema.validate_episode(p)
    assert rep.ok and any("format_version" in w for w in rep.warnings)


_DELETE = object()


def _set_attrs(path, **updates) -> None:
    g = zarr.open_group(str(path), mode="r+")
    attrs = dict(g.attrs)
    for k, v in updates.items():
        if v is _DELETE:
            attrs.pop(k, None)
        else:
            attrs[k] = v
    g.attrs.clear()
    g.attrs.update(attrs)


def test_validate_pose_missing_frame_markers_pass(tmp_path) -> None:
    p = write_episode(tmp_path, "aria", seed=0)
    pose = zarr.open_group(str(p), mode="r+")["left.obs_ee_pose"]
    pose[0] = np.zeros(7)  # Mecka undetected hand
    pose[1] = np.full(7, 1e9)  # Aria sentinel
    pose[2] = np.full(7, -1e8)
    rep = schema.validate_episode(p)
    assert rep.ok, rep.errors


@pytest.mark.parametrize(
    "row",
    [
        [0.1, 0.2, 0.3, np.nan, 0.0, 0.0, 0.0],  # NaN quaternion
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0],  # zero quaternion, real xyz
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9],  # partial zero / partial sentinel
        [1e9, 1e9, 1e9, 0.5, 0.0, 0.0, 0.0],  # sentinel xyz only
        [0.1, 0.2, 0.3, 1.0 + 2e-4, 0.0, 0.0, 0.0],
    ],
)
def test_validate_pose_bad_quaternion_is_error(tmp_path, row) -> None:
    p = write_episode(tmp_path, "aria", seed=0)
    pose = zarr.open_group(str(p), mode="r+")["left.obs_ee_pose"]
    pose[5] = row
    rep = schema.validate_episode(p)
    assert rep.errors == [
        "left.obs_ee_pose: 1 frames with non-unit quaternions (first: frame 5)"
    ]


def test_validate_malformed_attrs_are_errors_not_exceptions(tmp_path) -> None:
    p = write_episode(tmp_path, "aria", seed=0)
    features = dict(zarr.open_group(str(p), mode="r").attrs["features"])
    features["images.front_1"] = "jpeg"
    _set_attrs(
        p,
        intrinsics={"front_1": "bogus", "front_2": [[1, 2], [3]]},
        features=features,
        fps=_DELETE,
        total_frames="many",
        format_version=1,
    )
    pose = zarr.open_group(str(p), mode="r+")["left.obs_ee_pose"]
    pose[3] = [0.1, 0.2, 0.3, np.nan, 0.0, 0.0, 0.0]
    msgs = "\n".join(schema.validate_episode(p).errors)
    assert "intrinsics['front_1']: expected 3x4" in msgs
    assert "intrinsics['front_2']: expected 3x4" in msgs
    assert "features['images.front_1']: expected a dict, got str" in msgs
    assert "format_version=1, expected" in msgs
    assert "total_frames='many' is not an int" in msgs
    assert msgs.count("fps") == 1
    # checks that do not need total_frames still run
    assert (
        "left.obs_ee_pose: 1 frames with non-unit quaternions (first: frame 3)" in msgs
    )


@pytest.mark.parametrize("value", ["1.x", "-0.3", " 1.0", "1"])
def test_validate_rejects_malformed_version_string(tmp_path, value) -> None:
    p = write_episode(tmp_path, "aria", seed=0)
    _set_version(p, value)
    assert any("MAJOR.MINOR" in e for e in schema.validate_episode(p).errors)


def test_contributing_doc_schema_table_is_current() -> None:
    doc = (Path(egomimic.__file__).parent.parent / "CONTRIBUTING_DATA.md").read_text(
        encoding="utf-8"
    )
    begin, end = "<!-- schema:begin -->", "<!-- schema:end -->"
    body = doc.split(begin)[1].split(end)[0].strip()
    assert (
        body == schema.schema_markdown().strip()
    ), "CONTRIBUTING_DATA.md schema table is stale; run `egoverse schema` and paste it between the markers"
