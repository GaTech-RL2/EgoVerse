"""Test status writing, legacy defaults, validation, and resolver filtering."""

import numpy as np
import pytest
import zarr

from egomimic.rldb.zarr.episode_attrs import (
    DATA_STATUS_COMPLETE,
    DATA_STATUS_STRUCTURAL_SAMPLE,
    data_status,
    is_complete,
)
from egomimic.rldb.zarr.validate import ERROR, WARNING, validate_episode
from egomimic.rldb.zarr.zarr_dataset_multi import (
    EpisodeResolver,
    ZarrDataset,
    ZarrEpisode,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

K = np.array([[200.0, 0.0, 160.0, 0.0], [0.0, 200.0, 120.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
LENGTH = 4


def _write(path, **kwargs) -> None:
    ZarrWriter.create_and_write(
        episode_path=path,
        numeric_data={"left.obs_gripper": np.zeros((LENGTH, 1))},
        embodiment="eva_bimanual",
        chunk_timesteps=LENGTH,
        intrinsics={"front_1": K},
        **kwargs,
    )


def test_an_episode_without_the_attribute_reads_as_complete() -> None:
    assert data_status({}) == DATA_STATUS_COMPLETE
    assert is_complete({})
    assert not is_complete({"data_status": DATA_STATUS_STRUCTURAL_SAMPLE})


def test_the_writer_records_the_status(tmp_path) -> None:
    _write(tmp_path / "complete.zarr")
    _write(tmp_path / "sample.zarr", data_status=DATA_STATUS_STRUCTURAL_SAMPLE)

    assert ZarrEpisode(tmp_path / "complete.zarr").data_status == DATA_STATUS_COMPLETE
    assert (
        ZarrEpisode(tmp_path / "sample.zarr").data_status
        == DATA_STATUS_STRUCTURAL_SAMPLE
    )


def test_the_writer_rejects_an_unknown_status(tmp_path) -> None:
    with pytest.raises(ValueError, match="data_status must be one of"):
        _write(tmp_path / "bad.zarr", data_status="mostly_done")


def test_the_resolver_refuses_a_structural_sample(tmp_path, caplog) -> None:
    _write(tmp_path / "complete.zarr")
    _write(tmp_path / "sample.zarr", data_status=DATA_STATUS_STRUCTURAL_SAMPLE)
    resolver = EpisodeResolver(tmp_path, key_map={})

    with caplog.at_level("WARNING"):
        datasets = resolver._load_zarr_datasets(
            search_path=tmp_path, valid_folder_names={"complete", "sample"}
        )

    assert set(datasets) == {"complete"}
    assert "not 'complete'" in caplog.text


def test_the_dataset_exposes_the_status(tmp_path) -> None:
    _write(tmp_path / "sample.zarr", data_status=DATA_STATUS_STRUCTURAL_SAMPLE)

    dataset = ZarrDataset(Episode_path=tmp_path / "sample.zarr", key_map={})

    assert dataset.data_status == DATA_STATUS_STRUCTURAL_SAMPLE


def test_the_validator_reads_and_checks_the_status(tmp_path) -> None:
    _write(tmp_path / "sample.zarr", data_status=DATA_STATUS_STRUCTURAL_SAMPLE)
    report = validate_episode(tmp_path / "sample.zarr")
    assert next(
        f.level for f in report.findings if f.check == "attrs.data_status"
    ) == "ok"

    _write(tmp_path / "legacy.zarr")
    store = zarr.open_group(str(tmp_path / "legacy.zarr"), mode="a")
    del store.attrs["data_status"]
    required = validate_episode(tmp_path / "legacy.zarr")
    waived = validate_episode(
        tmp_path / "legacy.zarr", requirements={"data_status": False}
    )
    statuses = {
        "required": next(
            f.level for f in required.findings if f.check == "attrs.data_status"
        ),
        "waived": next(
            f.level for f in waived.findings if f.check == "attrs.data_status"
        ),
    }
    assert statuses == {"required": ERROR, "waived": WARNING}


def test_the_validator_rejects_an_unknown_status(tmp_path) -> None:
    _write(tmp_path / "odd.zarr")
    store = zarr.open_group(str(tmp_path / "odd.zarr"), mode="a")
    store.attrs["data_status"] = "in_review"

    report = validate_episode(
        tmp_path / "odd.zarr", requirements={"data_status": False}
    )

    finding = next(f for f in report.findings if f.check == "attrs.data_status")
    assert finding.level == ERROR
    assert "expected one of" in finding.message
