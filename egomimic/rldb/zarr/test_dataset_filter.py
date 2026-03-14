import pandas as pd
import pytest
import zarr

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr import zarr_dataset_multi
from egomimic.scripts.data_download.sync_s3 import parse_dataset_filter_key


def _write_episode(root, name: str, **attrs) -> None:
    group = zarr.open_group(str(root / f"{name}.zarr"), mode="w")
    group.attrs.update(attrs)


def test_dataset_filter_matches_rows_and_excludes_deleted_by_default() -> None:
    filters = DatasetFilter(
        filter_lambdas=["lambda row: row['episode_hash'] == 'episode-1'"]
    )

    assert filters.matches({"episode_hash": "episode-1"})
    assert not filters.matches({"episode_hash": "episode-1", "is_deleted": True})
    assert not filters.matches({"episode_hash": "episode-2"})


def test_dataset_filter_empty_list_matches_all_non_deleted_rows() -> None:
    filters = DatasetFilter()

    assert filters.matches({"episode_hash": "episode-1"})
    assert not filters.matches({"episode_hash": "episode-1", "is_deleted": True})


def test_dataset_filter_init_rejects_invalid_filter_and_prints_it(capsys) -> None:
    with pytest.raises(ValueError, match="Invalid filter"):
        DatasetFilter(filter_lambdas=["lambda row:"])

    captured = capsys.readouterr()
    assert "Invalid filter: lambda row:" in captured.err


def test_dataset_filter_matches_requires_bool_result() -> None:
    filters = DatasetFilter(filter_lambdas=["lambda row: 1"])

    with pytest.raises(TypeError, match="Filter must return bool"):
        filters.matches({"episode_hash": "episode-1"})


def test_s3_resolver_filters_dataframe_with_dataset_filter(monkeypatch) -> None:
    df = pd.DataFrame(
        [
            {
                "episode_hash": "match",
                "zarr_processed_path": "s3://rldb/processed/match/",
                "task": "fold_clothes",
                "robot_name": "aria_bimanual",
                "is_deleted": False,
            },
            {
                "episode_hash": "fallback",
                "zarr_processed_path": "s3://rldb/processed/fallback/",
                "task": "fold_clothes",
                "robot_name": None,
                "embodiment": "aria_bimanual",
                "is_deleted": False,
            },
            {
                "episode_hash": "deleted",
                "zarr_processed_path": "s3://rldb/processed/deleted/",
                "task": "fold_clothes",
                "robot_name": "aria_bimanual",
                "is_deleted": True,
            },
            {
                "episode_hash": "empty-path",
                "zarr_processed_path": "",
                "task": "fold_clothes",
                "robot_name": "aria_bimanual",
                "is_deleted": False,
            },
        ]
    )
    monkeypatch.setattr(zarr_dataset_multi, "create_default_engine", lambda: object())
    monkeypatch.setattr(zarr_dataset_multi, "episode_table_to_df", lambda engine: df)

    filters = DatasetFilter(
        filter_lambdas=[
            "lambda row: row['robot_name'] == 'aria_bimanual'",
            "lambda row: row['task'] == 'fold_clothes'",
        ]
    )

    paths = zarr_dataset_multi.S3EpisodeResolver._get_filtered_paths(filters=filters)

    assert paths == [
        ("s3://rldb/processed/match/", "match"),
        ("s3://rldb/processed/fallback/", "fallback"),
    ]


def test_local_resolver_filters_local_metadata_with_dataset_filter(tmp_path) -> None:
    _write_episode(
        tmp_path, "episode_a", embodiment="aria_bimanual", task="fold_clothes"
    )
    _write_episode(
        tmp_path, "episode_b", robot_type="aria_bimanual", task="fold_clothes"
    )
    _write_episode(
        tmp_path,
        "episode_c",
        robot_name="aria_bimanual",
        task="fold_clothes",
        is_deleted=True,
    )
    _write_episode(
        tmp_path, "episode_d", robot_name="eva_bimanual", task="fold_clothes"
    )

    filters = DatasetFilter(
        filter_lambdas=["lambda row: row['robot_name'] == 'aria_bimanual'"]
    )

    paths = zarr_dataset_multi.LocalEpisodeResolver._get_local_filtered_paths(
        tmp_path,
        filters=filters,
    )

    assert [episode_hash for _, episode_hash in paths] == ["episode_a", "episode_b"]


def test_sync_s3_parser_accepts_named_filter_key() -> None:
    filters = parse_dataset_filter_key("aria-fold-clothes")

    assert isinstance(filters, DatasetFilter)
    assert filters.matches({"embodiment": "aria", "task": "fold_clothes"})
    assert not filters.matches({"embodiment": "aria_bimanual", "task": "fold_clothes"})


def test_sync_s3_parser_rejects_unknown_filter_key() -> None:
    with pytest.raises(ValueError, match="Available filter keys"):
        parse_dataset_filter_key("does-not-exist")
