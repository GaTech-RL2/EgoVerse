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


def test_dataset_filter_cache_key_reflects_contents() -> None:
    a = DatasetFilter(filter_lambdas=["lambda row: True"], episode_hashes=["y", "x"])
    b = DatasetFilter(filter_lambdas=["lambda row: True"], episode_hashes=["x", "y"])
    c = DatasetFilter(filter_lambdas=["lambda row: False"], episode_hashes=["x", "y"])
    assert a.cache_key() == b.cache_key()
    assert a.cache_key() != c.cache_key()
    assert hash(a.cache_key())


def test_s3_resolver_filters_dataframe_with_dataset_filter(monkeypatch) -> None:
    zarr_dataset_multi.clear_resolve_cache()
    df = pd.DataFrame(
        [
            {
                "episode_hash": "match",
                "zarr_processed_path": "s3://rldb/processed/match/",
                "task": "fold_clothes",
                "embodiment": "human_bimanual",
                "is_deleted": False,
            },
            {
                "episode_hash": "fallback",
                "zarr_processed_path": "s3://rldb/processed/fallback/",
                "task": "fold_clothes",
                "embodiment": "human_bimanual",
                "is_deleted": False,
            },
            {
                "episode_hash": "deleted",
                "zarr_processed_path": "s3://rldb/processed/deleted/",
                "task": "fold_clothes",
                "embodiment": "human_bimanual",
                "is_deleted": True,
            },
            {
                "episode_hash": "empty-path",
                "zarr_processed_path": "",
                "task": "fold_clothes",
                "embodiment": "human_bimanual",
                "is_deleted": False,
            },
        ]
    )
    monkeypatch.setattr(zarr_dataset_multi, "create_default_engine", lambda: object())
    monkeypatch.setattr(zarr_dataset_multi, "episode_table_to_df", lambda engine: df)

    filters = DatasetFilter(
        filter_lambdas=[
            "lambda row: row['embodiment'] == 'human_bimanual'",
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
        tmp_path, "episode_a", embodiment="human_bimanual", task="fold_clothes"
    )
    _write_episode(
        tmp_path,
        "episode_c",
        embodiment="human_bimanual",
        task="fold_clothes",
        is_deleted=True,
    )
    _write_episode(
        tmp_path, "episode_d", embodiment="eva_bimanual", task="fold_clothes"
    )

    filters = DatasetFilter(
        filter_lambdas=["lambda row: row['embodiment'] == 'human_bimanual'"]
    )

    paths = zarr_dataset_multi.LocalEpisodeResolver._get_local_filtered_paths(
        tmp_path,
        filters=filters,
    )

    assert [episode_hash for _, episode_hash in paths] == ["episode_a"]


def test_sync_s3_parser_accepts_named_filter_key() -> None:
    filters = parse_dataset_filter_key("aria-fold-clothes")

    assert isinstance(filters, DatasetFilter)
    assert filters.matches({"embodiment": "aria", "task": "fold_clothes"})
    assert not filters.matches({"embodiment": "human_bimanual", "task": "fold_clothes"})


def test_sync_s3_parser_rejects_unknown_filter_key() -> None:
    with pytest.raises(ValueError, match="Available filter keys"):
        parse_dataset_filter_key("does-not-exist")


def test_dataset_filter_episode_hashes_pin_rows() -> None:
    filters = DatasetFilter(episode_hashes=["a", "b"])

    assert filters.episode_hashes == frozenset({"a", "b"})
    assert filters.matches({"episode_hash": "a"})
    assert not filters.matches({"episode_hash": "c"})
    assert not filters.matches({"episode_hash": "a", "is_deleted": True})


def test_dataset_filter_episode_hashes_combine_with_lambdas() -> None:
    filters = DatasetFilter(
        filter_lambdas=["lambda row: row['task'] == 'fold'"], episode_hashes=["a"]
    )

    assert filters.matches({"episode_hash": "a", "task": "fold"})
    assert not filters.matches({"episode_hash": "a", "task": "stack"})
    assert not filters.matches({"episode_hash": "b", "task": "fold"})


def test_dataset_filter_empty_episode_hashes_means_no_pin() -> None:
    assert DatasetFilter(episode_hashes=[]).matches({"episode_hash": "anything"})
    assert "episode_hashes" in repr(DatasetFilter(episode_hashes=["a"]))


def test_dataset_filter_single_string_pin_is_one_hash() -> None:
    filters = DatasetFilter(episode_hashes="2026-04-30-09-41-51-255837")

    assert filters.episode_hashes == {"2026-04-30-09-41-51-255837"}
