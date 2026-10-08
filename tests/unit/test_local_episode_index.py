"""``LocalEpisodeResolver`` filters from ``_episode_index.json`` when the folder
has one, without opening the episodes."""

from __future__ import annotations

import json
from unittest import mock

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LOCAL_EPISODE_INDEX,
    LocalEpisodeResolver,
)


def test_index_rows_drive_the_filter_and_no_episode_is_opened(tmp_path):
    index = {
        "a": {"embodiment": "human_bimanual", "task": "fold_tshirt", "user_id": "u1"},
        "b": {"embodiment": "human_bimanual", "task": "fold_tshirt", "user_id": "u2"},
        "c": {"embodiment": "human_bimanual", "task": "hang_clothes", "user_id": "u1"},
    }
    (tmp_path / LOCAL_EPISODE_INDEX).write_text(json.dumps(index))
    filters = DatasetFilter(
        filter_lambdas=[
            "lambda row: row.get('task') == 'fold_tshirt'",
            "lambda row: row.get('user_id') == 'u1'",
        ]
    )
    with mock.patch("zarr.open_group", side_effect=AssertionError("opened")):
        paths = LocalEpisodeResolver._get_local_filtered_paths(
            tmp_path, filters, expected_embodiment="human_bimanual"
        )
    assert paths == [(str(tmp_path / "a.zarr"), "a")]


def test_pins_are_checked_against_the_index(tmp_path):
    (tmp_path / LOCAL_EPISODE_INDEX).write_text(
        json.dumps({"a": {"embodiment": "human_bimanual"}})
    )
    paths = LocalEpisodeResolver._get_local_filtered_paths(
        tmp_path,
        DatasetFilter(episode_hashes=["a"]),
        expected_embodiment="human_bimanual",
    )
    assert paths == [(str(tmp_path / "a.zarr"), "a")]


def test_stage_versions_come_from_the_index_rows(tmp_path):
    rows = {"a": {"total_frames": 10}, "b": {"total_frames": 20}}
    (tmp_path / LOCAL_EPISODE_INDEX).write_text(json.dumps(rows))
    resolver = LocalEpisodeResolver(tmp_path)
    v = resolver._stage_versions({"a", "b", "missing"})
    assert set(v) == {"a", "b"} and v["a"] != v["b"]
    assert resolver._stage_versions({"a"}) == {"a": v["a"]}
    rows["a"]["total_frames"] = 11
    (tmp_path / LOCAL_EPISODE_INDEX).write_text(json.dumps(rows))
    assert resolver._stage_versions({"a"})["a"] != v["a"]
