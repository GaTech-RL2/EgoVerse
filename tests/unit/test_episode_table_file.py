"""EGOVERSE_EPISODE_TABLE: the S3 resolver reads the episode table from a
pickled snapshot instead of SQL, memoized per path."""

import pandas as pd

from egomimic.rldb.zarr import zarr_dataset_multi as zdm


def test_episode_table_from_pickle(tmp_path, monkeypatch):
    df = pd.DataFrame({"episode_hash": ["a"], "lab": ["abc_sim"]})
    path = tmp_path / "episodes.pkl"
    df.to_pickle(path)
    monkeypatch.setenv("EGOVERSE_EPISODE_TABLE", str(path))
    monkeypatch.setattr(zdm, "create_default_engine", lambda: (_ for _ in ()).throw(AssertionError("SQL touched")))
    assert list(zdm._episode_table().episode_hash) == ["a"]
    assert list(zdm._episode_table().episode_hash) == ["a"]  # second call: still no SQL
