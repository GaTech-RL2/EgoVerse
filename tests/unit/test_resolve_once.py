from unittest.mock import Mock

import pandas as pd
from fixtures.recipes import (
    RECIPES,
    common_overrides,
    cpu_trainer_overrides,
    hpt_small_overrides,
)
from fixtures.synthetic_episodes import write_episode
from fixtures.train_harness import compose_recipe, hermetic_env, write_fixtures

import egomimic.trainHydra as train_hydra
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.resolve_memo import resolve_once
from egomimic.rldb.zarr import zarr_dataset_multi as zdm

TABLE = pd.DataFrame(
    [
        {
            "episode_hash": "a",
            "zarr_processed_path": "s3://rldb/p/a/",
            "embodiment": "human_bimanual",
            "is_deleted": False,
            "task": "fold",
        },
        {
            "episode_hash": "b",
            "zarr_processed_path": "s3://rldb/p/b/",
            "embodiment": "human_bimanual",
            "is_deleted": False,
            "task": "stack",
        },
    ]
)


def _fold() -> DatasetFilter:
    return DatasetFilter(filter_lambdas=["lambda row: row['task'] == 'fold'"])


def _stack() -> DatasetFilter:
    return DatasetFilter(filter_lambdas=["lambda row: row['task'] == 'stack'"])


def _count_table_pulls(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(zdm, "create_default_engine", lambda: object())
    monkeypatch.setattr(
        zdm, "episode_table_to_df", lambda engine: calls.append(1) or TABLE
    )
    monkeypatch.setattr(
        zdm.S3EpisodeResolver, "_sync_s3_to_local", classmethod(lambda cls, **kw: None)
    )
    return calls


def _count_local_listings(monkeypatch) -> Mock:
    spy = Mock(wraps=zdm.LocalEpisodeResolver._get_local_filtered_paths)
    monkeypatch.setattr(zdm.LocalEpisodeResolver, "_get_local_filtered_paths", spy)
    return spy


def test_table_pulled_once_per_scope(monkeypatch, tmp_path) -> None:
    calls = _count_table_pulls(monkeypatch)
    r = zdm.S3EpisodeResolver(tmp_path)

    with resolve_once():
        fold = r.resolve_paths(_fold())
        stack = r.resolve_paths(_stack())
        with resolve_once():  # nested scope joins the outer memo
            again = r.resolve_paths(_fold())
        assert r.resolve_paths(_fold()) == fold  # inner exit kept the memo

    assert [h for _, h in fold] == ["a"]
    assert [h for _, h in stack] == ["b"]
    assert again == fold
    assert len(calls) == 1

    with resolve_once():
        r.resolve_paths(_fold())
    assert len(calls) == 2  # a new scope starts fresh


def test_no_memo_outside_scope(monkeypatch, tmp_path) -> None:
    calls = _count_table_pulls(monkeypatch)
    r = zdm.S3EpisodeResolver(tmp_path)
    with resolve_once():
        r.resolve_paths(_fold())
    r.resolve_paths(_fold())
    r.resolve_paths(_fold())
    assert len(calls) == 3


def test_local_listing_not_stale_outside_scope(monkeypatch, tmp_path) -> None:
    write_episode(tmp_path, "aria", seed=0)
    spy = _count_local_listings(monkeypatch)
    r = zdm.LocalEpisodeResolver(tmp_path)
    with resolve_once():
        assert {h for _, h in r.resolve_paths()} == {"aria_00"}
    write_episode(tmp_path, "aria", seed=1)  # converted after the first resolve
    assert {h for _, h in r.resolve_paths()} == {"aria_00", "aria_01"}
    assert spy.call_count == 2


def test_local_resolve_paths_memoized_and_load_is_fresh(monkeypatch, tmp_path) -> None:
    write_episode(tmp_path, "aria", seed=0)
    write_episode(tmp_path, "aria", seed=1)
    spy = _count_local_listings(monkeypatch)
    r1 = zdm.LocalEpisodeResolver(tmp_path, key_map=None)
    r2 = zdm.LocalEpisodeResolver(tmp_path, key_map={"norm_mode": True})
    pins = DatasetFilter(episode_hashes=["aria_00", "aria_01"])
    with resolve_once():
        d1 = r1.resolve(filters=pins)
        d2 = r2.resolve(filters=pins)
    assert set(d1) == set(d2) == {"aria_00", "aria_01"}
    # load() runs per resolver (keymap differs)
    assert d1["aria_00"] is not d2["aria_00"]
    assert spy.call_count == 1


def test_train_resolves_each_dataset_once(tmp_path, monkeypatch) -> None:
    hermetic_env(monkeypatch)
    recipe = RECIPES[("eva", "hpt")]
    data, out, hashes = write_fixtures(tmp_path, "eva")
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(2)
        + hpt_small_overrides(recipe.embodiment),
        out,
    )
    spy = _count_local_listings(monkeypatch)
    _, objects = train_hydra.train(cfg)
    assert spy.call_count == 1  # train + valid + norm-stat copy share one resolution
    dm = objects["datamodule"]
    assert (
        set(dm.train_datasets["eva_bimanual"].datasets)
        == set(dm.valid_datasets["eva_bimanual"].datasets)
        == set(hashes)
    )
