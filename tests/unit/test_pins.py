import logging

import pandas as pd
import pytest
import zarr
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
from egomimic.rldb.zarr import zarr_dataset_multi
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    PinError,
    S3EpisodeResolver,
)

TABLE = pd.DataFrame(
    [
        {
            "episode_hash": "h-ok",
            "zarr_processed_path": "s3://rldb/p/h-ok/",
            "embodiment": "human_bimanual",
            "is_deleted": False,
        },
        {
            "episode_hash": "h-deleted",
            "zarr_processed_path": "s3://rldb/p/h-deleted/",
            "embodiment": "human_bimanual",
            "is_deleted": True,
        },
        {
            "episode_hash": "h-nopath",
            "zarr_processed_path": "",
            "embodiment": "human_bimanual",
            "is_deleted": False,
        },
        {
            "episode_hash": "e-ok",
            "zarr_processed_path": "s3://rldb/p/e-ok/",
            "embodiment": "eva_bimanual",
            "is_deleted": False,
        },
    ]
)


@pytest.fixture
def fake_table(monkeypatch):
    zarr_dataset_multi.clear_resolve_cache()
    monkeypatch.setattr(zarr_dataset_multi, "create_default_engine", lambda: object())
    monkeypatch.setattr(zarr_dataset_multi, "episode_table_to_df", lambda engine: TABLE)


def test_valid_pin_resolves_only_that_episode(fake_table) -> None:
    paths = S3EpisodeResolver._get_filtered_paths(
        DatasetFilter(episode_hashes=["h-ok"]), expected_embodiment="human_bimanual"
    )
    assert paths == [("s3://rldb/p/h-ok/", "h-ok")]


def test_bad_pins_are_reported_together(fake_table) -> None:
    with pytest.raises(PinError) as exc:
        S3EpisodeResolver._get_filtered_paths(
            DatasetFilter(
                episode_hashes=["h-ok", "h-deleted", "h-nopath", "h-missing", "e-ok"]
            ),
            expected_embodiment="human_bimanual",
        )
    msg = str(exc.value)
    assert "4 pinned episode(s) invalid for dataset 'human_bimanual'" in msg
    assert "h-missing: not in app.episodes" in msg
    assert "h-deleted: is_deleted" in msg
    assert "h-nopath: empty zarr_processed_path" in msg
    assert "e-ok: embodiment 'eva_bimanual' != dataset 'human_bimanual'" in msg
    assert "h-ok" not in msg.split("invalid")[1]


def test_embodiment_check_skipped_without_dataset_name(fake_table) -> None:
    paths = S3EpisodeResolver._get_filtered_paths(
        DatasetFilter(episode_hashes=["e-ok"])
    )
    assert [h for _, h in paths] == ["e-ok"]


def test_unpinned_embodiment_mismatch_only_warns(fake_table, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        paths = S3EpisodeResolver._get_filtered_paths(
            DatasetFilter(), expected_embodiment="human_bimanual"
        )
    assert {h for _, h in paths} == {"h-ok", "e-ok"}
    assert any(
        "e-ok" in r.message and "eva_bimanual" in r.message for r in caplog.records
    )


def _attrs_only_episode(root, name, **attrs):
    zarr.open_group(str(root / f"{name}.zarr"), mode="w").attrs.update(attrs)


def test_local_pin_missing_dir_is_error(tmp_path) -> None:
    _attrs_only_episode(tmp_path, "l-ok", embodiment="human_bimanual")
    with pytest.raises(PinError, match="l-missing: not in local directory"):
        LocalEpisodeResolver._get_local_filtered_paths(
            tmp_path,
            DatasetFilter(episode_hashes=["l-ok", "l-missing"]),
            expected_embodiment="human_bimanual",
        )


def test_local_pin_embodiment_mismatch_is_error(tmp_path) -> None:
    _attrs_only_episode(tmp_path, "l-eva", embodiment="eva_bimanual")
    with pytest.raises(
        PinError, match="l-eva: embodiment 'eva_bimanual' != dataset 'human_bimanual'"
    ):
        LocalEpisodeResolver._get_local_filtered_paths(
            tmp_path,
            DatasetFilter(episode_hashes=["l-eva"]),
            expected_embodiment="human_bimanual",
        )


def test_local_unpinned_mismatch_warns(tmp_path, caplog) -> None:
    _attrs_only_episode(tmp_path, "l-eva", embodiment="eva_bimanual")
    _attrs_only_episode(tmp_path, "l-hum", embodiment="human_bimanual")
    with caplog.at_level(logging.WARNING):
        paths = LocalEpisodeResolver._get_local_filtered_paths(
            tmp_path, DatasetFilter(), expected_embodiment="human_bimanual"
        )
    assert {h for _, h in paths} == {"l-eva", "l-hum"}
    assert any("l-eva" in r.message for r in caplog.records)


def test_from_resolver_forwards_dataset_name_as_expected_embodiment(tmp_path) -> None:
    write_episode(tmp_path, "eva", seed=0)  # hash eva_00, embodiment eva_bimanual
    resolver = LocalEpisodeResolver(tmp_path)
    ok = MultiDataset._from_resolver(
        resolver,
        filters=DatasetFilter(episode_hashes=["eva_00"]),
        dataset_name="eva_bimanual",
        mode="total",
    )
    assert set(ok.datasets) == {"eva_00"}
    with pytest.raises(
        PinError, match="eva_00: embodiment 'eva_bimanual' != dataset 'human_bimanual'"
    ):
        MultiDataset._from_resolver(
            resolver,
            filters=DatasetFilter(episode_hashes=["eva_00"]),
            dataset_name="human_bimanual",
            mode="total",
        )


def test_from_resolver_skips_check_for_non_embodiment_names(tmp_path) -> None:
    write_episode(tmp_path, "eva", seed=0)
    ds = MultiDataset._from_resolver(
        LocalEpisodeResolver(tmp_path),
        filters=DatasetFilter(episode_hashes=["eva_00"]),
        dataset_name="my_mixture",
        mode="total",
    )
    assert set(ds.datasets) == {"eva_00"}


def test_train_fails_fast_on_bad_pin(tmp_path, monkeypatch) -> None:
    hermetic_env(monkeypatch)
    recipe = RECIPES[("eva", "hpt")]
    data, out, _ = write_fixtures(tmp_path, "eva")
    overrides = common_overrides(
        recipe.embodiment,
        data,
        out,
        batch_size=2,
        num_workers=0,
        episode_hashes=("eva_00", "nope"),
    )
    cfg = compose_recipe(
        recipe,
        overrides + cpu_trainer_overrides(2) + hpt_small_overrides(recipe.embodiment),
        out,
    )
    with pytest.raises(PinError, match="nope: not in local directory"):
        train_hydra.train(cfg)


def test_from_resolver_rejects_pin_excluded_by_filter_lambdas(tmp_path) -> None:
    write_episode(tmp_path, "eva", seed=0)
    write_episode(tmp_path, "eva", seed=1)
    with pytest.raises(PinError, match=r"1 pinned episode\(s\) not in dataset.*eva_01"):
        MultiDataset._from_resolver(
            LocalEpisodeResolver(tmp_path),
            filters=DatasetFilter(
                filter_lambdas=["lambda row: row['episode_hash'] != 'eva_01'"],
                episode_hashes=["eva_00", "eva_01"],
            ),
            dataset_name="eva_bimanual",
            mode="total",
        )


def test_from_resolver_rejects_pin_that_fails_to_load(tmp_path) -> None:
    write_episode(tmp_path, "eva", seed=0)
    # Readable attrs (so path resolution accepts the pin) but no arrays or
    # features, so constructing its ZarrDataset raises and the loader skips it.
    _attrs_only_episode(tmp_path, "eva_broken", embodiment="eva_bimanual")
    with pytest.raises(PinError, match="eva_broken"):
        MultiDataset._from_resolver(
            LocalEpisodeResolver(tmp_path),
            filters=DatasetFilter(episode_hashes=["eva_00", "eva_broken"]),
            dataset_name="eva_bimanual",
            mode="total",
        )
