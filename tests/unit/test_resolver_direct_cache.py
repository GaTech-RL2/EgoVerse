"""Explicit direct shared-cache reading preserves the default staging path."""

from pathlib import Path

import pytest

from egomimic.rldb.zarr import zarr_dataset_multi as zdm


class _Dataset:
    def __init__(self, path, *, key_map, transform_list):
        self.episode_path = Path(path)
        self.key_map = key_map
        self.transform_list = transform_list


@pytest.mark.parametrize(
    "resolver_cls", [zdm.LocalEpisodeResolver, zdm.S3EpisodeResolver]
)
def test_direct_cache_reads_only_selected_episodes_without_staging(
    tmp_path, monkeypatch, resolver_cls
):
    source = tmp_path / "shared"
    for name in ("b.zarr", "a", "unselected.zarr"):
        (source / name).mkdir(parents=True)
    stage = tmp_path / "stage-must-stay-absent"
    monkeypatch.setenv(zdm.STAGE_DIR_ENV, str(stage))

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Direct shared-cache reading must not stage or query staging versions"
        )

    monkeypatch.setattr(zdm, "stage_episodes", forbidden)
    monkeypatch.setattr(resolver_cls, "_stage_versions", forbidden)
    key_map = {"action": {"zarr_key": "raw_action"}}
    transforms = [object()]
    resolver = resolver_cls(
        source, key_map=key_map, transform_list=transforms, stage_to_local=False
    )
    resolver._dataset_class = _Dataset
    result = resolver.load(
        [("s3://rldb/example/b.zarr", "b"), ("s3://rldb/example/a", "a")]
    )

    assert list(result) == ["a", "b"]
    assert result["a"].episode_path == source / "a"
    assert result["b"].episode_path == source / "b.zarr"
    assert all(
        ds.key_map is key_map and ds.transform_list is transforms
        for ds in result.values()
    )
    assert not stage.exists()
    assert sorted(p.name for p in source.iterdir()) == [
        "a",
        "b.zarr",
        "unselected.zarr",
    ]


@pytest.mark.parametrize(
    "resolver_cls", [zdm.LocalEpisodeResolver, zdm.S3EpisodeResolver]
)
def test_resolver_still_stages_by_default(tmp_path, monkeypatch, resolver_cls):
    source, stage = tmp_path / "shared", tmp_path / "local"
    monkeypatch.setenv(zdm.STAGE_DIR_ENV, str(stage))
    resolver = resolver_cls(source)
    calls = []
    monkeypatch.setattr(resolver, "_stage_versions", lambda names: {"a": "v1"})

    def fake_stage(src, dst, names, versions):
        calls.append((src, dst, names, versions))
        return {"a"}

    def fake_load(*, search_path, valid_folder_names):
        assert search_path == zdm.staged_episode_dir(stage, source)
        assert valid_folder_names == {"a"}
        return {"a": "staged-dataset"}

    monkeypatch.setattr(zdm, "stage_episodes", fake_stage)
    monkeypatch.setattr(resolver, "_load_zarr_datasets", fake_load)
    assert resolver.load([("s3://rldb/example/a", "a")]) == {"a": "staged-dataset"}
    assert calls == [(source, stage, {"a"}, {"a": "v1"})]


def test_local_direct_and_staged_paths_preserve_pins_metadata_and_samples(tmp_path):
    import json

    import torch
    import zarr
    from fixtures.synthetic_episodes import write_episode

    from egomimic.rldb.embodiment.eva import Eva
    from egomimic.rldb.filters import DatasetFilter

    source = tmp_path / "shared"
    paths = [write_episode(source, "eva", T=24, H=16, W=16, seed=i) for i in range(3)]
    index = {
        p.name.removesuffix(".zarr"): dict(zarr.open_group(str(p), mode="r").attrs)
        for p in paths
    }
    (source / zdm.LOCAL_EPISODE_INDEX).write_text(json.dumps(index))
    selected = sorted(index)[:2]
    filters = DatasetFilter(episode_hashes=selected)
    kwargs = {
        "key_map": Eva.get_keymap(keymap_mode="cartesian", norm_mode=True),
        "transform_list": Eva.get_transform_list(mode="cartesian_wristframe_6d"),
    }
    direct = zdm.LocalEpisodeResolver(source, stage_to_local=False, **kwargs)
    staged = zdm.LocalEpisodeResolver(source, **kwargs)
    assert direct.resolve_paths(filters, "eva_bimanual") == staged.resolve_paths(
        filters, "eva_bimanual"
    )
    direct_leaves = direct.resolve(filters=filters, expected_embodiment="eva_bimanual")
    staged_leaves = staged.resolve(filters=filters, expected_embodiment="eva_bimanual")
    assert list(direct_leaves) == list(staged_leaves) == selected
    for episode_id in selected:
        a, b = direct_leaves[episode_id], staged_leaves[episode_id]
        assert a.episode_path.parent == source
        assert b.episode_path.parent != source
        assert len(a) == len(b) == 24
        assert dict(zarr.open_group(str(a.episode_path), mode="r").attrs) == dict(
            zarr.open_group(str(b.episode_path), mode="r").attrs
        )
        for frame in (0, 12, 23):
            direct_sample, staged_sample = a[frame], b[frame]
            assert direct_sample.keys() == staged_sample.keys()
            for key, value in direct_sample.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        value, staged_sample[key], rtol=0, atol=0
                    )
                else:
                    assert value == staged_sample[key]


def test_fold_moving_common_all_heads_read_verified_shared_cache(compose_resolve):
    cfg = compose_resolve("fold_moving_common", [])
    for head in ("train", "valid", "extra_valid", "unseen_op_valid"):
        resolver = cfg.data[f"{head}_datasets"].human_bimanual.resolver
        assert resolver.stage_to_local is False
        assert resolver.folder_path.endswith("/canonical_zarr")
