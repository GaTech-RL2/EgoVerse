"""Per-episode norm-sample cache: splits share episodes, strides nest."""

import numpy as np
import zarr
from fixtures.synthetic_episodes import write_episode
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr import episode_norm_samples as ens
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset

T = 48


def _episodes(tmp_path, n=3):
    data = tmp_path / "data"
    for i in range(n):
        write_episode(data, "eva", T=T, seed=i)
    resolver = LocalEpisodeResolver(
        data,
        key_map=Eva.get_keymap(keymap_mode="cartesian", norm_mode=True),
        transform_list=Eva.get_transform_list(mode="cartesian_wristframe_6d"),
    )
    return MultiDataset._from_resolver(resolver, mode="total").datasets


def _stats(leaves, cache=None, **kw):
    ds = MultiDataset(datasets=dict(leaves), mode="total")
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"eva_bimanual": ds})
    stats.infer_shapes_from_batch(ds[0])
    stats.infer_norm_from_dataset(
        ds, "eva_bimanual", num_workers=0, episode_cache=cache, **kw
    )
    return next(iter(stats.norm_stats.values())), stats._norm_run_metadata


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        for name in a[key]:
            np.testing.assert_allclose(a[key][name], b[key][name], rtol=1e-5, atol=1e-6)


def _spy(monkeypatch):
    sampled = []
    orig = ens._sample_episodes

    def spy(root, leaves, *a, **k):
        sampled.append(sorted(leaves))
        return orig(root, leaves, *a, **k)

    monkeypatch.setattr(ens, "_sample_episodes", spy)
    return sampled


def test_every_frame_matches_the_whole_dataset_pass(tmp_path):
    leaves = _episodes(tmp_path)
    cached, meta = _stats(leaves, tmp_path / "cache", sample_frac=1.0)
    legacy, _ = _stats(leaves, None, sample_frac=1.0)
    _assert_same(cached, legacy)
    assert meta["stride"] == 1 and meta["frames"] == 3 * T


def test_a_sub_split_reads_only_the_cache(tmp_path, monkeypatch):
    leaves = _episodes(tmp_path)
    sampled = _spy(monkeypatch)
    _stats(leaves, tmp_path / "cache", sample_frac=1.0)
    assert sampled == [sorted(leaves)]

    sub = {h: leaves[h] for h in sorted(leaves)[:2]}
    from_cache, meta = _stats(sub, tmp_path / "cache", sample_frac=1.0)
    assert len(sampled) == 1 and meta["episodes_sampled"] == 0
    fresh, _ = _stats(sub, tmp_path / "fresh", sample_frac=1.0)
    _assert_same(from_cache, fresh)


def test_a_finer_file_serves_a_coarser_stride_but_not_the_reverse(
    tmp_path, monkeypatch
):
    leaves = _episodes(tmp_path)
    sampled = _spy(monkeypatch)
    _, meta = _stats(leaves, tmp_path / "cache", sample_frac=1.0)
    _, meta = _stats(leaves, tmp_path / "cache", sample_frac=0.25)
    assert meta["stride"] == 4 and len(sampled) == 1
    assert meta["frames"] == sum(len(ens.frame_indices(h, T, 4)) for h in leaves)

    _stats(leaves, tmp_path / "coarse", sample_frac=0.25)
    _stats(leaves, tmp_path / "coarse", sample_frac=1.0)
    assert len(sampled) == 3


def test_max_samples_caps_rows(tmp_path):
    leaves = _episodes(tmp_path)
    _, meta = _stats(leaves, tmp_path / "cache", sample_frac=1.0, max_samples=50)
    assert meta["frames"] == 50 and meta["stride"] == 2


def test_rewritten_episode_is_resampled(tmp_path, monkeypatch):
    leaves = _episodes(tmp_path)
    sampled = _spy(monkeypatch)
    _stats(leaves, tmp_path / "cache", sample_frac=1.0)
    h = sorted(leaves)[0]
    zarr.open_group(str(leaves[h].episode_path), mode="r+").attrs["note"] = "v2"
    _stats(leaves, tmp_path / "cache", sample_frac=1.0)
    assert sampled[1] == [h]


def test_frame_indices_nest_across_strides():
    for h in ("a", "696d9bf05c2b2b8722b2a08f"):
        for s in (1, 2, 4, 8):
            coarse = set(ens.frame_indices(h, 1000, 2 * s))
            assert coarse <= set(ens.frame_indices(h, 1000, s))


def test_stride_is_the_largest_that_meets_the_target():
    assert ens.stride_for(11_520_569, 200_000) == 32
    assert ens.stride_for(100, 100) == 1
    assert ens.stride_for(100, 50) == 2
    assert ens.stride_for(100, 0) == 64


def test_recipe_key_ignores_the_split_but_not_the_transforms():
    cfg = OmegaConf.create(
        {
            "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver",
            "resolver": {
                "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.S3EpisodeResolver",
                "folder_path": "/data/a",
                "transform_list": {"stride": 1},
            },
            "filters": {"episode_hashes": ["h1"]},
            "mode": "train",
        }
    )
    base = ens.recipe_key("human_bimanual", cfg)
    split = OmegaConf.merge(
        cfg,
        {
            "filters": {"episode_hashes": ["h1", "h2"]},
            "mode": "total",
            "resolver": {"folder_path": "/elsewhere"},
        },
    )
    assert ens.recipe_key("human_bimanual", split) == base
    strided = OmegaConf.merge(cfg, {"resolver": {"transform_list": {"stride": 3}}})
    assert ens.recipe_key("human_bimanual", strided) != base
    assert ens.recipe_key("eva_bimanual", cfg) != base
