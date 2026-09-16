"""Horizon-pooled norm stats and the constant-channel normalization guard."""

import numpy as np
import pytest
import torch
from fixtures.synthetic_episodes import write_episode
from hydra import compose, initialize_config_module

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr import norm_cache
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset


def _compose(name, overrides=()):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        return compose(config_name=name, overrides=list(overrides))


def test_hpt_eva_dims_match_the_wrist_frame_6d_layout():
    cfg = _compose("train_zarr_cartesian")
    dims = cfg.model.robomimic_model.dims.eva_bimanual
    assert (dims.proprio, dims.action) == (20, 20)


def _eva_stats(tmp_path, pool_horizon):
    for i in range(3):
        write_episode(tmp_path, "eva", seed=i)
    resolver = LocalEpisodeResolver(
        tmp_path,
        key_map=Eva.get_keymap(keymap_mode="cartesian", norm_mode=True),
        transform_list=Eva.get_transform_list(mode="cartesian_wristframe_6d"),
    )
    ds = MultiDataset._from_resolver(resolver, mode="total")
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"eva_bimanual": ds})
    stats.infer_shapes_from_batch(ds[0])
    stats.infer_norm_from_dataset(
        ds, "eva_bimanual", sample_frac=1.0, num_workers=0, pool_horizon=pool_horizon
    )
    return next(iter(stats.norm_stats.values()))


def test_pool_horizon_tiles_one_stat_per_channel(tmp_path):
    per_t = _eva_stats(tmp_path / "a", pool_horizon=False)
    pooled = _eva_stats(tmp_path / "b", pool_horizon=True)
    act_t, act_p = per_t["actions_cartesian"], pooled["actions_cartesian"]
    for name in act_t:
        assert act_p[name].shape == act_t[name].shape
        assert act_p[name].ndim == 2
        # every timestep carries the same per-channel value
        np.testing.assert_array_equal(
            act_p[name], act_p[name][:1].repeat(len(act_p[name]), 0)
        )
    # every timestep has equal weight in the pool, so a pooled quantile lies
    # between the smallest and largest per-timestep quantile of that channel
    for name in ("quantile_1", "quantile_99", "median"):
        lo = act_t[name].min(axis=0) - 1e-4
        hi = act_t[name].max(axis=0) + 1e-4
        assert np.all((act_p[name][0] >= lo) & (act_p[name][0] <= hi)), name
    # proprio has no horizon axis: identical either way
    for name, arr in per_t["observations.state.ee_pose"].items():
        np.testing.assert_allclose(
            pooled["observations.state.ee_pose"][name], arr, rtol=1e-6
        )


def test_pool_horizon_is_part_of_the_cache_key():
    args = ("eva_bimanual", {"h": "fp"}, {"resolver": {}}, 1.0)
    assert norm_cache.norm_cache_key(
        norm_cache.cache_inputs(*args, pool_horizon=True)
    ) != norm_cache.norm_cache_key(norm_cache.cache_inputs(*args))


# ------------------------------------------------- degenerate-range guard
@pytest.mark.parametrize("mode", ["zscore", "minmax", "quantile"])
def test_norm_constant_channel_maps_to_zero_and_back_to_its_centre(mode):
    from egomimic.utils.action_utils import (
        NORM_MIN_RANGE,
        _apply_norm_one,
        _apply_unnorm_one,
    )

    lo = np.array([0.0, -1.0, 2.0], dtype=np.float32)
    hi = np.array([0.0, 1.0, 2.0 + NORM_MIN_RANGE / 4], dtype=np.float32)
    stats = {
        "mean": 0.5 * (lo + hi),
        "std": hi - lo,
        "min": lo,
        "max": hi,
        "quantile_1": lo,
        "quantile_99": hi,
    }
    x = torch.tensor([[1e-3, 0.5, 2.0 + 1e-3]])
    n = _apply_norm_one(x, stats, mode)
    # channels 0 and 2 are (near-)constant: 0 whatever the offset, not 1e3
    assert n[0, 0].item() == 0.0 and n[0, 2].item() == 0.0
    back = _apply_unnorm_one(torch.full_like(n, 0.7), stats, mode)
    np.testing.assert_allclose(back[0, [0, 2]].numpy(), stats["mean"][[0, 2]])

    # the regular channel keeps the historical formula exactly
    if mode == "zscore":
        ref = (0.5 - stats["mean"][1]) / (stats["std"][1] + 1e-6)
    else:
        ref = 2.0 * (0.5 - lo[1]) / (hi[1] - lo[1] + 1e-6) - 1.0
    assert n[0, 1].item() == pytest.approx(float(ref), rel=1e-6)
    assert _apply_unnorm_one(n, stats, mode)[0, 1].item() == pytest.approx(0.5)


def test_multidataset_normalize_uses_the_guard():
    stats = {"quantile_1": np.zeros(2), "quantile_99": np.array([0.0, 1.0])}
    md = MultiDataset.__new__(MultiDataset)
    md.norm_mode = "quantile"
    n = md._apply_norm_one(torch.tensor([[5e-4, 0.5]]), stats)
    assert n.tolist() == [[0.0, 0.0]]
