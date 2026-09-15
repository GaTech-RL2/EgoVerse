"""Norm-stat inference must not let a NaN/Inf or 1e9-sentinel sample poison the stats."""

import numpy as np
import pytest
import zarr
from fixtures.synthetic_episodes import write_episode

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset


def test_drop_invalid_rows_keeps_finite_samples_only():
    X = np.ones((5, 100, 14))
    X[1, 3, 7] = np.nan  # one NaN cell in sample 1
    X[4, 0, 0] = np.inf  # one Inf cell in sample 4
    kept = MultiDataset._drop_invalid_rows(X, "actions_cartesian")
    assert kept.shape == (3, 100, 14)
    assert np.isfinite(kept).all()


def test_drop_invalid_rows_drops_missing_frame_sentinel():
    X = np.ones((4, 100, 14))
    X[2, 10:20] = 1e9  # an undetected hand in part of the chunk
    X[3, 0, 3] = -1e9
    kept = MultiDataset._drop_invalid_rows(X, "actions_cartesian")
    assert kept.shape == (2, 100, 14)
    assert np.abs(kept).max() < 1e8


def test_drop_invalid_rows_is_noop_on_clean_input():
    X = np.random.default_rng(0).normal(size=(6, 14))
    assert MultiDataset._drop_invalid_rows(X, "k") is X


def test_drop_invalid_rows_raises_when_nothing_is_finite():
    X = np.full((3, 2), np.nan)
    with pytest.raises(ValueError, match="every collected norm sample is non-finite"):
        MultiDataset._drop_invalid_rows(X, "k")


def test_stats_are_finite_after_dropping_bad_samples():
    X = np.random.default_rng(1).normal(size=(50, 100, 14))
    X[7] = np.nan  # a whole bad sample (e.g. an all-NaN gripper command)
    stats = MultiDataset._compute_stats_for_array(
        MultiDataset._drop_invalid_rows(X, "actions_cartesian")
    )
    for name, arr in stats.items():
        assert np.isfinite(arr).all(), name


def test_infer_norm_from_dataset_ignores_episode_with_nan_gripper(tmp_path, caplog):
    """The real failure: one episode's right.cmd_gripper is all NaN. Stats over
    the other episodes must stay finite (and the drop must be logged)."""
    for i in range(3):
        write_episode(tmp_path, "eva", seed=i)
    g = zarr.open_group(str(tmp_path / "eva_01.zarr"), mode="a")
    g["right.cmd_gripper"][:] = np.nan
    resolver = LocalEpisodeResolver(
        tmp_path,
        key_map=Eva.get_keymap(keymap_mode="cartesian", norm_mode=True),
        transform_list=Eva.get_transform_list(mode="cartesian"),
    )
    ds = MultiDataset._from_resolver(resolver, mode="total")
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"eva_bimanual": ds})
    stats.infer_shapes_from_batch(ds[0])
    with caplog.at_level("WARNING"):
        stats.infer_norm_from_dataset(
            ds, "eva_bimanual", sample_frac=1.0, num_workers=0
        )
    emb = int(ds[0]["embodiment"])
    for key, st in stats.norm_stats[emb].items():
        for name, arr in st.items():
            assert np.isfinite(arr).all(), (key, name)
    assert "dropping" in caplog.text and "non-finite" in caplog.text


def test_infer_norm_from_dataset_ignores_aria_sentinel_frames(tmp_path, caplog):
    """Aria writes 1e9 rows for an undetected hand; the transforms pass them
    through as ~1e9, which must not reach mean/std/min/max."""
    for i in range(3):
        write_episode(tmp_path, "aria", seed=i)
    g = zarr.open_group(str(tmp_path / "aria_01.zarr"), mode="a")
    pose = g["right.obs_ee_pose"][:]
    pose[5:9] = 1e9
    g["right.obs_ee_pose"][:] = pose
    resolver = LocalEpisodeResolver(
        tmp_path,
        key_map=Human.get_keymap(keymap_mode="cartesian", norm_mode=True),
        transform_list=Human.get_transform_list(mode="cartesian", stride=3),
    )
    ds = MultiDataset._from_resolver(resolver, mode="total")
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"human_bimanual": ds})
    stats.infer_shapes_from_batch(ds[0])
    with caplog.at_level("WARNING"):
        stats.infer_norm_from_dataset(
            ds, "human_bimanual", sample_frac=1.0, num_workers=0
        )
    emb = int(ds[0]["embodiment"])
    for key, st in stats.norm_stats[emb].items():
        for name, arr in st.items():
            assert np.isfinite(arr).all() and np.abs(arr).max() < 1e8, (key, name)
    assert "missing-frame sentinel" in caplog.text
