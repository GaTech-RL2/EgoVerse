"""Range-based normalization modes of MultiDataset (minmax / quantile /
quantile_0_01): round trip, [-1, 1] mapping of the bounds, and state
round-trip of the mode name. State-only construction, no zarr on disk."""

import numpy as np
import pytest
import torch

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

EMB = 3
KEY = "actions"
ZARR_KEY = "actions_cartesian"


def _stats_from(x: np.ndarray) -> dict:
    return {
        k: np.asarray(v, dtype=np.float32)
        for k, v in MultiDataset._compute_stats_for_array(x).items()
    }


@pytest.fixture(scope="module")
def data_and_stats():
    rng = np.random.default_rng(0)
    # 20k frames, 4 dims, plus one extreme outlier on dim 0 and one on dim 2.
    # One frame is 0.005 % of the data, below the 0.01 % tail, so the true
    # min/max sits far outside the 0.01 / 99.99 percentiles.
    x = rng.normal(size=(20000, 4)).astype(np.float32)
    x[0, 0] = -50.0
    x[1, 2] = 80.0
    return x, _stats_from(x)


def _dataset(norm_mode: str, stats: dict) -> MultiDataset:
    state = {
        "norm_mode": norm_mode,
        "embodiments": [EMB],
        "key_types": {EMB: {KEY: "action_keys"}},
        "zarr_keys": {EMB: {KEY: ZARR_KEY}},
        "shapes": {EMB: {KEY: (4,)}},
        "norm_stats": {EMB: {KEY: stats}},
    }
    return MultiDataset.from_state(state)


@pytest.mark.parametrize(
    "norm_mode,lo_key,hi_key",
    [
        ("minmax", "min", "max"),
        ("quantile", "quantile_1", "quantile_99"),
        ("quantile_0_01", "quantile_0_01", "quantile_99_99"),
    ],
)
def test_range_modes_map_bounds_to_unit_interval(
    data_and_stats, norm_mode, lo_key, hi_key
):
    _, stats = data_and_stats
    ds = _dataset(norm_mode, stats)
    bounds = torch.from_numpy(np.stack([stats[lo_key], stats[hi_key]]))
    out = ds.normalize({ZARR_KEY: bounds}, EMB)[ZARR_KEY]
    # lo -> -1 and hi -> +1 (up to the 1e-6 epsilon in the denominator)
    assert torch.allclose(out[0], torch.full((4,), -1.0), atol=1e-4)
    assert torch.allclose(out[1], torch.full((4,), 1.0), atol=1e-4)


@pytest.mark.parametrize("norm_mode", ["minmax", "quantile", "quantile_0_01"])
def test_range_modes_round_trip(data_and_stats, norm_mode):
    x, stats = data_and_stats
    ds = _dataset(norm_mode, stats)
    t = torch.from_numpy(x[:512])
    back = ds.unnormalize(ds.normalize({ZARR_KEY: t}, EMB), EMB)[ZARR_KEY]
    assert torch.allclose(back, t, atol=1e-4)


def test_quantile_0_01_keeps_inliers_inside_unit_interval(data_and_stats):
    x, stats = data_and_stats
    ds = _dataset("quantile_0_01", stats)
    lo, hi = stats["quantile_0_01"], stats["quantile_99_99"]
    inliers = x[np.all((x >= lo) & (x <= hi), axis=1)]
    assert len(inliers) > 0.99 * len(x)
    out = ds.normalize({ZARR_KEY: torch.from_numpy(inliers)}, EMB)[ZARR_KEY]
    assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5
    # Under minmax the same inliers are squeezed well inside [-1, 1] on the
    # outlier dims because the extremes define the range.
    mm = _dataset("minmax", stats).normalize(
        {ZARR_KEY: torch.from_numpy(inliers)}, EMB
    )[ZARR_KEY]
    assert mm[:, 0].min() > -0.8 and mm[:, 2].max() < 0.8
    # And the new mode uses the full window on those dims.
    assert out[:, 0].min() < -0.99 and out[:, 2].max() > 0.99


def test_unknown_mode_raises(data_and_stats):
    _, stats = data_and_stats
    ds = _dataset("nope", stats)
    with pytest.raises(ValueError, match="Invalid normalization mode"):
        ds.normalize({ZARR_KEY: torch.zeros(2, 4)}, EMB)


def test_mode_survives_state_round_trip(data_and_stats):
    _, stats = data_and_stats
    ds = _dataset("quantile_0_01", stats)
    ds2 = MultiDataset.from_state(ds.to_state())
    assert ds2.norm_mode == "quantile_0_01"
    t = torch.randn(8, 4)
    assert torch.equal(
        ds.normalize({ZARR_KEY: t}, EMB)[ZARR_KEY],
        ds2.normalize({ZARR_KEY: t}, EMB)[ZARR_KEY],
    )


def _bounds_dataset(stats: dict, key: str, zarr_key: str, n_dims: int):
    return MultiDataset(
        state={
            "norm_mode": "quantile_0_01",
            "embodiments": [EMB],
            "key_types": {EMB: {key: "action_keys"}},
            "zarr_keys": {EMB: {key: zarr_key}},
            "shapes": {EMB: {key: (n_dims,)}},
            "norm_stats": {EMB: {key: stats}},
        },
        reject_outliers=True,
    )


def test_pose_position_dims_layouts():
    from egomimic.utils.pose_utils import is_pose_key, pose_position_dims

    assert pose_position_dims(6) == [0, 1, 2]
    assert pose_position_dims(7) == [0, 1, 2]
    assert pose_position_dims(9) == [0, 1, 2]
    assert pose_position_dims(12) == [0, 1, 2, 6, 7, 8]
    assert pose_position_dims(14) == [0, 1, 2, 7, 8, 9]
    assert pose_position_dims(18) == [0, 1, 2, 9, 10, 11]
    assert pose_position_dims(4) is None
    assert is_pose_key("actions_cartesian")
    assert is_pose_key("observations.state.ee_pose")
    assert not is_pose_key("actions_joints")


def test_bounds_check_skips_rotation_dims_for_pose_keys(data_and_stats):
    _, stats = data_and_stats  # 4-dim stats; non-pose keys check every dim
    hi = torch.from_numpy(stats["quantile_99_99"]).clone()
    inside = (hi - 0.5).unsqueeze(0)
    out3 = inside.clone()
    out3[0, 3] = hi[3] + 10.0

    def check(ds, zk, t):
        return ds._check_bounds({"embodiment": EMB, zk: t}, None, 0, "d")

    ds_all = _bounds_dataset(stats, KEY, ZARR_KEY, 4)
    assert check(ds_all, ZARR_KEY, inside) is None
    assert check(ds_all, ZARR_KEY, out3) is not None

    # A 12-dim pose key: rotation dims (3-5, 9-11) never reject, xyz dims do.
    stats12 = {k: np.tile(np.asarray(v, dtype=np.float32), 3) for k, v in stats.items()}
    ds_pose = _bounds_dataset(stats12, "actions_cartesian", "actions_cartesian", 12)
    hi12 = torch.from_numpy(stats12["quantile_99_99"]).clone()
    inside12 = (hi12 - 0.5).unsqueeze(0)
    assert check(ds_pose, "actions_cartesian", inside12) is None
    for rot_dim in (3, 4, 5, 9, 10, 11):
        t = inside12.clone()
        t[0, rot_dim] = hi12[rot_dim] + 10.0
        assert check(ds_pose, "actions_cartesian", t) is None, rot_dim
    for pos_dim in (0, 1, 2, 6, 7, 8):
        t = inside12.clone()
        t[0, pos_dim] = hi12[pos_dim] + 10.0
        assert check(ds_pose, "actions_cartesian", t) is not None, pos_dim
    # NaN is rejected regardless
    nan = inside12.clone()
    nan[0, 3] = float("nan")
    assert check(ds_pose, "actions_cartesian", nan) is not None
    # Unknown pose layout (4 dims) falls back to checking every dim.
    ds_unknown = _bounds_dataset(stats, "left.obs_ee_pose", "left.obs_ee_pose", 4)
    assert check(ds_unknown, "left.obs_ee_pose", out3) is not None
