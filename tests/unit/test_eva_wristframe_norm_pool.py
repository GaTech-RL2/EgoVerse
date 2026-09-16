"""Eva wrist-frame defaults, horizon-pooled norm stats, per-segment L2 metrics."""

import numpy as np
import pytest
import torch
from fixtures.synthetic_episodes import write_episode
from hydra import compose, initialize_config_module

from egomimic.eval.action_metrics import cartesian_metrics, keypoint_metrics
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr import norm_cache
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset

EVA_WRIST_REVERT = "egomimic.rldb.embodiment.eva._build_eva_cartesian_revert_6d_wristframe_transform_list"


def _compose(name, overrides=()):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize("top", ["train_zarr_cartesian", "train_zarr_cartesian_pi"])
def test_eva_defaults_are_wrist_frame_6d(top):
    cfg = _compose(top)
    eva = cfg.data.train_datasets.eva_bimanual
    assert eva.resolver.transform_list.mode == "cartesian_wristframe_6d"
    assert cfg.evaluator.transform_lists.eva_bimanual._target_ == EVA_WRIST_REVERT
    assert cfg.norm_stats.pool_horizon is False


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


def test_segment_l2_separates_early_from_late_error():
    B, T = 2, 100
    gt = torch.zeros(B, T, 20)
    pred = gt.clone()
    pred[:, :10, 0] = 0.03  # 3 cm on the left arm x, first 10 % only
    m = cartesian_metrics(pred, gt, "p")
    # mean over both arms: 3 cm on one of two
    assert m["p_xyz_l2_early"].item() == pytest.approx(0.015)
    assert m["p_xyz_l2_mid"].item() == pytest.approx(0.0)
    assert m["p_xyz_l2_late"].item() == pytest.approx(0.0)


def test_segment_l2_on_keypoints():
    B, T = 1, 100
    gt = torch.zeros(B, T, 144)
    pred = gt.clone()
    pred[:, 50:, 9:72] = 0.02  # every left-hand keypoint off by (2,2,2) cm late
    m = keypoint_metrics(pred, gt, "k")
    expected = 0.5 * float(np.linalg.norm([0.02] * 3))  # one hand of two
    assert m["k_kp_l2_late"].item() == pytest.approx(expected, rel=1e-5)
    assert m["k_kp_l2_early"].item() == pytest.approx(0.0)
    assert m["k_wrist_xyz_l2_late"].item() == pytest.approx(0.0)


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


def test_bounds_check_ignores_constant_cells():
    key = "actions_cartesian"
    q1 = np.full((4, 14), -1.0, dtype=np.float32)
    q99 = np.full((4, 14), 1.0, dtype=np.float32)
    q1[0, :3] = q99[0, :3] = 0.0  # wrist-frame t=0: xyz exactly 0
    md = MultiDataset.__new__(MultiDataset)
    md.norm_stats = {0: {key: {"quantile_1": q1, "quantile_99": q99}}}
    md.zarr_keys = {0: {key: key}}
    md._warned_violations = set()

    arr = np.zeros((4, 14), dtype=np.float32)
    arr[0, 0] = 1e-3  # off-convention offset at a constant cell: admitted
    assert md._check_bounds({"embodiment": 0, key: arr}, None, 0, "ep") is None
    arr[1, 0] = 50.0  # corrupt value at a regular cell: still rejected
    assert md._check_bounds({"embodiment": 0, key: arr}, None, 0, "ep") is not None


# ------------------------------------------------------- rollout reverts
def _rand_pose7(rng):
    from scipy.spatial.transform import Rotation as R

    q = R.random(random_state=int(rng.integers(1 << 31))).as_quat()
    return np.concatenate([rng.uniform(-0.5, 0.5, 3), q[[3, 0, 1, 2]]])


def _rand_chunk7(rng, start, n=45):
    from scipy.spatial.transform import Rotation as R

    out = np.zeros((n, 7))
    p, r = start[:3].copy(), R.from_quat(start[[4, 5, 6, 3]])
    for t in range(n):
        if t:
            p = p + rng.normal(0, 0.01, 3)
            r = R.from_rotvec(rng.normal(0, 0.05, 3)) * r
        out[t] = np.concatenate([p, r.as_quat()[[3, 0, 1, 2]]])
    return out


# Compared against cartesian_6d, so that mode is the reference, not a case.
@pytest.mark.parametrize(
    "mode", ["cartesian_wristframe_ypr", "cartesian_wristframe_6d"]
)
def test_eva_revert_for_rollout_recovers_camframe_ypr(mode):
    """rollout.py feeds Eva.get_transform_list(mode) outputs to the model and
    Eva.get_revert_transform_list(mode) must bring predictions back to the
    same cam-frame xyz+ypr+gripper (14-D) whatever the training mode."""
    from scipy.spatial.transform import Rotation as R

    from egomimic.rldb.embodiment.embodiment import Embodiment

    rng = np.random.default_rng(3)
    lobs, robs = _rand_pose7(rng), _rand_pose7(rng)
    raw = {
        "left.obs_ee_pose": lobs,
        "right.obs_ee_pose": robs,
        "left.cmd_ee_pose": _rand_chunk7(rng, lobs),
        "right.cmd_ee_pose": _rand_chunk7(rng, robs),
        "left.obs_gripper": np.array([0.2]),
        "right.obs_gripper": np.array([0.8]),
        "left.cmd_gripper": rng.uniform(0, 1, (45, 1)),
        "right.cmd_gripper": rng.uniform(0, 1, (45, 1)),
    }

    def run(m):
        s = {k: v.copy() for k, v in raw.items()}
        for t in Eva.get_transform_list(m):
            s = t.transform(s)
        batch = {
            "actions_cartesian": torch.as_tensor(s["actions_cartesian"])[None],
            "observations.state.ee_pose": torch.as_tensor(
                s["observations.state.ee_pose"]
            )[None],
        }
        rev = Eva.get_revert_transform_list(m)
        if rev is not None:
            batch = Embodiment.apply_transform(batch, rev)
        return np.asarray(batch["actions_cartesian"][0], dtype=np.float64)

    assert Eva.get_revert_transform_list("cartesian") is None
    got, ref = run(mode), run("cartesian_6d")
    assert got.shape == ref.shape == (100, 14)
    for off in (0, 7):
        np.testing.assert_allclose(
            got[:, off : off + 3], ref[:, off : off + 3], atol=1e-5
        )
        np.testing.assert_allclose(got[:, off + 6], ref[:, off + 6], atol=1e-5)
        Rg = R.from_euler("ZYX", got[:, off + 3 : off + 6]).as_matrix()
        Rr = R.from_euler("ZYX", ref[:, off + 3 : off + 6]).as_matrix()
        np.testing.assert_allclose(Rg, Rr, atol=1e-4)
