"""History norm stats (egomimic/rldb/zarr/history_stats.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr import history_stats
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from tests.unit.test_proprio_history import _write_varying_episode

STATE = "observations.state.ee_pose"
EMB = get_embodiment_id("human_bimanual")
STATS = ("mean", "std", "min", "max", "median", "quantile_1", "quantile_99")


def _base(width=20):
    return {s: np.full(width, 100.0 + i, dtype=np.float32) for i, s in enumerate(STATS)}


def _entry(max_lag=20, channels=(0, 1, 2)):
    lags = np.arange(max_lag + 1, dtype=np.float32)
    return {
        "zarr_key": STATE,
        "channels": list(channels),
        "count": np.full(max_lag + 1, 7),
        "stats": {
            s: np.tile(lags[:, None] + 1000 * i, (1, len(channels)))
            for i, s in enumerate(STATS)
        },
    }


def test_gather_tiles_the_base_and_takes_each_steps_lag_row():
    out = history_stats.gather(_base(), _entry(), history=3, stride=5)
    assert out["mean"].shape == (3, 20)
    # oldest first: lags 10, 5, 0
    np.testing.assert_array_equal(out["mean"][:, 0], [10.0, 5.0, 0.0])
    np.testing.assert_array_equal(out["quantile_99"][:, 2], [6010.0, 6005.0, 6000.0])
    # every other channel is the current-step stat at every step
    np.testing.assert_array_equal(out["mean"][:, 3:], np.full((3, 17), 100.0))


def test_gather_refuses_a_history_past_the_table():
    with pytest.raises(ValueError, match="max_lag 20"):
        history_stats.gather(_base(), _entry(max_lag=20), history=5, stride=6)


def test_gather_keeps_stats_the_table_does_not_hold():
    base = {**_base(), "quantile_0_01": np.full(20, -5.0, dtype=np.float32)}
    entry = _entry()
    del entry["stats"]["min"]
    out = history_stats.gather(base, entry, history=2, stride=1)
    np.testing.assert_array_equal(out["min"], np.full((2, 20), 102.0))
    np.testing.assert_array_equal(out["quantile_0_01"], np.full((2, 20), -5.0))


def test_save_load_round_trip(tmp_path):
    path = history_stats.save(str(tmp_path / "norm_stats"), EMB, {"s": _entry()}, 20)
    assert path.endswith(history_stats.HISTORY_STATS_FILE)
    table, max_lag = history_stats.load(str(tmp_path / "norm_stats"), EMB)
    assert max_lag == 20
    np.testing.assert_array_equal(
        table["s"]["stats"]["mean"], _entry()["stats"]["mean"]
    )
    assert table["s"]["channels"] == [0, 1, 2]
    with pytest.raises(ValueError, match="max_lag"):
        history_stats.save(str(tmp_path / "norm_stats"), EMB, {"s": _entry(10)}, 10)


class _Shell:
    def __init__(self, stats):
        self.norm_stats = {EMB: stats}


def test_apply_gathers_once_and_leaves_gathered_stats_alone(tmp_path):
    history_stats.save(str(tmp_path), EMB, {"s": _entry()}, 20)
    shell = _Shell({"s": _base(), "other": _base()})
    assert history_stats.apply(shell, EMB, str(tmp_path), history=3, stride=5) == ["s"]
    assert shell.norm_stats[EMB]["s"]["mean"].shape == (3, 20)
    assert shell.norm_stats[EMB]["other"]["mean"].shape == (20,)
    # a run's saved norm_stats.json already holds (K, D): reloading it is a no-op
    gathered = shell.norm_stats[EMB]["s"]
    assert history_stats.apply(shell, EMB, str(tmp_path), history=3, stride=5) == []
    assert shell.norm_stats[EMB]["s"] is gathered
    with pytest.raises(ValueError, match="4-step"):
        history_stats.apply(shell, EMB, str(tmp_path), history=4, stride=5)


def test_lag_channels_are_the_wrist_translation_only():
    assert history_stats.lag_channels(STATE, 20) == [0, 1, 2, 10, 11, 12]
    assert history_stats.lag_channels("observations.state.keypoints", 144) == [
        0,
        1,
        2,
        72,
        73,
        74,
    ]
    assert history_stats.lag_channels("actions_keypoints", 144) is None
    assert history_stats.lag_channels(STATE, 7) is None


# ---------------------------------------------------------------------------
# compute() on a real dataset + transform pipeline
# ---------------------------------------------------------------------------

TRANSFORMS = dict(
    mode="cartesian_wristframe_6d",
    stride=1,
    fix_left_wrist_convention=True,
    pad_proprio_gripper=True,
)


def _md(tmp_path, **km):
    key_map = Human.get_keymap(keymap_mode="cartesian", norm_mode=True, **km)
    tl = Human.get_transform_list(**TRANSFORMS)
    resolver = LocalEpisodeResolver(tmp_path, key_map=key_map, transform_list=tl)
    return MultiDataset._from_resolver(resolver, mode="total")


@pytest.fixture
def episodes(tmp_path):
    for seed in range(2):
        _write_varying_episode(tmp_path, seed=seed)
    return tmp_path


def test_compute_matches_a_direct_per_lag_computation(episodes):
    max_lag = 6
    base = _md(episodes)
    norm = MultiDataset(state={}, norm_mode="quantile")
    norm.populate_from_datasets({"human_bimanual": base})
    lag_ds = _md(episodes, **history_stats.history_keymap({}, max_lag))
    table = history_stats.compute(
        norm, lag_ds, EMB, max_lag, n_anchors=10**6, num_workers=0
    )
    ((name, entry),) = table.items()
    assert entry["zarr_key"] == STATE
    ch = entry["channels"]
    assert ch == [0, 1, 2, 10, 11, 12]

    # every frame is an anchor; lag L is real for frames >= L of each episode
    lens = [len(leaf) for leaf in base.datasets.values()]
    for lag in (0, 3, max_lag):
        assert entry["count"][lag] == sum(n - lag for n in lens)

    # lag 0 is the current-step value
    cur = np.stack([np.asarray(base[i][STATE])[ch] for i in range(len(base))])
    np.testing.assert_allclose(entry["stats"]["mean"][0], cur.mean(0), atol=1e-5)

    # lag L: the past wrist in the anchor's head frame, from a K = L + 1 read
    lag = 4
    hist = _md(episodes, proprio_history=lag + 1, history_stride=1)
    rows = []
    for i in range(len(hist)):
        s = hist[i]
        if np.asarray(s["proprio_history_mask"])[0] > 0:
            rows.append(np.asarray(s[STATE])[0, ch])
    rows = np.stack(rows)
    np.testing.assert_allclose(entry["stats"]["mean"][lag], rows.mean(0), atol=1e-5)
    np.testing.assert_allclose(entry["stats"]["max"][lag], rows.max(0), atol=1e-5)
    # the head moves, so the past wrist is not distributed like the current one
    assert not np.allclose(entry["stats"]["mean"][lag], entry["stats"]["mean"][0])


def test_gathered_stats_normalize_a_history_sample(episodes, tmp_path_factory):
    max_lag, K, stride = 6, 3, 2
    base = _md(episodes)
    norm = MultiDataset(state={}, norm_mode="quantile")
    norm.populate_from_datasets({"human_bimanual": base})
    norm.infer_norm_from_dataset(base, "human_bimanual", sample_frac=1.0, num_workers=0)
    lag_ds = _md(episodes, **history_stats.history_keymap({}, max_lag))
    out = tmp_path_factory.mktemp("stats")
    history_stats.save(
        str(out),
        EMB,
        history_stats.compute(
            norm, lag_ds, EMB, max_lag, n_anchors=10**6, num_workers=0
        ),
        max_lag,
    )
    name = norm.zarr_key_to_keyname(STATE, EMB)
    assert history_stats.apply(norm, EMB, str(out), K, stride) == [name]

    hist = _md(episodes, proprio_history=K, history_stride=stride)
    hist.set_norm_stats_from(norm)
    sample = hist[len(hist) // 2]
    assert tuple(sample[STATE].shape) == (K, 20)
    assert torch.isfinite(torch.as_tensor(sample[STATE])).all()


def test_history_stride_frames():
    assert history_stats.history_stride_frames({"history_stride": 5}, None) == 5
    assert history_stats.history_stride_frames({}, None) == 1


def test_history_stride_seconds_resolve_through_each_episodes_fps(episodes):
    ds = _md(episodes)
    assert history_stats.history_stride_frames({"history_stride_s": 0.5}, ds) == 15
