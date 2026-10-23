import numpy as np

from egomimic.rldb.lance.timeline import (
    FROZEN_RUN,
    build_timeline,
    nearest_dt,
    pose_outliers,
)

P = 33_333_333


def test_timeline_without_gaps_is_identity():
    ts = np.arange(10, dtype=np.int64) * P
    out, src, filler, after_gap = build_timeline(ts)
    np.testing.assert_array_equal(out, ts)
    np.testing.assert_array_equal(src, np.arange(10))
    assert not filler.any() and not after_gap.any()


def test_timeline_fills_dropped_frames():
    ts = np.array([0, 1, 2, 5, 6], dtype=np.int64) * P
    out, src, filler, after_gap = build_timeline(ts)
    assert len(out) == 7
    np.testing.assert_array_equal(src, [0, 1, 2, 2, 2, 3, 4])
    np.testing.assert_array_equal(filler, [0, 0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(after_gap, [0, 0, 0, 0, 0, 1, 0])
    assert (np.diff(out) > 0).all()
    assert abs(out[3] - 3 * P) <= 1


def test_timeline_tolerates_jitter():
    ts = np.array([0, P + 4_000_000, 2 * P - 3_000_000, 3 * P], dtype=np.int64)
    _, _, filler, after_gap = build_timeline(ts)
    assert not filler.any() and not after_gap.any()


def test_nearest_dt():
    ref = np.array([0, 100, 200], dtype=np.int64)
    np.testing.assert_array_equal(
        nearest_dt(np.array([0, 40, 160, 260]), ref), [0, 40, 40, 60]
    )


def test_pose_outliers_flags_jump_only_at_the_jump():
    xyz = np.zeros((6, 3))
    xyz[:, 0] = [0, 0.01, 0.02, 1.0, 1.01, 1.02]
    bad = pose_outliers(xyz, np.ones(6, bool))
    np.testing.assert_array_equal(bad, [0, 0, 0, 1, 0, 0])


def test_pose_outliers_skips_invalid_rows():
    xyz = np.zeros((4, 3))
    xyz[:, 0] = [0, 5.0, 0.01, 0.02]
    ok = np.array([1, 0, 1, 1], bool)
    assert not pose_outliers(xyz, ok).any()


def test_pose_outliers_frozen_run():
    n = FROZEN_RUN + 5
    xyz = np.cumsum(np.full((n, 3), 0.001), axis=0)
    xyz[2 : 2 + FROZEN_RUN] = xyz[2]
    bad = pose_outliers(xyz, np.ones(n, bool))
    assert bad[2 : 2 + FROZEN_RUN].all() and not bad[:2].any()
    short = np.cumsum(np.full((n, 3), 0.001), axis=0)
    short[3] = short[2]
    assert not pose_outliers(short, np.ones(n, bool)).any()
