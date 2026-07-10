"""Unit tests for data-driven trans_vel statistics.

Verifies the path that turns raw (T, 14) bimanual cartesian chunks into
normalization stats for the synthesized velocity channels:
- collect_trans_vel_samples produces per-mode shapes consistent with the
  tokenized layout, and skips invalid-pose sentinel chunks;
- trans_vel_stats_from_samples matches direct numpy reductions and emits the
  standard {min, max, mean, std, q01, q99} dict;
- compute_trans_vel_stats brackets the analytic speeds of constant-velocity
  chunks per arm.

Adapted from the GR00T stats tests (gr00t branch
``rpunamiya/arc-length-tokenizer``,
``tests/core/data/state_action/test_arc_length_stats.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.rldb.zarr.arc_length_stats import (
    TRANS_VEL_LEFT_KEY,
    TRANS_VEL_RIGHT_KEY,
    collect_trans_vel_samples,
    compute_trans_vel_stats,
    trans_vel_stats_from_samples,
)
from egomimic.rldb.zarr.arc_length_tokenizer import BimanualArcLengthConfig

T_IN = 40
DT = 1.0 / 30.0


def _bimanual_line_chunk(
    left_speed: float, right_speed: float, right_axis: int = 1
) -> np.ndarray:
    """(T, 14) chunk: left arm along +x at left_speed, right arm along
    right_axis at right_speed, identity rotations, linear grippers."""
    t = np.arange(T_IN) * DT
    left_pos = np.zeros((T_IN, 3))
    left_pos[:, 0] = left_speed * t
    right_pos = np.zeros((T_IN, 3))
    right_pos[:, right_axis] = right_speed * t
    ypr = np.zeros((T_IN, 3))
    grip = np.linspace(0.0, 1.0, T_IN)[:, None]
    left = np.concatenate([left_pos, ypr, grip], axis=-1)
    right = np.concatenate([right_pos, ypr, grip], axis=-1)
    return np.concatenate([left, right], axis=-1)


def _cfg(M: int = 20, mode: str = "mean_scalar") -> BimanualArcLengthConfig:
    return BimanualArcLengthConfig(
        min_distance_unit=0.05,
        resampled_vector_length=M,
        mode=mode,
        dt=DT,
    )


# --------------------------------------------------------------------------- #
# collect_trans_vel_samples — shapes per mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mode,M,vel_dim",
    [
        ("mean_scalar", 10, 1),
        ("mean_scalar", 20, 1),
        ("per_step_scalar", 10, 1),
        ("mean_per_dim", 20, 3),
        ("per_step_per_dim", 10, 3),
    ],
)
def test_collect_shapes(mode: str, M: int, vel_dim: int) -> None:
    n_chunks = 4
    chunks = [_bimanual_line_chunk(0.1, 0.05) for _ in range(n_chunks)]
    left, right = collect_trans_vel_samples(chunks, _cfg(M=M, mode=mode))
    assert left.shape == (n_chunks * M, vel_dim)
    assert right.shape == (n_chunks * M, vel_dim)
    assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))


def test_collect_skips_sentinel_chunks() -> None:
    good = _bimanual_line_chunk(0.1, 0.05)
    bad = good.copy()
    bad[5, 3] = 1e9
    left, right = collect_trans_vel_samples([good, bad, good], _cfg(M=20))
    assert left.shape == (2 * 20, 1)  # only the two good chunks contribute
    assert right.shape == (2 * 20, 1)


def test_collect_empty_input() -> None:
    left, right = collect_trans_vel_samples([], _cfg(M=20))
    assert left.shape == (0, 1) and right.shape == (0, 1)


# --------------------------------------------------------------------------- #
# trans_vel_stats_from_samples — shape & numerical correctness
# --------------------------------------------------------------------------- #
def test_stats_from_samples_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.2, scale=0.05, size=(10_000, 1))
    stats = trans_vel_stats_from_samples(samples)

    assert set(stats.keys()) == {"min", "max", "mean", "std", "q01", "q99"}
    np.testing.assert_allclose(stats["min"], samples.min(axis=0), atol=1e-12)
    np.testing.assert_allclose(stats["max"], samples.max(axis=0), atol=1e-12)
    np.testing.assert_allclose(stats["mean"], samples.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(stats["std"], samples.std(axis=0), atol=1e-12)
    np.testing.assert_allclose(
        stats["q01"], np.quantile(samples, 0.01, axis=0), atol=1e-12
    )
    np.testing.assert_allclose(
        stats["q99"], np.quantile(samples, 0.99, axis=0), atol=1e-12
    )
    # JSON-serializable lists of python floats, not numpy scalars.
    for v in stats.values():
        assert isinstance(v, list)
        assert all(isinstance(x, float) for x in v)


def test_stats_from_samples_per_dim_width() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(500, 3))
    stats = trans_vel_stats_from_samples(samples)
    assert all(len(v) == 3 for v in stats.values())


def test_stats_from_samples_empty_raises() -> None:
    with pytest.raises(ValueError, match="0 samples"):
        trans_vel_stats_from_samples(np.zeros((0, 1)))


# --------------------------------------------------------------------------- #
# End-to-end compute_trans_vel_stats
# --------------------------------------------------------------------------- #
def test_compute_stats_constant_speed_per_arm() -> None:
    """Identical constant-speed chunks: std == 0, min == max, and the mean is
    the true speed up to the whole-step timing quantization (the payload is
    chord over whole steps covered, so it can exceed the true speed by at most
    one step's worth)."""
    v_left, v_right = 0.10, 0.05
    chunks = [_bimanual_line_chunk(v_left, v_right) for _ in range(8)]
    stats = compute_trans_vel_stats(chunks, _cfg(M=20, mode="mean_scalar"))

    assert set(stats.keys()) == {TRANS_VEL_LEFT_KEY, TRANS_VEL_RIGHT_KEY}
    for key, v in ((TRANS_VEL_LEFT_KEY, v_left), (TRANS_VEL_RIGHT_KEY, v_right)):
        np.testing.assert_allclose(stats[key]["std"], [0.0], atol=1e-12)
        np.testing.assert_allclose(stats[key]["min"], stats[key]["max"], rtol=1e-12)
        np.testing.assert_allclose(stats[key]["min"], stats[key]["mean"], rtol=1e-12)
        assert v * 0.999 <= stats[key]["mean"][0] < v * 1.15


def test_compute_stats_mean_per_dim_direction() -> None:
    """mean_per_dim stats are per-component: left arm moves along +x only, so
    its y/z velocity stats are zero; right arm along +y likewise."""
    chunks = [_bimanual_line_chunk(0.10, 0.05) for _ in range(4)]
    stats = compute_trans_vel_stats(chunks, _cfg(M=20, mode="mean_per_dim"))
    left, right = stats[TRANS_VEL_LEFT_KEY], stats[TRANS_VEL_RIGHT_KEY]
    assert len(left["mean"]) == 3
    assert 0.10 * 0.999 <= left["mean"][0] < 0.10 * 1.15
    np.testing.assert_allclose(left["mean"][1:], [0.0, 0.0], atol=1e-12)
    assert 0.05 * 0.999 <= right["mean"][1] < 0.05 * 1.15
    np.testing.assert_allclose([right["mean"][0], right["mean"][2]], 0.0, atol=1e-12)


def test_compute_stats_spans_mixed_speeds() -> None:
    """Stats over a mix of fast and slow chunks must bracket both speeds."""
    speeds = [0.02, 0.05, 0.10, 0.20]
    chunks = [_bimanual_line_chunk(v, v) for v in speeds]
    stats = compute_trans_vel_stats(chunks, _cfg(M=20, mode="mean_scalar"))
    left = stats[TRANS_VEL_LEFT_KEY]
    assert min(speeds) <= left["min"][0] < min(speeds) * 1.2
    assert max(speeds) <= left["max"][0] < max(speeds) * 1.2
    assert min(speeds) < left["mean"][0] < max(speeds) * 1.2


def test_compute_stats_all_sentinel_raises() -> None:
    """If every chunk is invalid there are no samples — must raise, matching
    the no-silent-defaults contract of the stats pipeline."""
    bad = _bimanual_line_chunk(0.1, 0.05)
    bad[0, 0] = 1e9
    with pytest.raises(ValueError, match="0 samples"):
        compute_trans_vel_stats([bad, bad.copy()], _cfg(M=20))
