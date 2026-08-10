"""Unit tests for the bimanual arc-length tokenizer.

Covers, in order:
- core single-arm tokenizer: arc-length math, chunking, uniform resampling,
  SLERP rotation handling, velocity payloads, zero tokens, input validation;
- bimanual layer: (T, 14) -> (M, 16) shapes, per-arm independent arc length,
  stationary-arm zero velocity, invalid-pose sentinel;
- round-trip (deploy inverse): tokenize -> detokenize recovers the original
  time-indexed chunk for constant-velocity motion, holds pose for stationary
  tokens, and stays on-path for curved motion;
- the Transform-style pipeline adapter.

Adapted from the GR00T arc-length tokenizer tests
(gr00t branch ``rpunamiya/arc-length-tokenizer``,
``tests/core/data/state_action/test_arc_length_*.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from egomimic.rldb.zarr.arc_length_tokenizer import (
    ARC_BIMANUAL_DIM,
    ARM_DIM,
    BIMANUAL_CARTESIAN_DIM,
    INVALID_POSE_FILL,
    ArcLengthTokenizer,
    BimanualArcLengthConfig,
    BimanualArcLengthTokenizer,
    TokenizeBimanualArcLength,
    VelocityMode,
    cumulative_arc_length,
    velocity_dim,
)

T_IN = 40  # source action horizon (time steps)
DT = 1.0 / 30.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rotation_angle(ypr_a: np.ndarray, ypr_b: np.ndarray) -> float:
    """Geodesic angle (rad) between two ypr rotations — robust to euler wrap."""
    ra = R.from_euler("ZYX", np.asarray(ypr_a).reshape(1, 3))
    rb = R.from_euler("ZYX", np.asarray(ypr_b).reshape(1, 3))
    return float((ra.inv() * rb).magnitude()[0])


def _point_to_polyline_dist(p: np.ndarray, poly: np.ndarray) -> float:
    """Min distance from point p (3,) to the polyline through poly (N, 3)."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(
        ((p[None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0
    )
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(proj - p[None], axis=-1).min())


def _make_arm_line(
    T: int, total_dist: float, seed: int, rot_magnitude: float = 0.3
) -> np.ndarray:
    """(T, 7) arm chunk: constant-speed straight line, uniform SLERP rotation,
    linear gripper 0.2 -> 0.8."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    start = rng.normal(size=3) * 0.1
    s = np.linspace(0.0, total_dist, T)
    pos = start[None, :] + s[:, None] * direction[None, :]

    r0 = R.identity()
    r1 = R.from_rotvec(rng.normal(size=3) * rot_magnitude)
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    ypr = slerp(np.linspace(0.0, 1.0, T)).as_euler("ZYX", degrees=False)

    gripper = np.linspace(0.2, 0.8, T)[:, None]
    return np.concatenate([pos, ypr, gripper], axis=-1)


def _make_arm_static(T: int, seed: int) -> np.ndarray:
    """(T, 7) arm chunk holding one random pose with gripper 0.5."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=3) * 0.1
    ypr = R.from_rotvec(rng.normal(size=3) * 0.3).as_euler("ZYX", degrees=False)
    arm = np.concatenate([pos, ypr, [0.5]])
    return np.repeat(arm[None, :], T, axis=0)


def _bimanual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    assert left.shape == right.shape and left.shape[1] == ARM_DIM
    return np.concatenate([left, right], axis=-1)


def _axis_line_arm(
    T: int, speed: float, axis: int = 0, offset_z: float = 0.0
) -> np.ndarray:
    """(T, 7) arm moving along one axis at constant speed, identity rotation."""
    t = np.arange(T) * DT
    pos = np.zeros((T, 3))
    pos[:, axis] = speed * t
    pos[:, 2] += offset_z
    ypr = np.zeros((T, 3))
    gripper = np.linspace(0.0, 1.0, T)[:, None]
    return np.concatenate([pos, ypr, gripper], axis=-1)


def _cfg(
    M: int = 20, mode: str = "mean_scalar", unit: float = 0.05
) -> BimanualArcLengthConfig:
    return BimanualArcLengthConfig(
        min_distance_unit=unit,
        resampled_vector_length=M,
        mode=mode,
        dt=DT,
    )


# --------------------------------------------------------------------------- #
# Core: arc-length math
# --------------------------------------------------------------------------- #
def test_cumulative_arc_length_known_path() -> None:
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 2, 2]], dtype=np.float64)
    np.testing.assert_allclose(cumulative_arc_length(pos), [0.0, 1.0, 3.0, 5.0])


def test_tokenize_chunk_count_and_distance() -> None:
    """A 0.2 m straight line with a 0.05 m unit splits into 4 motion tokens."""
    arm = _axis_line_arm(T=61, speed=0.1)  # 60 steps * 0.1/30 m = 0.2 m
    tok = ArcLengthTokenizer(min_distance_unit=0.05, resampled_vector_length=10, dt=DT)
    tokens = tok.tokenize(arm[:, 0:3], arm[:, 3:6], arm[:, 6:7])
    assert len(tokens) == 4
    for token in tokens:
        assert token.kind == "motion"
        np.testing.assert_allclose(token.chunk_distance, 0.05, atol=1e-9)
    # Tokens tile the trajectory contiguously.
    assert tokens[0].start_idx == 0
    for prev, cur in zip(tokens, tokens[1:]):
        assert cur.start_idx >= prev.start_idx


def test_tokenize_stationary_trajectory_returns_no_tokens() -> None:
    """Full-trajectory chunking of a static arm covers zero arc length."""
    arm = _make_arm_static(T=10, seed=0)
    tok = ArcLengthTokenizer(min_distance_unit=0.05, resampled_vector_length=10)
    assert tok.tokenize(arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]) == []


def test_waypoints_uniform_in_arc_length() -> None:
    """Waypoints are uniform in arc length even when input timing is not."""
    # Straight line along +x with wildly non-uniform time sampling.
    s = np.concatenate([np.linspace(0.0, 0.01, 15), np.linspace(0.012, 0.08, 10)])
    pos = np.stack([s, np.zeros_like(s), np.zeros_like(s)], axis=-1)
    ypr = np.zeros((len(s), 3))
    tok = ArcLengthTokenizer(min_distance_unit=0.05, resampled_vector_length=20, dt=DT)
    token = tok.tokenize_at(pos, ypr, t=0)
    assert token.kind == "motion"
    spacing = np.linalg.norm(np.diff(token.pos, axis=0), axis=-1)
    np.testing.assert_allclose(spacing, spacing[0], atol=1e-9)
    np.testing.assert_allclose(spacing.sum(), 0.05, atol=1e-9)


def test_rotation_slerp_midpoint() -> None:
    """A 90-degree yaw sweep across one token SLERPs to 45 degrees mid-chunk."""
    T = 21
    s = np.linspace(0.0, 0.05, T)
    pos = np.stack([s, np.zeros(T), np.zeros(T)], axis=-1)
    yaw = np.linspace(0.0, np.pi / 2, T)
    ypr = np.stack([yaw, np.zeros(T), np.zeros(T)], axis=-1)
    tok = ArcLengthTokenizer(min_distance_unit=0.05, resampled_vector_length=21, dt=DT)
    token = tok.tokenize_at(pos, ypr, t=0)
    mid = token.ypr[len(token.ypr) // 2]
    expected_mid = np.array([np.pi / 4, 0.0, 0.0])
    assert _rotation_angle(mid, expected_mid) < 1e-6
    # Endpoints match the trajectory's endpoints exactly.
    assert _rotation_angle(token.ypr[0], ypr[0]) < 1e-9
    assert _rotation_angle(token.ypr[-1], ypr[-1]) < 1e-6


def test_velocity_mean_scalar_broadcast_and_magnitude() -> None:
    """mean_scalar velocity is a single value ~= true speed for straight motion."""
    speed = 0.1
    arm = _axis_line_arm(T=T_IN, speed=speed)
    tok = ArcLengthTokenizer(
        min_distance_unit=0.05, resampled_vector_length=20, mode="mean_scalar", dt=DT
    )
    token = tok.tokenize_at(arm[:, 0:3], arm[:, 3:6], arm[:, 6:7], t=0)
    assert token.trans_vel.shape == (1,)
    # Whole-step timing quantization bounds the payload within one step of truth.
    assert speed <= token.trans_vel[0] < speed * 1.15


def test_velocity_per_step_scalar_constant_for_straight_line() -> None:
    """per_step_scalar velocity is per-waypoint and constant for uniform motion."""
    speed = 0.1
    arm = _axis_line_arm(T=T_IN, speed=speed)
    M = 20
    tok = ArcLengthTokenizer(
        min_distance_unit=0.05, resampled_vector_length=M, mode="per_step_scalar", dt=DT
    )
    token = tok.tokenize_at(arm[:, 0:3], arm[:, 3:6], arm[:, 6:7], t=0)
    assert token.trans_vel.shape == (M,)
    np.testing.assert_allclose(token.trans_vel, token.trans_vel[0], rtol=1e-9)
    assert speed * 0.85 < token.trans_vel[0] < speed * 1.15


def test_stationary_produces_zero_token() -> None:
    arm = _make_arm_static(T=T_IN, seed=3)
    tok = ArcLengthTokenizer(min_distance_unit=0.05, resampled_vector_length=20, dt=DT)
    token = tok.tokenize_at(arm[:, 0:3], arm[:, 3:6], arm[:, 6:7], t=0)
    assert token.kind == "zero"
    assert token.zero_reason == "distance_below_epsilon"
    np.testing.assert_allclose(token.trans_vel, 0.0)
    np.testing.assert_allclose(token.pos, np.broadcast_to(arm[0, 0:3], token.pos.shape))


def test_input_validation() -> None:
    tok = ArcLengthTokenizer()
    with pytest.raises(ValueError, match=r"pos must have shape"):
        tok.tokenize(np.zeros((5, 2)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match=r"ypr must have shape"):
        tok.tokenize(np.zeros((5, 3)), np.zeros((5, 4)))
    with pytest.raises(ValueError, match=r"same length"):
        tok.tokenize(np.zeros((5, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match=r"at least 2"):
        tok.tokenize(np.zeros((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match=r"resampled_vector_length"):
        ArcLengthTokenizer(resampled_vector_length=1)
    with pytest.raises(ValueError):
        ArcLengthTokenizer(mode="not_a_mode")
    with pytest.raises(ValueError, match=r"t must be in"):
        tok.tokenize_at(np.zeros((5, 3)), np.zeros((5, 3)), t=7)


# --------------------------------------------------------------------------- #
# Bimanual layer: shapes and per-arm independence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M", [10, 20])
@pytest.mark.parametrize(
    "mode,expected_dim",
    [
        ("mean_scalar", 16),
        ("per_step_scalar", 16),
        ("mean_per_dim", 20),
        ("per_step_per_dim", 20),
    ],
)
@pytest.mark.parametrize("unit", [0.05, 0.10])
def test_bimanual_output_shapes(
    M: int, mode: str, expected_dim: int, unit: float
) -> None:
    chunk = _bimanual(
        _make_arm_line(T_IN, total_dist=0.13, seed=1),
        _make_arm_line(T_IN, total_dist=0.07, seed=2),
    )
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=M, mode=mode, unit=unit))
    assert tokenizer.arc_dim == expected_dim
    out = tokenizer.tokenize(chunk)
    assert out.shape == (M, expected_dim)
    assert np.all(np.isfinite(out))


def test_bimanual_first_waypoint_anchored_and_unit_span() -> None:
    """Waypoint 0 equals the input pose at t=0; the token spans one distance
    unit of each arm's own arc length."""
    left = _axis_line_arm(T_IN, speed=0.10)
    right = _axis_line_arm(T_IN, speed=0.05, axis=1, offset_z=0.5)
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=20))
    out = tokenizer.tokenize(_bimanual(left, right))

    left_out, right_out = out[:, 0:8], out[:, 8:16]
    np.testing.assert_allclose(left_out[0, 0:7], left[0], atol=1e-9)
    np.testing.assert_allclose(right_out[0, 0:7], right[0], atol=1e-9)
    # Straight lines: chord == arc == one distance unit, per arm.
    assert abs(np.linalg.norm(left_out[-1, 0:3] - left_out[0, 0:3]) - 0.05) < 1e-9
    assert abs(np.linalg.norm(right_out[-1, 0:3] - right_out[0, 0:3]) - 0.05) < 1e-9


def test_bimanual_independent_arm_parameterizations() -> None:
    """Each arm resamples along its own line at its own speed."""
    left = _axis_line_arm(T_IN, speed=0.10, axis=0)
    right = _axis_line_arm(T_IN, speed=0.05, axis=1, offset_z=0.5)
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=20))
    out = tokenizer.tokenize(_bimanual(left, right))

    left_pos, right_pos = out[:, 0:3], out[:, 8:11]
    # Left moves along +x only; right along +y at z=0.5 only.
    np.testing.assert_allclose(left_pos[:, 1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(right_pos[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(right_pos[:, 2], 0.5, atol=1e-12)
    assert np.all(np.diff(left_pos[:, 0]) > 0)
    assert np.all(np.diff(right_pos[:, 1]) > 0)
    # Velocity channels reflect each arm's own speed (fast left > slow right).
    vel_left, vel_right = out[:, 7], out[:, 15]
    np.testing.assert_allclose(vel_left, vel_left[0], atol=1e-12)
    np.testing.assert_allclose(vel_right, vel_right[0], atol=1e-12)
    assert vel_left[0] > vel_right[0] > 0.0
    assert abs(vel_left[0] - 0.10) / 0.10 < 0.15
    assert abs(vel_right[0] - 0.05) / 0.05 < 0.15


def test_bimanual_stationary_arm_zero_velocity() -> None:
    """A static right arm yields a zero token: held pose, zero velocity —
    while the left arm tokenizes normally."""
    left = _axis_line_arm(T_IN, speed=0.10)
    right = _make_arm_static(T_IN, seed=7)
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=20))
    out = tokenizer.tokenize(_bimanual(left, right))

    np.testing.assert_allclose(out[:, 15], 0.0, atol=1e-12)  # right trans_vel
    np.testing.assert_allclose(  # held pos
        out[:, 8:11], np.broadcast_to(right[0, 0:3], (20, 3)), atol=1e-12
    )
    assert np.all(out[:, 7] > 0.0)  # left still moving


def test_bimanual_invalid_pose_sentinel() -> None:
    chunk = _bimanual(
        _make_arm_line(T_IN, total_dist=0.13, seed=1),
        _make_arm_line(T_IN, total_dist=0.07, seed=2),
    )
    chunk[3, 2] = 1e9  # one invalid sample poisons the chunk
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=20))
    out = tokenizer.tokenize(chunk)
    assert out.shape == (20, ARC_BIMANUAL_DIM)
    np.testing.assert_allclose(out, INVALID_POSE_FILL)


def test_bimanual_validation() -> None:
    tokenizer = BimanualArcLengthTokenizer(_cfg())
    with pytest.raises(ValueError, match=r"\(T, 14\)"):
        tokenizer.tokenize(np.zeros((10, 12)))
    with pytest.raises(ValueError, match=r"\(M, 16\)"):
        tokenizer.detokenize(np.zeros((20, 14)), action_horizon=10)
    per_step = BimanualArcLengthTokenizer(_cfg(mode="per_step_scalar"))
    with pytest.raises(NotImplementedError, match="mean_scalar"):
        per_step.detokenize(np.zeros((20, ARC_BIMANUAL_DIM)), action_horizon=10)


# --------------------------------------------------------------------------- #
# Round trip: tokenize -> detokenize (deploy inverse)
# --------------------------------------------------------------------------- #
def test_roundtrip_constant_velocity_recovers_chunk() -> None:
    """For constant-velocity motion the mean-scalar timing is (near-)exact, so
    the round trip recovers the original time-indexed chunk to within the
    whole-step timing quantization of one waypoint spacing."""
    T = 20
    cfg = _cfg(M=20)
    # Each arm travels exactly one distance unit over the T-step chunk.
    left = _make_arm_line(T, total_dist=cfg.min_distance_unit, seed=1)
    right = _make_arm_line(T, total_dist=cfg.min_distance_unit, seed=2)
    chunk = _bimanual(left, right)

    tokenizer = BimanualArcLengthTokenizer(cfg)
    arc = tokenizer.tokenize(chunk)
    rec = tokenizer.detokenize(arc, action_horizon=T)
    assert rec.shape == (T, BIMANUAL_CARTESIAN_DIM)

    waypoint_spacing = cfg.min_distance_unit / (cfg.resampled_vector_length - 1)
    for arm_in, arm_start in ((left, 0), (right, ARM_DIM)):
        arm_rec = rec[:, arm_start : arm_start + ARM_DIM]
        # PATH fidelity: every emitted position lies on the original path.
        max_off_path = max(
            _point_to_polyline_dist(p, arm_in[:, 0:3]) for p in arm_rec[:, 0:3]
        )
        assert max_off_path < 5e-4
        # STEP-wise recovery within the timing-quantization bound.
        pos_err = np.abs(arm_rec[:, 0:3] - arm_in[:, 0:3]).max()
        assert pos_err < 1.5 * waypoint_spacing
        rot_err = max(
            _rotation_angle(arm_rec[i, 3:6], arm_in[i, 3:6]) for i in range(T)
        )
        assert rot_err < 0.05
        grip_err = np.abs(arm_rec[:, 6] - arm_in[:, 6]).max()
        assert grip_err < 0.1


@pytest.mark.parametrize("H", [5, 20, 50])
def test_detokenize_emits_exact_horizon(H: int) -> None:
    cfg = _cfg(M=20)
    chunk = _bimanual(
        _make_arm_line(20, total_dist=cfg.min_distance_unit, seed=3),
        _make_arm_line(20, total_dist=cfg.min_distance_unit, seed=4),
    )
    tokenizer = BimanualArcLengthTokenizer(cfg)
    rec = tokenizer.detokenize(tokenizer.tokenize(chunk), action_horizon=H)
    assert rec.shape == (H, BIMANUAL_CARTESIAN_DIM)
    assert np.all(np.isfinite(rec))


def test_roundtrip_zero_motion_holds_pose() -> None:
    T = 20
    cfg = _cfg(M=20)
    left = _make_arm_static(T, seed=5)
    right = _make_arm_static(T, seed=6)
    tokenizer = BimanualArcLengthTokenizer(cfg)
    rec = tokenizer.detokenize(
        tokenizer.tokenize(_bimanual(left, right)), action_horizon=T
    )
    for arm_in, arm_start in ((left, 0), (right, ARM_DIM)):
        arm_rec = rec[:, arm_start : arm_start + ARM_DIM]
        np.testing.assert_allclose(
            arm_rec[:, 0:3], np.broadcast_to(arm_in[0, 0:3], (T, 3)), atol=1e-9
        )
        rot_err = max(
            _rotation_angle(arm_rec[i, 3:6], arm_in[0, 3:6]) for i in range(T)
        )
        assert rot_err < 1e-9
        np.testing.assert_allclose(arm_rec[:, 6], arm_in[0, 6], atol=1e-9)


def test_roundtrip_curved_path_fidelity() -> None:
    """A sharp quarter-circle token stays ON the predicted path within ~mm.

    mean_scalar velocity is chord-based, so per-chunk *progress* may
    under-shoot on sharp curvature (recovered by closed-loop re-inference) —
    but every emitted waypoint must lie on the path."""
    T = 24
    cfg = _cfg(M=20)
    r = cfg.min_distance_unit / (np.pi / 2)  # quarter circle of arc 0.05 m
    theta = np.linspace(0.0, np.pi / 2, T)
    pos = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros(T)], axis=-1)
    arm = np.concatenate(
        [pos, np.zeros((T, 3)), np.linspace(0, 1, T)[:, None]], axis=-1
    )
    tokenizer = BimanualArcLengthTokenizer(cfg)
    rec = tokenizer.detokenize(
        tokenizer.tokenize(_bimanual(arm, arm)), action_horizon=T
    )
    max_off_path = max(_point_to_polyline_dist(p, pos) for p in rec[:, 0:3])
    assert max_off_path < 1e-3


def test_roundtrip_gentle_curve_recovered_tightly() -> None:
    """A gentle curve (representative of real ~5 cm EEF motion) recovers
    tightly: chord ~= arc, so timing is near-exact."""
    T = 20
    cfg = _cfg(M=20)
    r = 0.5  # 0.05 m of arc on a 0.5 m radius circle -> nearly straight
    theta = np.linspace(0.0, cfg.min_distance_unit / r, T)
    pos = np.stack([r * np.sin(theta), r * (1 - np.cos(theta)), np.zeros(T)], axis=-1)
    arm = np.concatenate(
        [pos, np.zeros((T, 3)), np.linspace(0, 1, T)[:, None]], axis=-1
    )
    tokenizer = BimanualArcLengthTokenizer(cfg)
    rec = tokenizer.detokenize(
        tokenizer.tokenize(_bimanual(arm, arm)), action_horizon=T
    )
    assert np.abs(rec[:, 0:3] - pos).max() < 5e-4


def test_mean_vs_per_step_velocity_diverges_for_curved_motion() -> None:
    """mean_scalar measures chord/time; per_step_scalar measures per-segment
    arc/time. They agree for straight lines and must diverge on tight curves."""
    t = np.arange(T_IN) * DT
    r, omega = 0.02, 6.0  # ~143 degrees of a small circle within one 5 cm token
    theta = omega * t
    pos = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(t)], axis=-1)
    curved_arm = np.concatenate(
        [pos, np.zeros((T_IN, 3)), np.linspace(0, 1, T_IN)[:, None]], axis=-1
    )
    chunk = _bimanual(curved_arm, _axis_line_arm(T_IN, speed=0.1))

    vels = {}
    for mode in ("mean_scalar", "per_step_scalar"):
        out = BimanualArcLengthTokenizer(_cfg(M=20, mode=mode)).tokenize(chunk)
        vels[mode] = out[:, 7]
    assert not np.isclose(vels["mean_scalar"][0], vels["per_step_scalar"][0], rtol=1e-3)


# --------------------------------------------------------------------------- #
# Transform-style pipeline adapter
# --------------------------------------------------------------------------- #
def test_transform_adapter() -> None:
    chunk = _bimanual(
        _make_arm_line(T_IN, total_dist=0.13, seed=8),
        _make_arm_line(T_IN, total_dist=0.07, seed=9),
    )
    batch = {"actions_cartesian": chunk.copy(), "unrelated": np.arange(3)}
    transform = TokenizeBimanualArcLength(
        action_key="actions_cartesian",
        output_action_key="actions_arc",
        config=_cfg(M=20),
    )
    out = transform.transform(batch)
    assert out is batch
    assert out["actions_arc"].shape == (20, ARC_BIMANUAL_DIM)
    np.testing.assert_allclose(out["actions_cartesian"], chunk)  # input untouched
    np.testing.assert_allclose(out["unrelated"], np.arange(3))
    # Adapter output matches calling the tokenizer directly.
    direct = BimanualArcLengthTokenizer(_cfg(M=20)).tokenize(chunk)
    np.testing.assert_allclose(out["actions_arc"], direct)


def test_velocity_mode_enum_values() -> None:
    assert VelocityMode("mean_scalar") is VelocityMode.MEAN_SCALAR
    assert VelocityMode("mean_per_dim") is VelocityMode.MEAN_PER_DIM
    assert VelocityMode("per_step_scalar") is VelocityMode.PER_STEP_SCALAR
    assert VelocityMode("per_step_per_dim") is VelocityMode.PER_STEP_PER_DIM
    assert velocity_dim("mean_scalar") == velocity_dim("per_step_scalar") == 1
    assert velocity_dim("mean_per_dim") == velocity_dim("per_step_per_dim") == 3
    with pytest.raises(ValueError):
        VelocityMode("not_a_mode")


# --------------------------------------------------------------------------- #
# Per-dim velocity modes
# --------------------------------------------------------------------------- #
def test_velocity_mean_per_dim_payload() -> None:
    """mean_per_dim payload is the displacement/time vector: for motion along
    +x it points along +x with near-zero other components."""
    speed = 0.1
    arm = _axis_line_arm(T=T_IN, speed=speed)
    tok = ArcLengthTokenizer(
        min_distance_unit=0.05, resampled_vector_length=20, mode="mean_per_dim", dt=DT
    )
    token = tok.tokenize_at(arm[:, 0:3], arm[:, 3:6], arm[:, 6:7], t=0)
    assert token.trans_vel.shape == (3,)
    assert speed <= token.trans_vel[0] < speed * 1.15
    np.testing.assert_allclose(token.trans_vel[1:], 0.0, atol=1e-12)


def test_velocity_per_step_per_dim_payload() -> None:
    """per_step_per_dim payload is (M, 3), constant rows for uniform motion."""
    speed = 0.1
    arm = _axis_line_arm(T=T_IN, speed=speed)
    M = 20
    tok = ArcLengthTokenizer(
        min_distance_unit=0.05,
        resampled_vector_length=M,
        mode="per_step_per_dim",
        dt=DT,
    )
    token = tok.tokenize_at(arm[:, 0:3], arm[:, 3:6], arm[:, 6:7], t=0)
    assert token.trans_vel.shape == (M, 3)
    np.testing.assert_allclose(
        token.trans_vel, np.broadcast_to(token.trans_vel[0], (M, 3)), atol=1e-9
    )
    assert speed * 0.85 < token.trans_vel[0, 0] < speed * 1.15
    np.testing.assert_allclose(token.trans_vel[:, 1:], 0.0, atol=1e-12)


def test_per_dim_zero_velocity_shapes() -> None:
    """Stationary trajectories produce per-dim zero payloads of the right shape."""
    arm = _make_arm_static(T=T_IN, seed=3)
    for mode, shape in (("mean_per_dim", (3,)), ("per_step_per_dim", (20, 3))):
        tok = ArcLengthTokenizer(
            min_distance_unit=0.05, resampled_vector_length=20, mode=mode, dt=DT
        )
        token = tok.tokenize_at(arm[:, 0:3], arm[:, 3:6], arm[:, 6:7], t=0)
        assert token.kind == "zero"
        assert token.trans_vel.shape == shape
        np.testing.assert_allclose(token.trans_vel, 0.0)


def test_bimanual_per_dim_layout_and_values() -> None:
    """mean_per_dim: (M, 20) layout with per-arm 3-vector velocity columns that
    point along each arm's own motion direction."""
    left = _axis_line_arm(T_IN, speed=0.10, axis=0)
    right = _axis_line_arm(T_IN, speed=0.05, axis=1, offset_z=0.5)
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=20, mode="mean_per_dim"))
    out = tokenizer.tokenize(_bimanual(left, right))
    assert out.shape == (20, 20)

    vel_left, vel_right = out[:, 7:10], out[:, 17:20]
    np.testing.assert_allclose(vel_left, np.broadcast_to(vel_left[0], (20, 3)))
    np.testing.assert_allclose(vel_right, np.broadcast_to(vel_right[0], (20, 3)))
    # Left moves along +x, right along +y — each velocity vector matches.
    assert 0.10 <= vel_left[0, 0] < 0.10 * 1.15
    np.testing.assert_allclose(vel_left[0, 1:], 0.0, atol=1e-12)
    assert 0.05 <= vel_right[0, 1] < 0.05 * 1.15
    np.testing.assert_allclose(vel_right[0, [0, 2]], 0.0, atol=1e-12)


def test_roundtrip_mean_per_dim_recovers_chunk() -> None:
    """The mean_per_dim round trip matches the mean_scalar one: the timing
    speed is the norm of the velocity vector."""
    T = 20
    cfg = _cfg(M=20, mode="mean_per_dim")
    left = _make_arm_line(T, total_dist=cfg.min_distance_unit, seed=1)
    right = _make_arm_line(T, total_dist=cfg.min_distance_unit, seed=2)
    tokenizer = BimanualArcLengthTokenizer(cfg)
    arc = tokenizer.tokenize(_bimanual(left, right))
    rec = tokenizer.detokenize(arc, action_horizon=T)
    assert rec.shape == (T, BIMANUAL_CARTESIAN_DIM)

    waypoint_spacing = cfg.min_distance_unit / (cfg.resampled_vector_length - 1)
    for arm_in, arm_start in ((left, 0), (right, ARM_DIM)):
        arm_rec = rec[:, arm_start : arm_start + ARM_DIM]
        assert np.abs(arm_rec[:, 0:3] - arm_in[:, 0:3]).max() < 1.5 * waypoint_spacing
        rot_err = max(
            _rotation_angle(arm_rec[i, 3:6], arm_in[i, 3:6]) for i in range(T)
        )
        assert rot_err < 0.05
        assert np.abs(arm_rec[:, 6] - arm_in[:, 6]).max() < 0.1


def test_core_detokenize_per_step_modes() -> None:
    """Core full-trajectory detokenize supports the per-step modes: a straight
    line reconstructs on-path with matching endpoints."""
    arm = _axis_line_arm(T=61, speed=0.1)  # 0.2 m total
    pos_in = arm[:, 0:3]
    for mode in ("per_step_scalar", "per_step_per_dim"):
        tok = ArcLengthTokenizer(
            min_distance_unit=0.05, resampled_vector_length=10, mode=mode, dt=DT
        )
        tokens = tok.tokenize(pos_in, arm[:, 3:6], arm[:, 6:7])
        pos_out, ypr_out, grip_out = tok.detokenize(tokens)
        np.testing.assert_allclose(pos_out[0], pos_in[0], atol=1e-9)
        np.testing.assert_allclose(pos_out[-1], pos_in[-1], atol=1e-9)
        max_off_path = max(_point_to_polyline_dist(p, pos_in) for p in pos_out)
        assert max_off_path < 1e-9
        # Progress is monotone and the step count is near the input's: each of
        # the M-1 waypoint segments per chunk rounds its step count up (ceil),
        # so the reconstruction can overshoot by at most M-1 steps per chunk.
        assert np.all(np.diff(pos_out[:, 0]) >= -1e-12)
        M = tok.M
        assert abs(len(pos_out) - len(pos_in)) <= len(tokens) * (M - 1)


# --------------------------------------------------------------------------- #
# Joint resampling
# --------------------------------------------------------------------------- #
def _linear_joints(T: int, J: int = 6, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return np.linspace(lo, hi, T)[:, None] * np.ones(J)[None, :]


def test_tokenize_with_joints_shapes_and_anchor() -> None:
    """Joints resample to (M, J) along each arm's own arc length; the first
    waypoint is anchored at the input joints at t=0, and an arm covering
    exactly one distance unit ends at the input's final joints."""
    T = 20
    cfg = _cfg(M=20)
    left = _make_arm_line(T, total_dist=cfg.min_distance_unit, seed=1)
    right = _make_arm_line(T, total_dist=cfg.min_distance_unit, seed=2)
    joints_left = _linear_joints(T)
    joints_right = _linear_joints(T, lo=1.0, hi=-1.0)
    tokenizer = BimanualArcLengthTokenizer(cfg)
    actions, jl, jr = tokenizer.tokenize_with_joints(
        _bimanual(left, right), joints_left, joints_right
    )
    assert actions.shape == (20, tokenizer.arc_dim)
    assert jl.shape == (20, 6) and jr.shape == (20, 6)
    np.testing.assert_allclose(jl[0], joints_left[0], atol=1e-12)
    np.testing.assert_allclose(jr[0], joints_right[0], atol=1e-12)
    np.testing.assert_allclose(jl[-1], joints_left[-1], atol=1e-9)
    np.testing.assert_allclose(jr[-1], joints_right[-1], atol=1e-9)
    # Constant-speed line: joints linear in time == linear in arc length, so
    # the resampled joints are uniformly spaced across waypoints.
    steps = np.diff(jl[:, 0])
    np.testing.assert_allclose(steps, steps[0], atol=1e-9)


def test_tokenize_without_joints_returns_none() -> None:
    chunk = _bimanual(
        _make_arm_line(T_IN, total_dist=0.13, seed=1),
        _make_arm_line(T_IN, total_dist=0.07, seed=2),
    )
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=20))
    actions, jl, jr = tokenizer.tokenize_with_joints(chunk)
    assert jl is None and jr is None
    np.testing.assert_allclose(actions, tokenizer.tokenize(chunk))


def test_tokenize_with_joints_zero_arm_holds_initial() -> None:
    """A stationary arm's joints are held at their initial value (zero-token
    semantics), while the moving arm's joints resample normally."""
    left = _axis_line_arm(T_IN, speed=0.10)
    right = _make_arm_static(T_IN, seed=7)
    joints_left = _linear_joints(T_IN)
    joints_right = _linear_joints(T_IN)  # vary in time, but arm is static
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=20))
    _, jl, jr = tokenizer.tokenize_with_joints(
        _bimanual(left, right), joints_left, joints_right
    )
    np.testing.assert_allclose(jr, np.broadcast_to(joints_right[0], (20, 6)))
    assert jl[-1, 0] > jl[0, 0]  # moving arm's joints actually progress


def test_detokenize_with_joints_roundtrip() -> None:
    """Joints survive the tokenize -> detokenize round trip within the same
    timing-quantization bound as the gripper channel."""
    T = 20
    cfg = _cfg(M=20)
    left = _make_arm_line(T, total_dist=cfg.min_distance_unit, seed=1)
    right = _make_arm_line(T, total_dist=cfg.min_distance_unit, seed=2)
    joints_left = _linear_joints(T)
    joints_right = _linear_joints(T, lo=0.5, hi=-0.5)
    tokenizer = BimanualArcLengthTokenizer(cfg)
    arc, jl, jr = tokenizer.tokenize_with_joints(
        _bimanual(left, right), joints_left, joints_right
    )
    rec, jl_rec, jr_rec = tokenizer.detokenize_with_joints(
        arc, action_horizon=T, joints_left=jl, joints_right=jr
    )
    assert jl_rec.shape == (T, 6) and jr_rec.shape == (T, 6)
    assert np.abs(jl_rec - joints_left).max() < 0.1
    assert np.abs(jr_rec - joints_right).max() < 0.1


def test_detokenize_with_joints_stationary_holds() -> None:
    T = 20
    cfg = _cfg(M=20)
    left = _make_arm_static(T, seed=5)
    right = _make_arm_static(T, seed=6)
    joints = _linear_joints(T)
    tokenizer = BimanualArcLengthTokenizer(cfg)
    arc, jl, jr = tokenizer.tokenize_with_joints(_bimanual(left, right), joints, joints)
    _, jl_rec, jr_rec = tokenizer.detokenize_with_joints(
        arc, action_horizon=T, joints_left=jl, joints_right=jr
    )
    np.testing.assert_allclose(jl_rec, np.broadcast_to(joints[0], (T, 6)), atol=1e-12)
    np.testing.assert_allclose(jr_rec, np.broadcast_to(joints[0], (T, 6)), atol=1e-12)


def test_joints_validation_and_sentinel() -> None:
    chunk = _bimanual(
        _make_arm_line(T_IN, total_dist=0.13, seed=1),
        _make_arm_line(T_IN, total_dist=0.07, seed=2),
    )
    tokenizer = BimanualArcLengthTokenizer(_cfg(M=20))
    with pytest.raises(ValueError, match="joints_left"):
        tokenizer.tokenize_with_joints(chunk, joints_left=_linear_joints(T_IN + 1))
    with pytest.raises(ValueError, match="joints_right"):
        tokenizer.detokenize_with_joints(
            np.zeros((20, 16)), action_horizon=5, joints_right=np.zeros((19, 6))
        )
    # Sentinel chunks poison the joints output too.
    bad = chunk.copy()
    bad[0, 0] = 1e9
    actions, jl, jr = tokenizer.tokenize_with_joints(
        bad, _linear_joints(T_IN), _linear_joints(T_IN)
    )
    np.testing.assert_allclose(actions, INVALID_POSE_FILL)
    np.testing.assert_allclose(jl, INVALID_POSE_FILL)
    np.testing.assert_allclose(jr, INVALID_POSE_FILL)


def test_transform_adapter_with_joints() -> None:
    chunk = _bimanual(
        _make_arm_line(T_IN, total_dist=0.13, seed=8),
        _make_arm_line(T_IN, total_dist=0.07, seed=9),
    )
    batch = {
        "actions_cartesian": chunk.copy(),
        "left.cmd_joint_pos": _linear_joints(T_IN),
        "right.cmd_joint_pos": _linear_joints(T_IN, lo=1.0, hi=0.0),
    }
    transform = TokenizeBimanualArcLength(
        action_key="actions_cartesian",
        output_action_key="actions_arc",
        config=_cfg(M=20),
        left_joint_key="left.cmd_joint_pos",
        right_joint_key="right.cmd_joint_pos",
    )
    out = transform.transform(batch)
    assert out["actions_arc"].shape == (20, 16)
    assert out["left.cmd_joint_pos_arc"].shape == (20, 6)
    assert out["right.cmd_joint_pos_arc"].shape == (20, 6)
    # Matches the direct tokenizer call.
    _, jl, jr = BimanualArcLengthTokenizer(_cfg(M=20)).tokenize_with_joints(
        chunk, batch["left.cmd_joint_pos"], batch["right.cmd_joint_pos"]
    )
    np.testing.assert_allclose(out["left.cmd_joint_pos_arc"], jl)
    np.testing.assert_allclose(out["right.cmd_joint_pos_arc"], jr)
