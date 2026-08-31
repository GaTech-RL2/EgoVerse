import torch
import pytest

from egomimic.utils.action_utils import BaseActionConverter, RobotBimanualCartesianEuler


def _stats_for() -> dict[str, torch.Tensor]:
    # Deliberately asymmetric per-dim ranges exercise non-rotation normalization.
    q1 = torch.tensor(
        [-1.0, -2.0, -3.0, -torch.pi, -1.0, -1.0, 0.0] * 2,
        dtype=torch.float32,
    )
    q99 = torch.tensor(
        [1.0, 2.0, 3.0, torch.pi, 1.0, 1.0, 1.0] * 2,
        dtype=torch.float32,
    )
    return {
        "mean": torch.zeros_like(q1),
        "std": torch.ones_like(q1),
        "min": q1,
        "max": q99,
        "quantile_1": q1,
        "quantile_99": q99,
    }


def _normalize(raw: torch.Tensor, stats: dict[str, torch.Tensor], norm_mode: str) -> torch.Tensor:
    if norm_mode == "zscore":
        return (raw - stats["mean"]) / (stats["std"] + 1e-6)
    if norm_mode == "minmax":
        return 2.0 * ((raw - stats["min"]) / (stats["max"] - stats["min"] + 1e-6)) - 1.0
    if norm_mode == "quantile":
        return 2.0 * (
            (raw - stats["quantile_1"])
            / (stats["quantile_99"] - stats["quantile_1"] + 1e-6)
        ) - 1.0
    raise AssertionError(f"unexpected norm mode {norm_mode}")


@pytest.mark.parametrize("norm_mode", ["zscore", "minmax", "quantile"])
def test_raw_rotation_encoding_round_trips_robot_bimanual_actions(norm_mode):
    converter = RobotBimanualCartesianEuler()
    raw = torch.tensor(
        [
            [
                [
                    0.25,
                    -0.5,
                    1.0,
                    0.3,
                    -0.2,
                    0.1,
                    0.75,
                    -0.2,
                    0.4,
                    -1.2,
                    -0.4,
                    0.25,
                    -0.15,
                    0.2,
                ]
            ]
        ],
        dtype=torch.float32,
    )
    stats = _stats_for()

    packed = converter.to32_raw_rotation(raw, stats=stats, norm_mode=norm_mode)
    decoded = converter.from32_raw_rotation(
        packed,
        stats=stats,
        norm_mode=norm_mode,
        unnormalize_non_rotation=True,
    )

    torch.testing.assert_close(decoded, raw, atol=1e-5, rtol=1e-5)


def test_raw_rotation_encoding_preserves_yaw_wrap_continuity():
    converter = RobotBimanualCartesianEuler()
    eps = 1e-4
    raw = torch.zeros(1, 2, 14, dtype=torch.float32)
    raw[:, 0, 3] = torch.pi - eps
    raw[:, 1, 3] = -torch.pi + eps
    stats = _stats_for()
    normalized = _normalize(raw, stats, "quantile")

    legacy_packed = converter.to32(normalized)
    fixed_packed = converter.to32_raw_rotation(
        raw,
        normalized_actions=normalized,
        stats=stats,
        norm_mode="quantile",
    )

    legacy_rot_distance = torch.linalg.norm(
        legacy_packed[0, 0, 3:9] - legacy_packed[0, 1, 3:9]
    )
    fixed_rot_distance = torch.linalg.norm(
        fixed_packed[0, 0, 3:9] - fixed_packed[0, 1, 3:9]
    )

    assert legacy_rot_distance > 2.0
    assert fixed_rot_distance < 1e-3


def test_base_converter_rejects_raw_rotation_encoding():
    converter = BaseActionConverter()
    actions = torch.zeros(1, 1, 14)

    with pytest.raises(NotImplementedError, match="raw-rotation action encoding"):
        converter.to32_raw_rotation(actions)
