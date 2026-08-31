import numpy as np
import pytest
import torch

from egomimic.pipeline.losses import conditional_energy_score
from egomimic.pipeline.pushshapes import (
    ChainGripperPointActionCanonicalizer,
    ChainGripperPointRolloutAdapter,
    USocketRotVecActionCanonicalizer,
    USocketRotVecRolloutAdapter,
)
from egomimic.pipeline.stages_sampler import PerEmbodimentActionCanonicalizer
from Tsimulation.sim_v2.pushshapes.chain_gripper_control import (
    pose_control_to_points,
)


def _normalize(value, minimum, maximum):
    minimum = torch.as_tensor(minimum, dtype=torch.float32)
    maximum = torch.as_tensor(maximum, dtype=torch.float32)
    return 2.0 * (value.float() - minimum) / (maximum - minimum + 1e-6) - 1.0


def _unnormalize(value, minimum, maximum):
    minimum = torch.as_tensor(minimum, dtype=torch.float32)
    maximum = torch.as_tensor(maximum, dtype=torch.float32)
    return (value.float() + 1.0) * 0.5 * (maximum - minimum + 1e-6) + minimum


def test_usocket_canonicalizer_removes_radius_and_out_of_arena_null_directions():
    canonicalizer = USocketRotVecActionCanonicalizer(world_size=512.0)
    physical = torch.tensor(
        [
            [-10.0, 700.0, 3.0, 4.0],
            [0.0, 512.0, 30.0, 40.0],
        ]
    )
    canonical = canonicalizer(physical)

    torch.testing.assert_close(canonical[0], canonical[1])
    torch.testing.assert_close(
        torch.linalg.vector_norm(canonical[:, 2:4], dim=-1), torch.ones(2)
    )
    torch.testing.assert_close(canonicalizer(canonical), canonical)
    decoded = USocketRotVecRolloutAdapter().decode(canonical[:, None])[:, 0]
    torch.testing.assert_close(
        decoded[:, :2], torch.tensor([[0.0, 512.0]]).repeat(2, 1)
    )
    torch.testing.assert_close(
        decoded[:, 2],
        torch.full((2,), torch.atan2(torch.tensor(4.0), torch.tensor(3.0))),
    )


def test_usocket_canonicalization_uses_physical_units_with_asymmetric_stats():
    minimum = [-20.0, 5.0, -1.0, -1.0]
    maximum = [700.0, 600.0, 1.0, 1.0]
    stage = PerEmbodimentActionCanonicalizer(
        {"u": USocketRotVecActionCanonicalizer(world_size=512.0)},
        input_key="raw_pred_action",
        target_output_key="canonical_target",
    )
    stage.bind_action_normalization(
        "u", norm_mode="minmax", stats={"min": minimum, "max": maximum}
    )
    physical = torch.tensor([[[600.0, -5.0, -6.0, 8.0]]])
    normalized = _normalize(physical, minimum, maximum)
    output = stage(
        {
            "embodiment": "u",
            "raw_pred_action": normalized.clone(),
            "raw_pred_action_samples": normalized[:, None].clone(),
            "target": normalized.clone(),
        }
    )
    recovered = _unnormalize(output["pred_action"], minimum, maximum)
    expected = USocketRotVecActionCanonicalizer(world_size=512.0)(physical)
    torch.testing.assert_close(recovered, expected, atol=2e-5, rtol=0.0)


@pytest.mark.parametrize(
    "rotvec", [(0.0, 0.0), (1e-12, -1e-12), (1.0, 0.0), (-1.0, 1e-8)]
)
def test_usocket_zero_and_near_zero_have_finite_backward(rotvec):
    actions = torch.tensor([[[10.0, 20.0, *rotvec]]], requires_grad=True)
    output = USocketRotVecActionCanonicalizer()(actions)
    output.square().sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(actions.grad).all()


def test_usocket_radial_only_sample_diversity_has_zero_energy_distance():
    samples = torch.tensor([[[[10.0, 20.0, 0.6, 0.8]], [[10.0, 20.0, 6.0, 8.0]]]])
    canonical = USocketRotVecActionCanonicalizer()(samples)
    target = canonical[:, 0]
    result = conditional_energy_score(canonical, target)
    assert result["pairwise_distance"].item() == pytest.approx(0.0)
    assert result["score"].item() == pytest.approx(0.0)


def test_chain_canonicalizer_is_idempotent_on_authoritative_fk_points():
    controls = np.array(
        [
            [0.0, 512.0, -np.pi, 0.0],
            [512.0, 0.0, np.pi - 1e-4, 1.0],
            [220.0, 180.0, 0.4, 0.3],
            [250.0, 200.0, -2.4, 0.8],
        ]
    )
    points = torch.from_numpy(pose_control_to_points(controls)).float()
    canonicalizer = ChainGripperPointActionCanonicalizer()
    canonical = canonicalizer(points)

    torch.testing.assert_close(canonical, points, atol=8e-5, rtol=0.0)
    torch.testing.assert_close(canonicalizer(canonical), canonical, atol=8e-5, rtol=0.0)


@pytest.mark.parametrize(
    "points",
    [
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 140.0, 100.0],
        [60.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [140.0, 80.0, 100.0, 100.0, 60.0, 120.0],
        [-500.0, 900.0, -20.0, 700.0, 1000.0, -300.0],
    ],
)
def test_chain_off_manifold_and_degenerate_inputs_have_finite_backward(points):
    actions = torch.tensor([[points]], requires_grad=True)
    output = ChainGripperPointActionCanonicalizer()(actions)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(actions.grad).all()


def test_chain_canonical_output_matches_production_ik_at_fp32_tolerance():
    controls = np.array([[[220.0, 180.0, 0.4, 0.3], [250.0, 200.0, -2.4, 0.8]]])
    points = torch.from_numpy(pose_control_to_points(controls)).float()
    canonical = ChainGripperPointActionCanonicalizer()(points)
    adapter = ChainGripperPointRolloutAdapter(action_horizon=2, input_is_canonical=True)
    decoded = adapter.decode(canonical)

    torch.testing.assert_close(
        decoded.float(), torch.from_numpy(controls).float(), atol=2e-4, rtol=0.0
    )
    diagnostics = adapter.last_projection_diagnostics
    assert diagnostics is not None
    assert diagnostics["max_point_rmse"] < 1e-4
    assert diagnostics["used_exact_inverse"].all()
    assert diagnostics["wrong_chirality_count"] == 0
    assert diagnostics["degenerate_count"] == 0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an accelerator tensor"
)
def test_chain_canonical_rollout_keeps_accelerator_device_without_numpy_boundary():
    controls = torch.tensor(
        [[[220.0, 180.0, 0.4, 0.3], [250.0, 200.0, -2.4, 0.8]]],
        device="mps",
    )
    points = ChainGripperPointActionCanonicalizer.native_to_points(controls)
    decoded = ChainGripperPointRolloutAdapter(
        action_horizon=2, input_is_canonical=True
    ).decode(points)

    assert decoded.device.type == "mps"
    assert decoded.dtype == points.dtype
    torch.testing.assert_close(decoded.cpu(), controls.cpu(), atol=2e-4, rtol=0.0)


def test_canonicalization_stage_replaces_train_eval_and_rollout_tensor():
    stage = PerEmbodimentActionCanonicalizer(
        {"u": USocketRotVecActionCanonicalizer()},
        input_key="raw_pred_action",
        target_output_key="canonical_target",
    )
    stage.bind_action_normalization(
        "u",
        norm_mode="minmax",
        stats={"min": [0.0, 0.0, -1.0, -1.0], "max": [512.0, 512.0, 1.0, 1.0]},
    )
    raw = torch.tensor([[[0.0, 0.0, 0.2, 0.4]]])
    grouped = raw[:, None].repeat(1, 2, 1, 1)
    train = stage(
        {
            "embodiment": "u",
            "raw_pred_action": raw.clone(),
            "raw_pred_action_samples": grouped.clone(),
            "target": raw.clone(),
        }
    )
    rollout = stage(
        {
            "embodiment": "u",
            "raw_pred_action": raw.clone(),
            "raw_pred_action_samples": raw[:, None].clone(),
        }
    )

    assert "raw_pred_action" in train and "raw_target" in train
    assert "raw_pred_action" in rollout and "canonical_target" not in rollout
    assert "canonical_target" in train
    torch.testing.assert_close(train["pred_action"], train["pred_action_samples"][:, 0])
    torch.testing.assert_close(train["pred_action"], rollout["pred_action"])
    assert not torch.equal(train["raw_pred_action"], train["pred_action"])


def test_canonicalizer_historical_default_batch_keys_remain_compatible():
    stage = PerEmbodimentActionCanonicalizer({"u": USocketRotVecActionCanonicalizer()})
    stage.bind_action_normalization(
        "u",
        norm_mode="minmax",
        stats={"min": [0.0, 0.0, -1.0, -1.0], "max": [512.0, 512.0, 1.0, 1.0]},
    )
    raw = torch.tensor([[[0.0, 0.0, 0.2, 0.4]]])
    grouped = raw[:, None].repeat(1, 2, 1, 1)

    output = stage(
        {
            "embodiment": "u",
            "pred_action": raw.clone(),
            "target": raw.clone(),
        }
    )

    assert output["raw_pred_action"] is not output["pred_action"]
    torch.testing.assert_close(output["raw_pred_action"], raw)
    torch.testing.assert_close(output["raw_pred_action_samples"], grouped[:, :1])
    torch.testing.assert_close(output["target"], output["pred_action"])
