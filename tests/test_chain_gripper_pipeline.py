from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.pushshapes import (
    ChainGripperPointArcLengthRolloutAdapter,
    ChainGripperPointRolloutAdapter,
)
from egomimic.pipeline.stages_sampler import NativeActionMSELoss
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes import get_keymap_hpt
from egomimic.rldb.zarr.action_chunk_transforms import RequireLastDim
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeChainGripperPointArcLength,
    chain_gripper_arc_embedding_to_points,
    chain_gripper_point_step_norm,
    chain_gripper_points_to_arc_embedding,
)
from egomimic.robot.rollout import PolicyRollout
from Tsimulation.sim_v2.pushshapes.chain_gripper_control import (
    pose_control_to_points,
)


def _translation_controls(count: int = 101) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(100.0, 300.0, count),
            np.full(count, 240.0),
            np.full(count, 0.35),
            np.full(count, 0.4),
        ]
    )


def test_chain_point_phi_embedding_round_trip_and_metric() -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(2, 3, 6))
    embedding = chain_gripper_points_to_arc_embedding(points)
    np.testing.assert_allclose(
        chain_gripper_arc_embedding_to_points(embedding),
        points,
        atol=1e-12,
    )

    delta = np.tile(np.array([3.0, 4.0]), 3)
    assert chain_gripper_point_step_norm(delta) == pytest.approx(5.0)

    # Translation and relative tip articulation occupy orthogonal coordinates.
    tip_only = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    assert chain_gripper_point_step_norm(tip_only) == pytest.approx(1.0)
    assert chain_gripper_point_step_norm(delta + tip_only) == pytest.approx(
        np.sqrt(26.0)
    )


def test_chain_point_arc_tokenizer_round_trips_rigid_translation() -> None:
    controls = _translation_controls()
    points = pose_control_to_points(controls).astype(np.float32)
    tokenizer = TokenizeChainGripperPointArcLength(
        min_distance_unit=200.0,
        resampled_vector_length=25,
        dt=1.0 / 30.0,
    )

    token = tokenizer.transform({"actions": points.copy()})["actions"]
    decoded = tokenizer.detokenize(token, action_horizon=len(points))

    assert token.shape == (26, 6)
    assert token.dtype == np.float32
    np.testing.assert_allclose(
        token[0],
        chain_gripper_points_to_arc_embedding(points[0]),
        atol=1e-5,
    )
    np.testing.assert_allclose(token[-1], [60.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(decoded, points, atol=2e-5)


def test_chain_point_phi_detokenizer_returns_points_for_degenerate_paths() -> None:
    tokenizer = TokenizeChainGripperPointArcLength(
        min_distance_unit=200.0,
        resampled_vector_length=5,
    )
    base_points = pose_control_to_points(np.array([220.0, 180.0, 0.4, 0.3]))
    base_phi = chain_gripper_points_to_arc_embedding(base_points)

    stationary_token = np.concatenate(
        [np.repeat(base_phi[None], 5, axis=0), np.zeros((1, 6))],
        axis=0,
    )
    stationary = tokenizer.detokenize(stationary_token, action_horizon=9)
    np.testing.assert_allclose(stationary, np.repeat(base_points[None], 9, axis=0))

    translation_loop = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
    )
    loop_points = base_points + np.tile(translation_loop, (1, 3))
    loop_phi = chain_gripper_points_to_arc_embedding(loop_points)
    loop_token = np.concatenate([loop_phi, np.zeros((1, 6))], axis=0)
    decoded_loop = tokenizer.detokenize(loop_token, action_horizon=9)
    assert np.isfinite(decoded_loop).all()
    np.testing.assert_allclose(decoded_loop[0], base_points, atol=1e-12)
    np.testing.assert_allclose(decoded_loop[-1], base_points, atol=1e-12)


def test_chain_point_rollout_adapter_uses_shared_sim_projection() -> None:
    controls = _translation_controls(count=8)
    points = pose_control_to_points(controls)
    decoded = ChainGripperPointRolloutAdapter().decode(
        torch.from_numpy(points).unsqueeze(0)
    )

    assert decoded.shape == (1, 8, 4)
    torch.testing.assert_close(
        decoded[0], torch.from_numpy(controls), atol=1e-9, rtol=0
    )

    noisy = points.copy()
    noisy[:, [0, 3, 4]] += np.array([7.0, -5.0, 4.0])
    projected = ChainGripperPointRolloutAdapter().decode(noisy[None])
    assert projected.shape == (1, 8, 4)
    assert np.isfinite(projected).all()
    assert np.logical_and(projected[..., :2] >= 0.0, projected[..., :2] <= 512.0).all()
    assert np.logical_and(projected[..., 3] >= 0.0, projected[..., 3] <= 1.0).all()


def test_chain_point_rollout_adapter_temporal_fallback_and_diagnostics() -> None:
    initial_controls = np.array(
        [
            [180.0, 210.0, 1.1, 0.35],
            [310.0, 260.0, -0.8, 0.65],
        ]
    )
    final_controls = initial_controls.copy()
    final_controls[:, 0] += 5.0
    valid_start = pose_control_to_points(initial_controls)
    valid_end = pose_control_to_points(final_controls)
    collapsed = np.repeat(initial_controls[:, None, :2], 3, axis=1).reshape(2, 6)
    one_ray = valid_start.copy()
    one_ray[:, 0:2] = one_ray[:, 2:4]
    predicted = np.stack([valid_start, collapsed, one_ray, valid_end], axis=1)

    adapter = ChainGripperPointRolloutAdapter(action_horizon=4)
    decoded = adapter.decode(torch.from_numpy(predicted))

    assert decoded.shape == (2, 4, 4)
    torch.testing.assert_close(
        decoded[:, 0], torch.from_numpy(initial_controls), atol=1e-9, rtol=0
    )
    torch.testing.assert_close(
        decoded[:, 3], torch.from_numpy(final_controls), atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(
        decoded[:, 1:3, 2].numpy(),
        np.repeat(initial_controls[:, None, 2], 2, axis=1),
    )

    diagnostics = adapter.last_projection_diagnostics
    assert diagnostics is not None
    assert diagnostics["point_rmse"].shape == (2, 4)
    assert diagnostics["degenerate"][:, 1:3].all()
    assert diagnostics["used_exact_inverse"][:, [0, 3]].all()
    assert diagnostics["degenerate_count"] == 4
    assert diagnostics["wrong_chirality_count"] == 4
    assert diagnostics["max_point_rmse"] >= diagnostics["mean_point_rmse"] >= 0.0


def test_chain_point_rollout_adapter_uses_context_for_first_degenerate_row() -> None:
    centers = np.array([[100.0, 120.0], [300.0, 330.0]])
    collapsed = np.repeat(centers[:, None], 3, axis=1).reshape(2, 1, 6)
    state = np.array(
        [
            [100.0, 120.0, 0.75, 0.0, 0.0, 0.0],
            [300.0, 330.0, -1.25, 0.0, 0.0, 0.0],
        ]
    )
    decoded = ChainGripperPointRolloutAdapter(action_horizon=1).decode(
        collapsed,
        context={"state_agent_obj": state},
    )
    np.testing.assert_allclose(decoded[:, 0, 2], state[:, 2], atol=1e-12)


def test_chain_point_arc_adapter_detokenizes_then_projects_to_native4() -> None:
    controls = _translation_controls()
    points = pose_control_to_points(controls)
    tokenizer = TokenizeChainGripperPointArcLength(
        min_distance_unit=200.0,
        resampled_vector_length=25,
    )
    token = tokenizer.transform({"actions": points.copy()})["actions"]
    adapter = ChainGripperPointArcLengthRolloutAdapter(
        min_distance_unit=200.0,
        resampled_vector_length=25,
        action_horizon=len(points),
    )

    decoded = adapter.decode(torch.from_numpy(token).unsqueeze(0))

    assert decoded.shape == (1, len(points), 4)
    torch.testing.assert_close(
        decoded[0], torch.from_numpy(controls), atol=2e-6, rtol=0
    )


def test_chain_point_keymap_uses_additive_array_and_rejects_native4() -> None:
    keymap = get_keymap_hpt(
        action_horizon=16,
        action_zarr_key="actions.points",
    )
    assert keymap["actions"]["zarr_key"] == "actions.points"
    assert keymap["actions"]["horizon"] == 16

    validator = RequireLastDim(keys=["actions"], width=6)
    validator.transform({"actions": np.zeros((16, 6), dtype=np.float32)})
    with pytest.raises(ValueError, match="last dimension 6"):
        validator.transform({"actions": np.zeros((16, 4), dtype=np.float32)})


class _IdentityPushNormStats:
    def keys_of_type(self, key_type, emb_id):
        return {
            "proprio_keys": [],
            "lang_keys": [],
            "camera_keys": [],
            "action_keys": ["actions"],
        }[key_type]

    def is_key_with_embodiment(self, key, emb_id):
        return key == "actions"

    def zarr_key_to_keyname(self, key, emb_id):
        return key if key == "actions" else None

    def keyname_to_zarr_key(self, key, emb_id):
        return key if key == "actions" else None

    def normalize(self, data, emb_id):
        return dict(data)

    def unnormalize(self, data, emb_id):
        return dict(data)


class _NoObsCondition(Stage):
    reads = ["embodiment"]
    writes = ["condition", "target"]
    rollout_obs_steps = 1

    def forward(self, batch):
        batch["condition"] = torch.zeros(1, 1)
        return batch


class _SixDimPrediction(Stage):
    reads = ["condition", "embodiment"]
    writes = ["pred_action"]

    def forward(self, batch):
        batch["pred_action"] = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
        return batch


class _SliceAdapter:
    preserves_decoded_timing = True

    def __init__(self, width):
        self.width = int(width)

    def decode(self, actions, context=None):
        return actions[..., : self.width]


def test_pipeline_algo_selects_per_domain_adapter_and_keeps_singleton_fallback() -> (
    None
):
    singleton = _SliceAdapter(2)
    u_adapter = _SliceAdapter(3)
    chain_adapter = _SliceAdapter(4)
    algo = PipelineAlgo(
        stages=[_NoObsCondition(), _SixDimPrediction(), NativeActionMSELoss()],
        norm_stats=_IdentityPushNormStats(),
        domains=["pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"],
        ac_keys={
            "pushshapes_sim_u_socket": "actions",
            "pushshapes_sim_chain_gripper": "actions",
        },
        rollout_adapter=singleton,
        rollout_adapters={
            "pushshapes_sim_u_socket": u_adapter,
            "pushshapes_sim_chain_gripper": chain_adapter,
        },
        action_horizon=2,
        device=torch.device("cpu"),
    )
    processed = algo.process_batch_for_rollout(
        {"pushshapes_sim_u_socket": {}, "pushshapes_sim_chain_gripper": {}}
    )

    predictions = algo.forward_rollout(processed)

    assert algo.rollout_adapter is singleton
    assert algo.rollout_adapter_for("pushshapes_sim_u_socket") is u_adapter
    assert (
        algo.rollout_adapter_for(get_embodiment_id("pushshapes_sim_chain_gripper"))
        is chain_adapter
    )
    assert predictions["pushshapes_sim_u_socket_actions"].shape == (1, 2, 3)
    assert predictions["pushshapes_sim_chain_gripper_actions"].shape == (1, 2, 4)
    assert predictions["pushshapes_sim_chain_gripper_actions_tokens"].shape == (1, 2, 6)

    fallback_algo = PipelineAlgo(
        stages=[_NoObsCondition(), _SixDimPrediction(), NativeActionMSELoss()],
        norm_stats=_IdentityPushNormStats(),
        domains=["pushshapes_sim_chain_gripper"],
        ac_keys={"pushshapes_sim_chain_gripper": "actions"},
        rollout_adapter=singleton,
        action_horizon=2,
        device=torch.device("cpu"),
    )
    assert (
        fallback_algo.rollout_adapter_for("pushshapes_sim_chain_gripper") is singleton
    )


def test_robot_rollout_contract_uses_domain_adapter_lookup() -> None:
    adapter = SimpleNamespace(
        action_horizon=100,
        preserves_decoded_timing=True,
    )
    model = SimpleNamespace(
        domains=["pushshapes_sim_chain_gripper"],
        rollout_adapter=None,
        rollout_adapter_for=lambda domain: adapter,
    )
    assert (
        PolicyRollout._resolve_action_chunk_contract(
            model=model,
            embodiment_name="pushshapes_sim_chain_gripper",
            arm="both",
            cartesian=True,
            query_frequency=8,
            requested_resampled_len=25,
        )
        == 100
    )


def test_robot_rollout_contract_accepts_direct_chain_point_adapter() -> None:
    adapter = ChainGripperPointRolloutAdapter(action_horizon=16)
    model = SimpleNamespace(
        domains=["pushshapes_sim_chain_gripper"],
        rollout_adapter=None,
        rollout_adapter_for=lambda domain: adapter,
    )
    assert (
        PolicyRollout._resolve_action_chunk_contract(
            model=model,
            embodiment_name="pushshapes_sim_chain_gripper",
            arm="both",
            cartesian=True,
            query_frequency=8,
            requested_resampled_len=16,
        )
        == 16
    )
