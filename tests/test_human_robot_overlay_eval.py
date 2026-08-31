from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.eval_video import EvalVideo
from egomimic.eval.human_robot_overlay_eval import HumanRobotOverlayEval
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.pushshapes import USocketRotVecRolloutAdapter
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import (
    build_fold_keypoint_wristframe_revert_transform_list,
)
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


class IdentityNorm:
    def unnormalize(self, batch, _emb_id):
        return batch


def _evaluator(emb_id, action_key, prediction, **kwargs):
    evaluator = HumanRobotOverlayEval(**kwargs)
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: action_key},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: {f"emb{emb_id}_{action_key}": prediction},
    )
    return evaluator


def test_metrics_score_the_full_unnormalized_denoised_chunk():
    emb_id = get_embodiment_id("eva_bimanual")
    action_key = "actions_cartesian"
    target = torch.zeros(2, 4, 3)
    prediction = target.clone()
    prediction[:, -1] = 2.0
    evaluator = _evaluator(
        emb_id, action_key, prediction, viz_func=None, frame_stride=1
    )

    metrics, images = evaluator.compute_metrics_and_viz({emb_id: {action_key: target}})

    prefix = f"Valid/emb{emb_id}_{action_key}_action"
    assert metrics[f"{prefix}_mse"].item() == 1.0
    assert metrics[f"{prefix}_squared_error_median"].item() == 0.0
    assert metrics[f"{prefix}_squared_error_max"].item() == 4.0
    assert metrics[f"Valid/emb{emb_id}_{action_key}_copybaseline_mse"] == 0.0
    assert images == {}


def test_metrics_include_per_domain_and_cotrain_mse_aliases():
    emb_ids = [
        get_embodiment_id("eva_bimanual"),
        get_embodiment_id("human_bimanual"),
    ]
    action_key = "actions_cartesian"
    targets = {
        emb_ids[0]: torch.zeros(2, 4, 3),
        emb_ids[1]: torch.zeros(2, 4, 3),
    }
    predictions = {
        f"emb{emb_ids[0]}_{action_key}": torch.ones(2, 4, 3),
        f"emb{emb_ids[1]}_{action_key}": torch.full((2, 4, 3), 2.0),
    }
    evaluator = HumanRobotOverlayEval(viz_func=None)
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: action_key for emb_id in emb_ids},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: predictions,
    )

    metrics, _ = evaluator.compute_metrics_and_viz(
        {emb_id: {action_key: targets[emb_id]} for emb_id in emb_ids}
    )

    assert metrics["Valid/MSE/eva_bimanual"].item() == 1.0
    assert metrics["Valid/MSE/human_bimanual"].item() == 4.0
    assert metrics["Valid/MSE"].item() == 2.5
    assert metrics["Valid/Native_MSE"].item() == 2.5


def test_grouped_validation_logs_energy_score_and_k_sample_mse():
    emb_id = get_embodiment_id("eva_bimanual")
    action_key = "actions_cartesian"
    target = torch.zeros(1, 1, 1)
    samples = torch.tensor([[[[-1.0]], [[1.0]]]])
    predictions = {
        f"emb{emb_id}_{action_key}": samples[:, 0],
        f"emb{emb_id}_{action_key}_samples": samples,
    }
    evaluator = HumanRobotOverlayEval(viz_func=None)
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: action_key},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: predictions,
    )

    metrics, _ = evaluator.compute_metrics_and_viz({emb_id: {action_key: target}})

    assert metrics["Valid/MSE/eva_bimanual"].item() == 1.0
    assert metrics["Valid/Native_MSE/eva_bimanual"].item() == 1.0
    assert metrics["Valid/EnergyScore/eva_bimanual"].item() == 0.0
    assert metrics["Valid/EnergyAttraction/eva_bimanual"].item() == 1.0
    assert metrics["Valid/EnergyRepulsion/eva_bimanual"].item() == 1.0
    assert metrics["Valid/PairwiseDistance/eva_bimanual"].item() == 2.0


def test_native_mse_decodes_rotvec_and_wraps_theta_circularly():
    emb_id = get_embodiment_id("pushshapes_sim_u_socket")
    action_key = "actions"
    target_theta = torch.tensor(torch.pi - 0.1)
    prediction_theta = torch.tensor(-torch.pi + 0.1)
    target = torch.tensor(
        [[[[10.0, 20.0, torch.cos(target_theta), torch.sin(target_theta)]]]]
    ).squeeze(1)
    prediction = torch.tensor(
        [[[10.0, 20.0, torch.cos(prediction_theta), torch.sin(prediction_theta)]]]
    )
    evaluator = HumanRobotOverlayEval(viz_func=None)
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: action_key},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: {f"emb{emb_id}_{action_key}": prediction},
        rollout_adapter_for=lambda _emb_id: USocketRotVecRolloutAdapter(),
    )

    metrics, _ = evaluator.compute_metrics_and_viz({emb_id: {action_key: target}})

    assert metrics["Valid/Native_MSE/pushshapes_sim_u_socket"].item() == pytest.approx(
        0.2**2 / 3.0, rel=1e-5
    )
    assert metrics[
        "Valid/ActionToken_MSE/pushshapes_sim_u_socket"
    ].item() == pytest.approx(((prediction - target) ** 2).mean().item())


def test_training_log_info_includes_per_domain_and_cotrain_mse_aliases():
    algo = SimpleNamespace(domain_by_id={3: "u_socket", 7: "chain_grabber"})
    info = {
        "losses": {
            "action_loss": torch.tensor(2.5),
            "3_loss_native_action": torch.tensor(1.0),
            "7_loss_native_action": torch.tensor(4.0),
        }
    }

    logged = PipelineAlgo.log_info(algo, info)

    assert logged["MSE/u_socket"] == 1.0
    assert logged["MSE/chain_grabber"] == 4.0
    assert logged["MSE"] == 2.5


def test_training_log_info_aliases_energy_objective_and_diagnostic_mse():
    algo = SimpleNamespace(domain_by_id={3: "u_socket", 7: "chain_grabber"})
    info = {
        "losses": {
            "action_loss": torch.tensor(0.5),
            "3_loss_conditional_energy_score": torch.tensor(0.4),
            "7_loss_conditional_energy_score": torch.tensor(0.6),
            "3_log_native_action": torch.tensor(1.0),
            "7_log_native_action": torch.tensor(3.0),
        }
    }

    logged = PipelineAlgo.log_info(algo, info)

    assert logged["EnergyScore/u_socket"] == pytest.approx(0.4)
    assert logged["EnergyScore/chain_grabber"] == pytest.approx(0.6)
    assert logged["EnergyScore"] == pytest.approx(0.5)
    assert logged["MSE/u_socket"] == 1.0
    assert logged["MSE/chain_grabber"] == 3.0
    assert logged["MSE"] == 2.0


def test_training_log_info_aliases_two_node_latent_stability_diagnostics():
    algo = SimpleNamespace(domain_by_id={3: "u_socket", 7: "chain_grabber"})
    info = {
        "losses": {
            "action_loss": torch.tensor(0.5),
            "3_loss_latent_endpoint_gauge": torch.tensor(0.002),
            "7_loss_latent_endpoint_gauge": torch.tensor(0.004),
            "3_log_latent_endpoint_hinge_active_fraction": torch.tensor(0.25),
            "7_log_latent_endpoint_hinge_active_fraction": torch.tensor(0.5),
            "3_log_latent_endpoint_hinge_excess_m2": torch.tensor(2.0),
            "7_log_latent_endpoint_hinge_excess_m2": torch.tensor(4.0),
            "3_log_latent_endpoint_total_rms": torch.tensor(8.0),
            "7_log_latent_endpoint_total_rms": torch.tensor(10.0),
            "3_log_latent_endpoint_stabilized_rms": torch.tensor(7.5),
            "7_log_latent_endpoint_stabilized_rms": torch.tensor(8.0),
            "3_log_latent_endpoint_saturation_fraction": torch.tensor(0.25),
            "7_log_latent_endpoint_saturation_fraction": torch.tensor(0.5),
            "3_log_latent_endpoint_above_cap_fraction": torch.tensor(0.125),
            "7_log_latent_endpoint_above_cap_fraction": torch.tensor(0.25),
            "3_log_latent_endpoint_radial_scale_mean": torch.tensor(0.9),
            "7_log_latent_endpoint_radial_scale_mean": torch.tensor(0.8),
            "3_log_latent_endpoint_radial_scale_min": torch.tensor(0.7),
            "7_log_latent_endpoint_radial_scale_min": torch.tensor(0.6),
            "3_log_latent_endpoint_candidate_rms_max": torch.tensor(9.0),
            "7_log_latent_endpoint_candidate_rms_max": torch.tensor(11.0),
            "3_log_latent_endpoint_stabilized_candidate_rms_max": torch.tensor(7.7),
            "7_log_latent_endpoint_stabilized_candidate_rms_max": torch.tensor(7.9),
            "3_log_latent_endpoint_centered_within_k_rms": torch.tensor(2.0),
            "7_log_latent_endpoint_centered_within_k_rms": torch.tensor(4.0),
            "3_log_decoder_first_linear_weight_frobenius_norm": torch.tensor(1.5),
            "7_log_decoder_first_linear_weight_frobenius_norm": torch.tensor(2.5),
            "3_log_latent_decoder_scale_product": torch.tensor(12.0),
            "7_log_latent_decoder_scale_product": torch.tensor(25.0),
        }
    }

    logged = PipelineAlgo.log_info(algo, info)

    assert logged["LatentGauge/Loss/u_socket"] == pytest.approx(0.002)
    assert logged["LatentGauge/Loss/chain_grabber"] == pytest.approx(0.004)
    assert logged["LatentGauge/Loss"] == pytest.approx(0.003)
    assert logged["LatentGauge/Hinge_Active_Fraction"] == pytest.approx(0.375)
    assert logged["LatentGauge/Hinge_Excess_M2"] == pytest.approx(3.0)
    assert logged["LatentGauge/Endpoint_RMS/u_socket"] == pytest.approx(8.0)
    assert logged["LatentGauge/Endpoint_RMS/chain_grabber"] == pytest.approx(10.0)
    assert logged["LatentGauge/Endpoint_RMS"] == pytest.approx(9.0)
    assert logged["LatentGauge/Stabilized_RMS"] == pytest.approx(7.75)
    assert logged["LatentGauge/Saturation_Fraction"] == pytest.approx(0.375)
    assert logged["LatentGauge/Above_Cap_Fraction"] == pytest.approx(0.1875)
    assert logged["LatentGauge/Radial_Scale_Mean"] == pytest.approx(0.85)
    assert logged["LatentGauge/Radial_Scale_Min"] == pytest.approx(0.65)
    assert logged["LatentGauge/Candidate_RMS_Max"] == pytest.approx(10.0)
    assert logged["LatentGauge/Stabilized_Candidate_RMS_Max"] == pytest.approx(7.8)
    assert logged["LatentGauge/WithinK_RMS"] == pytest.approx(3.0)
    assert logged["LatentGauge/Decoder_FirstLinear_Frobenius"] == pytest.approx(2.0)
    assert logged["LatentGauge/Latent_Decoder_Scale_Product"] == pytest.approx(18.5)


def test_prediction_unnormalize_preserves_slotwise_arc_token_stats():
    emb_id = get_embodiment_id("eva_bimanual")
    action_key = "actions_cartesian"
    horizon, action_dim = 4, 3
    stats = {
        "quantile_1": np.zeros((horizon, action_dim), dtype=np.float32),
        "quantile_99": np.arange(1, horizon * action_dim + 1, dtype=np.float32).reshape(
            horizon, action_dim
        ),
    }
    norm_stats = MultiDataset.from_state(
        {
            "norm_mode": "quantile",
            "embodiments": [emb_id],
            "key_types": {emb_id: {action_key: "action_keys"}},
            "zarr_keys": {emb_id: {action_key: action_key}},
            "shapes": {emb_id: {action_key: (horizon, action_dim)}},
            "norm_stats": {emb_id: {action_key: stats}},
        }
    )
    evaluator = HumanRobotOverlayEval(viz_func=None)
    evaluator.model = SimpleNamespace(norm_stats=norm_stats)

    actual = evaluator._unnormalize_prediction(
        torch.zeros(2, horizon, action_dim), emb_id, action_key
    )

    expected = torch.from_numpy(stats["quantile_99"]) * 0.5 + 0.5e-6
    assert actual.shape == (2, horizon, action_dim)
    assert torch.allclose(actual[0], expected)
    assert torch.allclose(actual[1], expected)


def test_frame_limit_is_cumulative_across_validation_batches():
    emb_id = get_embodiment_id("eva_bimanual")
    action_key = "actions_cartesian"
    actions = torch.zeros(8, 4, 3)

    def viz(*, batch, **_kwargs):
        return np.zeros((batch[action_key].shape[0], 8, 8, 3), dtype=np.uint8)

    evaluator = _evaluator(
        emb_id,
        action_key,
        actions.clone(),
        frame_stride=1,
        max_frames=11,
        viz_func={"eva_bimanual": viz},
    )
    batch = {
        emb_id: {
            action_key: actions,
            "front_img_1": torch.zeros(8, 2, 3, 8, 8),
        }
    }

    _, first = evaluator.compute_metrics_and_viz(batch)
    _, second = evaluator.compute_metrics_and_viz(batch)
    _, third = evaluator.compute_metrics_and_viz(batch)

    assert first[emb_id].shape[0] == 8
    assert second[emb_id].shape[0] == 3
    assert third == {}


def test_canonical_126d_overlay_reverts_keypoints_to_head_frame():
    emb_id = get_embodiment_id("human_bimanual")
    action_key = "actions_keypoints"
    actions = torch.zeros(2, 3, 126)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    wrist_poses = torch.cat((identity, identity)).repeat(2, 1)
    seen = {}

    def viz(*, predictions, batch):
        seen["target"] = tuple(batch[action_key].shape)
        seen["prediction"] = tuple(predictions[f"human_bimanual_{action_key}"].shape)
        return np.zeros((2, 8, 8, 3), dtype=np.uint8)

    evaluator = _evaluator(
        emb_id,
        action_key,
        actions.clone(),
        frame_stride=1,
        max_frames=2,
        viz_func={"human_bimanual": viz},
        transform_lists={
            "human_bimanual": build_fold_keypoint_wristframe_revert_transform_list()
        },
    )

    metrics, images = evaluator.compute_metrics_and_viz(
        {
            emb_id: {
                action_key: actions,
                "viz_current_wrist_poses": wrist_poses,
                "front_img_1": torch.zeros(2, 2, 3, 8, 8),
            }
        }
    )

    assert seen == {"target": (2, 3, 126), "prediction": (2, 3, 126)}
    assert f"Valid/human_bimanual_{action_key}_camera_action_mse" in metrics
    assert images[emb_id].shape == (2, 8, 8, 3)


def test_deterministic_validation_noise_is_repeatable_and_rng_isolated(monkeypatch):
    seen = []

    def fake_step(_self, _batch, _batch_idx, _dataloader_idx=0):
        seen.append(torch.randn(4))

    monkeypatch.setattr(EvalVideo, "on_validation_step", fake_step)
    evaluator = HumanRobotOverlayEval(deterministic_seed=42)
    evaluator.trainer = SimpleNamespace(
        global_rank=0,
        lightning_module=SimpleNamespace(device=torch.device("cpu")),
    )

    torch.manual_seed(123)
    expected_next = torch.randn(4)
    torch.manual_seed(123)
    evaluator.on_validation_step({}, 7)
    actual_next = torch.randn(4)
    evaluator.on_validation_step({}, 7)

    assert torch.equal(seen[0], seen[1])
    assert torch.equal(actual_next, expected_next)


def test_exact_epoch_metrics_weight_every_action_element_once():
    emb_ids = [
        get_embodiment_id("eva_bimanual"),
        get_embodiment_id("human_bimanual"),
    ]
    logged = {}

    def log_dict(metrics, **kwargs):
        logged.update(metrics)

    evaluator = HumanRobotOverlayEval(exact_epoch_metrics=True)
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(device=torch.device("cpu"), log_dict=log_dict)
    )
    evaluator._accumulate_exact(
        emb_ids[0],
        torch.tensor([1.0, 3.0]),
        torch.tensor([10.0, 20.0]),
        torch.tensor([2.0, 4.0]),
    )
    evaluator._accumulate_exact(
        emb_ids[0], torch.tensor([5.0]), torch.tensor([30.0]), torch.tensor([6.0])
    )
    evaluator._accumulate_exact(
        emb_ids[1],
        torch.tensor([4.0, 4.0]),
        torch.tensor([40.0, 40.0]),
        torch.tensor([8.0, 8.0]),
    )

    evaluator.on_validation_end()

    assert logged["Valid/MSE/eva_bimanual"].item() == 3.0
    assert logged["Valid/MSE/human_bimanual"].item() == 4.0
    assert logged["Valid/MSE"].item() == 3.5
    assert logged["Valid/ActionToken_MSE/eva_bimanual"].item() == 20.0
    assert logged["Valid/ActionToken_MSE/human_bimanual"].item() == 40.0
    assert logged["Valid/ActionToken_MSE"].item() == 30.0
    assert logged["Valid/Native_MSE/eva_bimanual"].item() == 4.0
    assert logged["Valid/Native_MSE/human_bimanual"].item() == 8.0
