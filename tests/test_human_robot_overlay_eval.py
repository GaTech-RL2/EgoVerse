from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.human_robot_overlay_eval import HumanRobotOverlayEval
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import (
    build_fold_keypoint_wristframe_revert_transform_list,
)


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


def test_validation_metric_gate_prints_finite_values_and_rejects_nan():
    summary = HumanRobotOverlayEval.validate_metric_values(
        {"Valid/action_mse": torch.tensor(1.25)}
    )
    assert summary == "Valid/action_mse=1.25"

    with pytest.raises(FloatingPointError, match="Valid/action_mse"):
        HumanRobotOverlayEval.validate_metric_values(
            {"Valid/action_mse": torch.tensor(float("nan"))}
        )


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


def test_viz_gt_preds_uses_the_configured_per_sample_intrinsics(monkeypatch):
    emb_id = get_embodiment_id("eva_bimanual")
    action_key = "actions_cartesian"
    actions = torch.zeros(2, 3, 14)
    intrinsics = torch.tensor(
        [
            [[100.0, 0.0, 10.0, 0.0], [0.0, 101.0, 11.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[200.0, 0.0, 20.0, 0.0], [0.0, 201.0, 21.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ]
    )
    seen = []

    def capture_viz(cls, image, viz_data, *, intrinsics=None, **_kwargs):
        seen.append(np.asarray(intrinsics))
        return np.asarray(image)

    monkeypatch.setattr(Eva, "viz", classmethod(capture_viz))
    images = Eva.viz_gt_preds(
        predictions={"eva_bimanual_actions_cartesian": actions},
        batch={
            "embodiment": torch.full((2,), emb_id),
            "front_img_1": torch.zeros(2, 8, 8, 3),
            action_key: actions,
            "front_intrinsics": intrinsics,
        },
        image_key="front_img_1",
        action_key=action_key,
        intrinsics_batch_key="front_intrinsics",
    )

    assert images.shape == (2, 8, 8, 3)
    assert len(seen) == 4
    np.testing.assert_allclose(seen[0], intrinsics[0].numpy())
    np.testing.assert_allclose(seen[1], intrinsics[0].numpy())
    np.testing.assert_allclose(seen[2], intrinsics[1].numpy())
    np.testing.assert_allclose(seen[3], intrinsics[1].numpy())


def test_viz_gt_preds_rejects_missing_explicit_intrinsics():
    emb_id = get_embodiment_id("eva_bimanual")
    actions = torch.zeros(1, 3, 14)

    with pytest.raises(ValueError, match="front_intrinsics"):
        Eva.viz_gt_preds(
            predictions={"eva_bimanual_actions_cartesian": actions},
            batch={
                "embodiment": torch.tensor([emb_id]),
                "front_img_1": torch.zeros(1, 8, 8, 3),
                "actions_cartesian": actions,
            },
            image_key="front_img_1",
            action_key="actions_cartesian",
            intrinsics_batch_key="front_intrinsics",
        )
