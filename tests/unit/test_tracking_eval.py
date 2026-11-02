"""Tracking-invalid validation windows cannot reach loss, metrics, or prediction."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval import eval_hpt
from egomimic.eval.eval_hpt import HPTEvalVideo, filter_tracking_batch
from egomimic.eval.moving_opscale_eval import (
    COHORT_HEADS,
    MovingOpscaleEvaluation,
    _NamedReferenceRenderer,
)
from egomimic.eval.opscale_reference_open_loop import OpenLoopSimEval
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

EMB = get_embodiment_id("human_bimanual")


def batch(values, flags):
    samples = {
        "actions_keypoints": torch.tensor(values, dtype=torch.float32)[:, None, None]
        .expand(-1, 2, 3)
        .clone(),
        "sampled_prompt": [f"prompt-{i}" for i in range(len(values))],
        "episode_hash": [f"episode-{i}" for i in range(len(values))],
        "frame_idx": torch.arange(len(values)),
        "embodiment": torch.tensor([EMB]),
    }
    if flags is not None:
        samples["evaluation_tracking_valid"] = torch.tensor(flags)
    return {EMB: samples}


class FakeAlgo:
    ac_keys = {EMB: "actions_keypoints"}
    shared_ac_key = None
    auxiliary_ac_keys = {}
    norm_stats = SimpleNamespace(unnormalize=lambda samples, embodiment: samples)

    def __init__(self):
        self.forward_batches = []

    def forward_eval(self, samples):
        self.forward_batches.append(samples)
        target = samples[EMB]["actions_keypoints"]
        assert target.abs().max() < 1e8
        return {
            "human_bimanual_loss": target[:, 0, 0].mean(),
            "human_bimanual_actions_keypoints": target + 1,
        }


@pytest.fixture
def evaluator(monkeypatch):
    instance = HPTEvalVideo(viz_every_n_epochs=0)
    instance.model = FakeAlgo()
    monkeypatch.setattr(
        eval_hpt,
        "frechet_gaussian_over_time",
        lambda pred, gt: torch.ones(pred.shape[0]),
    )
    monkeypatch.setattr(
        eval_hpt,
        "layout_metrics",
        lambda pred, gt, prefix, mask=None: {
            prefix + "_layout": (pred - gt).abs().mean()
        },
    )
    return instance


def test_mixed_batch_slices_every_sample_field_before_forward_without_mutation(
    evaluator,
):
    original = batch([2, 1e9, 4], [True, False, True])
    metrics, images = evaluator.compute_metrics_and_viz(original, do_viz=False)
    received = evaluator.model.forward_batches[0][EMB]
    assert received["sampled_prompt"] == ["prompt-0", "prompt-2"]
    assert received["frame_idx"].tolist() == [0, 2]
    assert received["actions_keypoints"].shape[0] == 2
    assert original[EMB]["actions_keypoints"].shape[0] == 3
    assert original[EMB]["actions_keypoints"][1, 0, 0] == 1e9
    assert metrics["Valid/action_loss"] == 3
    assert images == {}
    assert set(evaluator.metric_batch_sizes) == set(metrics)
    assert set(evaluator.metric_batch_sizes.values()) == {2}


def test_all_invalid_batch_never_calls_model_or_fabricates_scalars(evaluator):
    metrics, images = evaluator.compute_metrics_and_viz(
        batch([1e9, 1e9], [False, False]), do_viz=False
    )
    assert metrics == images == {}
    assert evaluator.model.forward_batches == []
    assert evaluator.metric_batch_sizes == {}


def test_flag_absent_keeps_legacy_evaluation(evaluator):
    original = batch([2, 4], None)
    filtered, counts, active = filter_tracking_batch(original)
    assert filtered is original and counts == {} and active is False
    evaluator.compute_metrics_and_viz(original, do_viz=False)
    assert evaluator.metric_batch_sizes is None


@pytest.mark.parametrize(
    "flags",
    [
        torch.tensor([float("nan")]),
        torch.tensor([2]),
        torch.tensor([[True]]),
        torch.tensor([True, False]),
    ],
)
def test_malformed_tracking_flag_is_rejected_before_forward(evaluator, flags):
    samples = batch([2], [True])
    samples[EMB]["evaluation_tracking_valid"] = flags
    with pytest.raises(ValueError):
        evaluator.compute_metrics_and_viz(samples, do_viz=False)
    assert evaluator.model.forward_batches == []


def test_epoch_logging_weights_each_batch_by_surviving_samples_and_skips_empty(
    evaluator,
):
    logged = []
    module = SimpleNamespace(
        device=torch.device("cpu"),
        log_dict=lambda metrics, **kwargs: logged.append((metrics, kwargs)),
    )
    evaluator.trainer = SimpleNamespace(lightning_module=module)
    evaluator.on_validation_step(
        batch([2, 1e9, 1e9, 1e9], [True, False, False, False]), 0, mode="metrics"
    )
    evaluator.on_validation_step(batch([1e9], [False]), 1, mode="metrics")
    evaluator.on_validation_step(batch([4, 4, 4, 4], [True] * 4), 2, mode="metrics")
    assert [kwargs["batch_size"] for _, kwargs in logged] == [1, 4]
    weighted = (
        sum(
            float(metrics["Valid/action_loss"]) * kwargs["batch_size"]
            for metrics, kwargs in logged
        )
        / 5
    )
    assert weighted == pytest.approx(3.6)
    assert len(evaluator.model.forward_batches) == 2


def test_horizon_metrics_weight_only_rows_with_real_steps_in_that_region(
    evaluator, monkeypatch
):
    samples = batch([2, 4], [True, True])
    samples[EMB]["actions_keypoints"] = torch.ones(2, 6, 3)
    samples[EMB]["action_pad_mask"] = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]]
    )
    monkeypatch.setattr(
        eval_hpt,
        "layout_metrics",
        lambda pred, gt, prefix, mask=None: {
            prefix + "_wrist_xyz_paired_mse_late_third": torch.tensor(1.0),
            prefix + "_kp_l2_late": torch.tensor(1.0),
        },
    )
    evaluator.compute_metrics_and_viz(samples, do_viz=False)
    weights = evaluator.metric_batch_sizes
    assert weights["Valid/action_loss"] == 2
    assert (
        weights[
            "Valid/human_bimanual_actions_keypoints_wrist_xyz_paired_mse_late_third"
        ]
        == 1
    )
    assert weights["Valid/human_bimanual_actions_keypoints_kp_l2_late"] == 1


def callback_fixture(tmp_path):
    callback = MovingOpscaleEvaluation(str(tmp_path), "synthetic")
    callback.split = {cohort: [cohort + "-a", cohort + "-b"] for cohort in COHORT_HEADS}
    trainer = SimpleNamespace(
        world_size=1,
        global_step=2,
        sanity_checking=False,
        current_epoch=0,
        max_epochs=2,
        default_root_dir=str(tmp_path),
        callback_metrics={
            f"val/{cohort}/Valid/action_loss": torch.tensor(1.0)
            for cohort in COHORT_HEADS
        },
    )
    module = SimpleNamespace(val_loader_names=list(COHORT_HEADS.values()))
    callback.on_validation_start(trainer, module)
    return callback, trainer, module


def test_visited_and_scored_episode_coverage_are_separate_and_required(tmp_path):
    callback, trainer, module = callback_fixture(tmp_path)
    for index, cohort in enumerate(COHORT_HEADS):
        a, b = callback.split[cohort]
        callback.on_validation_batch_end(
            trainer,
            module,
            None,
            {
                "human_bimanual": {
                    "episode_hash": [a, b, a],
                    "evaluation_tracking_valid": torch.tensor([True, False, True]),
                }
            },
            0,
            index,
        )
    with pytest.raises(RuntimeError, match="tracking-valid scored coverage"):
        callback.on_validation_epoch_end(trainer, module)
    assert not (tmp_path / "evaluation_status.json").exists()
    for index, cohort in enumerate(COHORT_HEADS):
        b = callback.split[cohort][1]
        callback.on_validation_batch_end(
            trainer,
            module,
            None,
            {
                "human_bimanual": {
                    "episode_hash": [b],
                    "evaluation_tracking_valid": torch.tensor([True]),
                }
            },
            1,
            index,
        )
    callback.on_validation_epoch_end(trainer, module)
    status = json.loads((tmp_path / "evaluation_status.json").read_text())
    assert status["evaluation_policy"] == "tracking_valid_windows_v1"
    for cohort in COHORT_HEADS:
        totals = status["tracking_coverage"][cohort]
        assert {key: totals[key] for key in ["visited", "scored", "skipped"]} == {
            "visited": 4,
            "scored": 3,
            "skipped": 1,
        }
        assert totals["episodes"][cohort + "-a"] == {
            "visited": 2,
            "scored": 2,
            "skipped": 0,
        }
        assert totals["episodes"][cohort + "-b"] == {
            "visited": 2,
            "scored": 1,
            "skipped": 1,
        }
        assert status["coverage"][cohort] == {"expected": 2, "observed": 2}


def test_moving_callback_cannot_assume_missing_tracking_metadata_is_valid(tmp_path):
    callback, trainer, module = callback_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires evaluation_tracking_valid"):
        callback.on_validation_batch_end(
            trainer,
            module,
            None,
            {"human_bimanual": {"episode_hash": ["seen_matched-a"]}},
            0,
            0,
        )


class CameraLeaf:
    total_frames = 6

    def __getitem__(self, frame):
        return {
            "frame_idx": frame,
            "observations.images.front_img_1": np.full(
                (48, 160, 3), frame, dtype=np.uint8
            ),
        }


def renderer_fixture(monkeypatch, flags):
    renderer = _NamedReferenceRenderer(
        leaves={"episode": CameraLeaf()},
        cohort="seen_matched",
        operator="operator",
        episode="episode",
        execute_steps=2,
        seen_episodes=["episode"],
        log_metrics=False,
    )
    renderer.trainer = SimpleNamespace(
        is_global_zero=False, current_epoch=0, max_epochs=1
    )
    renderer.on_validation_start()
    observed = []

    def valid_render(instance, filtered, do_viz=True):
        anchors = filtered[EMB]["frame_idx"].tolist()
        observed.extend(anchors)
        sequences = [
            np.stack(
                [
                    np.full((48, 160, 3), frame, np.uint8)
                    for frame in range(anchor, anchor + 2)
                ]
            )
            for anchor in anchors
        ]
        instance._buffer_sequences(["episode"] * len(anchors), sequences)
        return {}, {}

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid camera-only spans must never revert sentinel poses")

    monkeypatch.setattr(OpenLoopSimEval, "compute_metrics_and_viz", valid_render)
    monkeypatch.setattr(renderer, "_reverted_keypoints", forbidden)
    monkeypatch.setattr(OpenLoopSimEval, "on_validation_end", lambda instance: None)
    samples = batch([0 if flag else 1e9 for flag in flags], flags)
    samples[EMB]["frame_idx"] = torch.tensor([0, 2, 4])
    return renderer, samples, observed


def test_renderer_preserves_frame_order_and_duration_with_labeled_invalid_spans(
    monkeypatch,
):
    renderer, samples, observed = renderer_fixture(monkeypatch, [True, False, True])
    renderer.compute_metrics_and_viz(samples)
    renderer.on_validation_end()
    assert observed == [0, 4]
    frames = renderer._frames["episode"]
    assert len(frames) == 6
    assert [int(frame[-1, 0, 0]) for frame in frames] == list(range(6))
    assert not torch.all(
        frames[2][:25] == 2
    )  # The unavailable span is visibly labeled.
    evidence = renderer._video_tracking
    assert evidence["tracking"] == {"visited": 3, "scored": 2, "skipped": 1}
    assert evidence["source_frames"] == 6 and evidence["unscored_frames"] == 2
    assert evidence["unscored_spans"] == [{"start_frame": 2, "frame_count": 2}]


def test_pinned_video_with_no_valid_prediction_cannot_pass(monkeypatch):
    renderer, samples, observed = renderer_fixture(monkeypatch, [False] * 3)
    renderer.compute_metrics_and_viz(samples)
    assert observed == []
    with pytest.raises(RuntimeError, match="no tracking-valid prediction"):
        renderer.on_validation_end()


def test_video_current_gt_stays_correct_despite_later_missing_future(tmp_path):
    from egomimic.rldb.embodiment.human import (
        Human,
        _build_human_keypoints_revert_6d_wristframe_transform_list,
    )
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

    def leaf(name, missing):
        frames = 45
        numeric = {
            "obs_head_pose": np.tile([0, 0, 0, 1, 0, 0, 0], (frames, 1)),
            "images.front_1": np.zeros((frames, 3, 48, 64), np.float32),
        }
        for hand in ("left", "right"):
            poses = np.tile([0.1, 0.2, 0.3, 1, 0, 0, 0], (frames, 1))
            points = np.tile(np.linspace(0.1, 0.3, 63), (frames, 1))
            poses[:, 0] += np.arange(frames) / 100
            points += np.arange(frames)[:, None] / 100
            if missing:
                poses[35] = 1e9
                points[35] = 0
            numeric[f"{hand}.obs_wrist_pose"] = poses
            numeric[f"{hand}.obs_keypoints"] = points
        path = tmp_path / (name + ".zarr")
        ZarrWriter.create_and_write(
            path,
            numeric_data=numeric,
            image_data={},
            embodiment="human_bimanual",
            fps=30,
            task_name="synthetic",
            intrinsics={"front_1": np.eye(3, 4)},
        )
        return ZarrDataset(
            path,
            key_map=Human.get_keymap("keypoints", proprio_history=4),
            transform_list=Human.get_transform_list(
                "keypoints_wristframe_6d", stride=1
            ),
        )

    missing = leaf("missing", True)
    tracked = leaf("tracked", False)
    renderer = _NamedReferenceRenderer(
        leaves={"missing": missing},
        cohort="seen_matched",
        operator="operator",
        episode="missing",
        execute_steps=30,
        seen_episodes=["missing"],
        log_metrics=False,
    )
    renderer._pose_cache = {}
    transforms = _build_human_keypoints_revert_6d_wristframe_transform_list()
    # The scored anchor is fully tracked; a displayed current frame's normal
    # future read reaches a missing pose outside that anchor's source window.
    assert missing.evaluation_tracking_validity(0)
    assert not missing.evaluation_tracking_validity(10)
    original_windows = missing._read_windows(10)
    legacy_bad = OpenLoopSimEval._frame_pose(
        renderer, missing, "legacy-bad", 10, Human.VIZ_IMAGE_KEY, transforms
    )
    legacy_good = OpenLoopSimEval._frame_pose(
        renderer, tracked, "legacy-good", 10, Human.VIZ_IMAGE_KEY, transforms
    )
    actual = renderer._frame_pose(
        missing, "missing", 10, Human.VIZ_IMAGE_KEY, transforms
    )
    # The later sentinel really contaminates action wrist channels. The
    # reference revert discards those slices and projects the first keypoint
    # row using current proprio, so the displayed tracked skeleton stays exact.
    assert missing[10]["actions_keypoints"].abs().max() >= 1e8
    np.testing.assert_allclose(legacy_bad[1], legacy_good[1], atol=1e-6)
    np.testing.assert_allclose(actual[1], legacy_good[1], atol=1e-6)
    np.testing.assert_array_equal(actual[0], legacy_good[0])
    assert not missing.evaluation_tracking_validity(10)
    for key, (raw, window) in original_windows.items():
        after_raw, after_window = missing._read_windows(10)[key]
        assert raw == after_raw
        np.testing.assert_array_equal(window, after_window)
