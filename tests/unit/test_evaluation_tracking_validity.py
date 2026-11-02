"""Raw missing-hand metadata is evaluation-only and preserves exact starts."""

from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.rldb.zarr.zarr_dataset_multi import (
    EvenStrideDataset,
    MultiDataset,
    ZarrDataset,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


def _leaf(
    tmp_path,
    *,
    missing=(),
    side="left",
    history=3,
    stride=1,
    horizon=3,
    malformed=None,
    transform_list=None,
):
    frames = 12
    numeric = {}
    key_map = {}
    for hand in ("left", "right"):
        poses = np.tile([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], (frames, 1))
        keypoints = np.full((frames, 63), 0.01)
        if hand == side:
            for frame in missing:
                poses[frame] = 1e9
                keypoints[frame] = 0
            if malformed is not None:
                poses[4], keypoints[4] = malformed
        numeric[f"{hand}.obs_wrist_pose"] = poses
        numeric[f"{hand}.obs_keypoints"] = keypoints
        for suffix in ("wrist_pose", "keypoints"):
            raw_key = f"{hand}.obs_{suffix}"
            key_map[f"{hand}.future_{suffix}"] = {
                "zarr_key": raw_key,
                "key_type": "action_keys",
                "horizon": horizon,
            }
            key_map[f"{hand}.history_{suffix}"] = {
                "zarr_key": raw_key,
                "key_type": "proprio_keys",
                "history": history,
                "history_stride": stride,
            }
    path = tmp_path / "tracking.zarr"
    ZarrWriter.create_and_write(
        path,
        numeric_data=numeric,
        image_data={},
        embodiment="human_bimanual",
        fps=30,
        task_name="synthetic",
        intrinsics={"front_1": np.eye(3, 4)},
    )
    return ZarrDataset(path, key_map=key_map, transform_list=transform_list or [])


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("selected", [2, 4, 6])
def test_future_current_and_real_history_each_make_exact_start_unscorable(
    tmp_path, side, selected
):
    # Missing frame4 is respectively future support, current anchor, or history.
    leaf = _leaf(tmp_path, missing=[4], side=side)
    base = MultiDataset(datasets={"tracking": leaf}, mode="total")
    exact = EvenStrideDataset(
        base, stride=1, preserve_indices=True, replace_invalid=True
    )
    sample = exact[selected]
    assert sample["evaluation_tracking_valid"].dtype == torch.bool
    assert sample["evaluation_tracking_valid"].shape == ()
    assert sample["evaluation_tracking_valid"].item() is False
    assert int(sample["frame_idx"]) == selected
    assert sample["episode_hash"] == leaf[selected]["episode_hash"]
    assert leaf.evaluation_tracking_validity(9) is True


def test_tracking_classification_does_not_change_training_or_norm_samples(tmp_path):
    leaf = _leaf(tmp_path, missing=[4])
    base = MultiDataset(datasets={"tracking": leaf}, mode="total")
    before = leaf[4]
    assert "evaluation_tracking_valid" not in before
    exact = EvenStrideDataset(base, stride=1, preserve_indices=True)
    evaluated = exact[4]
    after = leaf[4]
    assert "evaluation_tracking_valid" not in after
    assert "evaluation_tracking_valid" not in base[4]
    assert "evaluation_tracking_valid" not in EvenStrideDataset(base, stride=1)[4]
    assert (
        before.keys()
        == after.keys()
        == evaluated.keys() - {"evaluation_tracking_valid"}
    )
    for key, expected in before.items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(
                after[key], expected, rtol=0, atol=0, equal_nan=True
            )
            torch.testing.assert_close(
                evaluated[key], expected, rtol=0, atol=0, equal_nan=True
            )
        else:
            assert after[key] == evaluated[key] == expected
    # The validity marker did not clamp or erase the sentinel numerical values.
    assert torch.all(after["left.future_wrist_pose"][0] == 1e9)


def test_unrequested_history_gap_is_not_classified_as_a_required_read(
    tmp_path, monkeypatch
):
    leaf = _leaf(tmp_path, missing=[4], history=3, stride=3, horizon=2)
    calls = []
    original = type(leaf.episode_reader).read_intervals

    def read(reader, intervals):
        calls.append(intervals)
        return original(reader, intervals)

    monkeypatch.setattr(type(leaf.episode_reader), "read_intervals", read)
    assert leaf.evaluation_tracking_validity(6) is True
    assert len(calls) == 1
    assert set(calls[0]) == {
        "left.obs_wrist_pose",
        "left.obs_keypoints",
        "right.obs_wrist_pose",
        "right.obs_keypoints",
    }
    for windows in calls[0].values():
        assert {start for start, _ in windows} == {0, 3, 6, 7}


def test_null_padded_history_does_not_add_an_unrequired_missing_start(tmp_path):
    leaf = _leaf(tmp_path, missing=[0], history=4, stride=3, horizon=2)
    # All negative history slots repeat frame1 and are nulled, not frame0.
    assert leaf.evaluation_tracking_validity(1) is True
    assert leaf.evaluation_tracking_validity(3) is False  # frame0 is now real history


def test_interior_raw_interpolation_support_is_checked(tmp_path):
    leaf = _leaf(tmp_path, missing=[5], history=1, horizon=7)
    assert (
        leaf.evaluation_tracking_validity(2) is False
    )  # support [2,9), endpoints valid
    assert leaf.evaluation_tracking_validity(6) is True


@pytest.mark.parametrize(
    "poses,keypoints,error",
    [
        (np.array([1e9, 0, 0, 1, 0, 0, 0]), np.zeros(63), "Malformed large"),
        (np.full(7, -1e9), np.zeros(63), "Malformed large"),
        (np.full(7, 1e9), np.ones(63), "nonzero keypoints"),
        (np.full(7, np.nan), np.zeros(63), "Non-finite raw"),
    ],
)
def test_malformed_raw_tracking_is_rejected_not_silently_masked(
    tmp_path, poses, keypoints, error
):
    leaf = _leaf(tmp_path, malformed=(poses, keypoints))
    with pytest.raises(ValueError, match=error):
        leaf.evaluation_tracking_validity(4)


def test_zero_keypoints_without_legacy_sentinel_keep_existing_semantics(tmp_path):
    leaf = _leaf(tmp_path, malformed=(np.array([0, 0, 0, 1, 0, 0, 0]), np.zeros(63)))
    assert leaf.evaluation_tracking_validity(4) is True


def test_raw_validity_precedes_transforms_and_does_not_mutate_their_output(tmp_path):
    class HideSentinel:
        def transform(self, data):
            return {key: np.zeros_like(value) for key, value in data.items()}

    leaf = _leaf(tmp_path, missing=[4], transform_list=[HideSentinel()])
    base = MultiDataset(datasets={"tracking": leaf}, mode="total")
    sample = EvenStrideDataset(base, stride=1, preserve_indices=True)[4]
    assert sample["evaluation_tracking_valid"].item() is False
    assert torch.all(sample["left.future_wrist_pose"] == 0)


def test_legacy_stub_leaf_stays_compatible_and_retains_frame_identity():
    class Stub:
        sample_indices = None
        episode_path = Path("synthetic.zarr")

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {
                "frame_idx": torch.tensor(index),
                "value": torch.tensor(float(index)),
            }

    base = MultiDataset(datasets={"synthetic": Stub()}, mode="total")
    sample = EvenStrideDataset(base, stride=1, preserve_indices=True)[1]
    assert sample["evaluation_tracking_valid"].item() is True
    assert sample["frame_idx"].item() == 1
    assert sample["value"].item() == 1
