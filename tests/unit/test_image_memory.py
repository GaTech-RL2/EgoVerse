"""Long-range image memory in the data path: ``Human.get_keymap(image_memory=N,
image_memory_stride_s=s)`` adds a ``<front key>_mem`` window of N frames s
seconds apart ending at the current frame, with a ``_mem_mask`` of the steps inside
the episode; ``history_stride_s`` spaces proprio history in seconds too."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    ZarrEpisode,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

T = 150
CAM = Human.VIZ_IMAGE_KEY
MEM = f"{CAM}_mem"
MASK = f"{MEM}_mask"


def _leaf(tmp_path, fps: int = 30, **keymap_kwargs):
    rng = np.random.default_rng(0)
    t = np.arange(T, dtype=np.float64)[:, None]
    pose = np.concatenate(
        [t * 0.001 + np.zeros((T, 3)), np.tile([[1.0, 0.0, 0.0, 0.0]], (T, 1))], axis=1
    )
    ZarrWriter.create_and_write(
        tmp_path / "ep.zarr",
        numeric_data={
            "left.obs_ee_pose": pose,
            "right.obs_ee_pose": pose,
            "obs_head_pose": pose,
        },
        image_data={"images.front_1": rng.integers(0, 255, (T, 32, 32, 3), np.uint8)},
        embodiment="human_bimanual",
        fps=fps,
        task_name="synthetic",
        intrinsics={"front_1": np.eye(3, 4)},
    )
    key_map = Human.get_keymap(keymap_mode="cartesian", **keymap_kwargs)
    resolver = LocalEpisodeResolver(tmp_path, key_map=key_map, transform_list=[])
    ds = MultiDataset._from_resolver(resolver, mode="total")
    return next(iter(ds.datasets.values()))


@pytest.mark.parametrize("fps", [10, 30])
def test_memory_frames_are_stride_seconds_apart_ending_at_the_current_frame(
    tmp_path, fps
):
    leaf = _leaf(tmp_path, fps, image_memory=4, image_memory_stride_s=0.5)
    step = round(0.5 * fps)
    idx = T - 1
    sample = leaf[idx]
    assert sample[MEM].shape == (4, 3, 32, 32)
    for j, frame in enumerate(sample[MEM]):
        assert torch.equal(frame, leaf[idx - step * (3 - j)][CAM])
    np.testing.assert_array_equal(np.asarray(sample[MASK]), np.ones(4))


def test_memory_front_pads_with_the_oldest_real_step_and_masks_the_rest(tmp_path):
    leaf = _leaf(tmp_path, image_memory=4, image_memory_stride_s=1.0)
    sample = leaf[35]  # steps 5, 35 are real; -25, -55 are not
    np.testing.assert_array_equal(np.asarray(sample[MASK]), [0, 0, 1, 1])
    for j, src in enumerate((5, 5, 5, 35)):
        assert torch.equal(sample[MEM][j], leaf[src][CAM])


def test_memory_at_the_first_frame_holds_only_the_current_step(tmp_path):
    leaf = _leaf(tmp_path, image_memory=3, image_memory_stride_s=1.0)
    sample = leaf[0]
    np.testing.assert_array_equal(np.asarray(sample[MASK]), [0, 0, 1])
    assert sample[MEM].shape == (3, 3, 32, 32)


def test_a_one_frame_memory_keeps_its_window_axis(tmp_path):
    sample = _leaf(tmp_path, image_memory=1)[40]
    assert sample[MEM].shape == (1, 3, 32, 32)
    assert sample[MASK].shape == (1,)


def test_memory_reads_and_decodes_each_distinct_frame_once(tmp_path, monkeypatch):
    leaf = _leaf(
        tmp_path, image_memory=10, image_memory_stride_s=1.0, image_history_gap_s=0.1
    )
    rows: list[dict] = []
    inner = ZarrEpisode.read_intervals

    def spy(self, intervals):
        out = inner(self, intervals)
        rows.append({k: sum(len(r) for _, r in g) for k, g in out.items()})
        return out

    monkeypatch.setattr(ZarrEpisode, "read_intervals", spy)
    leaf[T - 1]
    # current (also the newest memory step) + 0.1 s pair frame + 119 - 30k
    assert rows[-1]["images.front_1"] == 1 + 1 + 4


def test_default_keymap_has_no_memory(tmp_path):
    sample = _leaf(tmp_path)[3]
    assert MEM not in sample and MASK not in sample


def test_norm_mode_keymap_drops_the_memory_camera():
    key_map = Human.get_keymap("cartesian", norm_mode=True, image_memory=10)
    assert not [k for k in key_map if "img" in k]


@pytest.mark.parametrize("fps", [10, 30])
def test_proprio_history_stride_in_seconds_follows_the_episode_fps(tmp_path, fps):
    leaf = _leaf(tmp_path, fps, proprio_history=3, history_stride_s=0.5)
    step = round(0.5 * fps)
    idx = T - 1
    got = np.asarray(leaf[idx]["right.obs_ee_pose"])
    want = np.stack(
        [np.asarray(leaf[idx - step * j]["right.obs_ee_pose"])[-1] for j in (2, 1, 0)]
    )
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(
        np.asarray(leaf[step]["proprio_history_mask"]), [0, 1, 1]
    )
