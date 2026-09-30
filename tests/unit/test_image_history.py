"""Image history in the data path: ``Human.get_keymap(image_history_gap_s=...)``
adds a ``<front key>_hist`` camera entry that ``ZarrDataset`` reads that many
seconds back (in the episode's own fps), and every sample carries ``fps``."""

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

T = 90
CAM = Human.VIZ_IMAGE_KEY
PAST = f"{CAM}_hist"


def _leaf(tmp_path, fps: int, **keymap_kwargs):
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


@pytest.mark.parametrize("fps,gap_frames", [(30, 3), (10, 1), (60, 6)])
def test_past_frame_is_gap_seconds_back_in_the_episodes_fps(tmp_path, fps, gap_frames):
    leaf = _leaf(tmp_path, fps, image_history_gap_s=0.1)
    idx = 8
    assert torch.equal(leaf[idx][PAST], leaf[idx - gap_frames][CAM])
    assert not torch.equal(leaf[idx][PAST], leaf[idx][CAM])
    assert leaf[idx]["fps"].item() == fps


def test_past_frame_clamps_to_the_episode_start(tmp_path):
    leaf = _leaf(tmp_path, 30, image_history_gap_s=0.1)
    for idx in (0, 1, 2):
        assert torch.equal(leaf[idx][PAST], leaf[0][CAM])


def test_a_gap_below_one_frame_still_reads_the_previous_frame(tmp_path):
    leaf = _leaf(tmp_path, 30, image_history_gap_s=0.001)
    assert torch.equal(leaf[5][PAST], leaf[4][CAM])


def test_default_keymap_has_no_history_entry_but_samples_carry_fps(tmp_path):
    leaf = _leaf(tmp_path, 30)
    sample = leaf[3]
    assert PAST not in sample and sample[CAM].shape == (3, 32, 32)
    assert sample["fps"].dtype == torch.float32
    assert "proprio_history_mask" not in sample  # a camera lag is not proprio history


def test_norm_mode_keymap_drops_the_history_camera_too():
    key_map = Human.get_keymap("cartesian", norm_mode=True, image_history_gap_s=0.1)
    assert not [k for k in key_map if "img" in k]


@pytest.mark.parametrize("gap_s", [0.1, 0.5, 2.0])
def test_the_past_frame_costs_one_jpeg_row_whatever_the_gap(
    tmp_path, monkeypatch, gap_s
):
    """The past frame shares the front camera's zarr key, so the read merges the
    two windows. JPEGs are one frame per chunk: a plain union span would read
    every frame between them (61 rows at 2 s and 30 fps) to use two."""
    leaf = _leaf(tmp_path, 30, image_history_gap_s=gap_s)
    rows: list[dict] = []
    inner = ZarrEpisode.read_intervals

    def spy(self, intervals):
        out = inner(self, intervals)
        rows.append({k: sum(len(r) for _, r in g) for k, g in out.items()})
        return out

    monkeypatch.setattr(ZarrEpisode, "read_intervals", spy)
    idx = 80
    sample = leaf[idx]
    assert rows[-1]["images.front_1"] == 2
    assert torch.equal(sample[PAST], leaf[idx - round(gap_s * 30)][CAM])
