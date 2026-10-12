"""Exact-index validation reads: the metric loaders score the frames
``EvenStrideDataset`` selected, not whatever ``MultiDataset``'s bounds-check
retry swapped in."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import egomimic.trainHydra as th
from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import (
    EvenStrideDataset,
    LocalEpisodeResolver,
    MultiDataset,
    pinned_episode_subset,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

KEY = "actions_cartesian"
OUTLIER, NAN, SWAPPED = 4, 6, 8


class _Leaf:
    """n frames; frame i carries i in channel 1. Frame OUTLIER is far outside
    the stats range, NAN is non-finite, SWAPPED reports another frame_idx (a
    JPEG retry inside the leaf)."""

    def __init__(self, n=12, sample_indices=None):
        self.n = n
        self.sample_indices = sample_indices

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        arr = np.zeros((3, 18), np.float32)
        arr[:, 1] = i / 100
        if i == OUTLIER:
            arr[1, 0] = 50.0
        if i == NAN:
            arr[1, 0] = np.nan
        return {
            "embodiment": 0,
            KEY: arr,
            "frame_idx": torch.tensor(i + 1 if i == SWAPPED else i),
        }


def _split(**leaves) -> MultiDataset:
    split = MultiDataset(datasets=leaves, mode="total")
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.norm_stats = {
        0: {
            KEY: {
                "quantile_1": np.full(18, -1.0, np.float32),
                "quantile_99": np.full(18, 1.0, np.float32),
            }
        }
    }
    stats.key_types = {0: {KEY: "action_keys"}}
    stats.zarr_keys = {0: {KEY: KEY}}
    split.set_norm_stats_from(stats)
    return split


def _frame(sample) -> int:
    return int(sample["frame_idx"])


def test_exact_read_keeps_an_out_of_quantile_frame():
    split = _split(ep=_Leaf())
    loose = EvenStrideDataset(split, stride=1)
    exact = EvenStrideDataset(split, stride=1, preserve_indices=True)
    assert _frame(exact[OUTLIER]) == OUTLIER
    assert _frame(loose[OUTLIER]) != OUTLIER, "the old path swaps it out"
    # normalized like the base path: quantile maps [-1, 1] to [-1, 1]
    torch.testing.assert_close(exact[3][KEY], split[3][KEY])


@pytest.mark.parametrize("bad", [NAN, SWAPPED])
def test_invalid_frame_fails_without_replacement(bad):
    exact = EvenStrideDataset(_split(ep=_Leaf()), stride=1, preserve_indices=True)
    with pytest.raises(ValueError, match="non-finite|replaced"):
        exact[bad]


@pytest.mark.parametrize("bad", [NAN, SWAPPED])
def test_replacement_is_seeded_logged_and_inside_the_mask(bad, caplog):
    kept = np.array([i for i in range(12) if i not in (0, 1, 2)])
    split = _split(ep=_Leaf(sample_indices=kept))
    exact = EvenStrideDataset(
        split, stride=1, preserve_indices=True, replace_invalid=True
    )
    pos = [split.index_map[g][1] for g in exact.indices].index(bad)
    with caplog.at_level(logging.WARNING):
        first = _frame(exact[pos])
    assert "Replacing invalid val frame" in caplog.text
    assert first in kept and first not in (bad, NAN, SWAPPED)
    assert all(_frame(exact[pos]) == first for _ in range(5))
    rebuilt = EvenStrideDataset(
        split, stride=1, preserve_indices=True, replace_invalid=True
    )
    assert _frame(rebuilt[pos]) == first, "same cohort after a resume"


def test_nested_multidatasets_resolve_to_the_leaf():
    inner = MultiDataset(datasets={"ep": _Leaf()}, mode="total")
    outer = _split(group=inner)
    exact = EvenStrideDataset(outer, stride=1, preserve_indices=True)
    frames = [_frame(exact[i]) for i in range(NAN)]
    assert frames == list(range(NAN))


def test_metric_heads_read_exactly_by_default():
    split = _split(ep=_Leaf())
    cfg = OmegaConf.create({"data": {"metric_frames_per_episode": {"valid": 12}}})
    (wrapped,) = th._subsample_val_datasets(cfg, "valid", {"h": split}).values()
    assert wrapped.preserve_indices and wrapped.replace_invalid
    assert _frame(wrapped[OUTLIER]) == OUTLIER
    assert _frame(wrapped[NAN]) != NAN
    cfg.data.exact_val_reads = False
    (wrapped,) = th._subsample_val_datasets(cfg, "valid", {"h": split}).values()
    assert not wrapped.preserve_indices


# ------------------------------------------------- real zarr: leaf + mask + video
T = 90


def _zarr_split(tmp_path, jump_at=None):
    t = np.arange(T, dtype=np.float64)[:, None]
    ee = np.concatenate(
        [t * 0.001 + np.zeros((T, 3)), np.tile([[1.0, 0, 0, 0]], (T, 1))], 1
    )
    head = ee.copy()
    if jump_at is not None:
        head[jump_at:, 0] += 0.2
    ZarrWriter.create_and_write(
        tmp_path / "ep.zarr",
        numeric_data={
            "left.obs_ee_pose": ee,
            "right.obs_ee_pose": ee,
            "obs_head_pose": head,
        },
        image_data={
            "images.front_1": np.random.default_rng(0).integers(
                0, 255, (T, 16, 16, 3), np.uint8
            )
        },
        embodiment="human_bimanual",
        fps=30,
        task_name="synthetic",
        intrinsics={"front_1": np.eye(3, 4)},
    )
    resolver = LocalEpisodeResolver(
        tmp_path,
        key_map=Human.get_keymap(keymap_mode="cartesian"),
        transform_list=[],
    )
    return MultiDataset._from_resolver(
        resolver, mode="total", pose_glitch_mask={"step_m": 0.05, "rot_deg": 10.0}
    )


def test_zarr_leaf_emits_frame_idx_and_exact_reads_respect_the_mask(tmp_path):
    split = _zarr_split(tmp_path, jump_at=45)
    (leaf,) = split.datasets.values()
    assert leaf.sample_indices is not None
    assert int(leaf[17]["frame_idx"]) == 17
    exact = EvenStrideDataset(split, frames_per_episode=20, preserve_indices=True)
    kept = set(leaf.sample_indices.tolist())
    for i in range(len(exact)):
        expected = split.index_map[exact.indices[i]][1]
        assert expected in kept
        assert int(exact[i]["frame_idx"]) == expected


def test_pinned_video_loader_still_collates(tmp_path):
    split = _zarr_split(tmp_path, jump_at=45)
    video = pinned_episode_subset(split, list(split.datasets))
    loader = torch.utils.data.DataLoader(
        video, batch_size=8, shuffle=False, collate_fn=annotation_collate
    )
    batch = next(iter(loader))
    assert batch["frame_idx"].tolist() == list(range(8))
