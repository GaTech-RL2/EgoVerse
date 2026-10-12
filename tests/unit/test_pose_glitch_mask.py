"""``pose_glitch_mask``: start frames whose non-image reads touch a camera-pose
jump are left out of ``MultiDataset``'s index, everything else is kept."""

from __future__ import annotations

import numpy as np

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import (
    EvenStrideDataset,
    LocalEpisodeResolver,
    MultiDataset,
    pose_glitch_frames,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

T = 120


def _pose(jumps=(), turn_at=None):
    t = np.arange(T, dtype=np.float64)[:, None]
    xyz = t * 0.001 + np.zeros((T, 3))
    for f in jumps:
        xyz[f:, 0] += 0.2
    quat = np.tile([[1.0, 0.0, 0.0, 0.0]], (T, 1))
    if turn_at is not None:
        half = np.radians(30) / 2
        quat[turn_at:] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    return np.concatenate([xyz, quat], axis=1)


def _dataset(tmp_path, head_pose, mask=True, **keymap_kwargs):
    rng = np.random.default_rng(0)
    ee = _pose()
    ZarrWriter.create_and_write(
        tmp_path / "ep.zarr",
        numeric_data={
            "left.obs_ee_pose": ee,
            "right.obs_ee_pose": ee,
            "obs_head_pose": head_pose,
        },
        image_data={"images.front_1": rng.integers(0, 255, (T, 16, 16, 3), np.uint8)},
        embodiment="human_bimanual",
        fps=30,
        task_name="synthetic",
        intrinsics={"front_1": np.eye(3, 4)},
    )
    key_map = Human.get_keymap(keymap_mode="cartesian", **keymap_kwargs)
    resolver = LocalEpisodeResolver(tmp_path, key_map=key_map, transform_list=[])
    return MultiDataset._from_resolver(
        resolver,
        mode="total",
        pose_glitch_mask={"step_m": 0.05, "rot_deg": 10.0} if mask else None,
    )


def test_glitch_frames_are_both_sides_of_a_jump_or_turn():
    assert pose_glitch_frames(_pose(jumps=[40])).tolist() == [39, 40]
    assert pose_glitch_frames(_pose(turn_at=70)).tolist() == [69, 70]
    assert pose_glitch_frames(_pose()).size == 0


def test_masked_starts_are_exactly_those_whose_reads_touch_a_glitch(tmp_path):
    ds = _dataset(tmp_path, _pose(jumps=[60]), proprio_history=4, history_stride=2)
    leaf = next(iter(ds.datasets.values()))
    lo, hi = leaf._read_span_offsets()
    assert lo == -6 and hi > 0
    expected = [t for t in range(T) if not (t + lo <= 60 and 59 <= t + hi)]
    assert leaf.sample_indices.tolist() == expected
    assert [i for _, i in ds.index_map] == expected
    assert len(ds) == len(expected) < T


def test_clean_episode_and_mask_off_keep_every_frame(tmp_path):
    clean = _dataset(tmp_path / "a", _pose())
    assert (
        len(clean) == T and next(iter(clean.datasets.values())).sample_indices is None
    )
    off = _dataset(tmp_path / "b", _pose(jumps=[60]), mask=False)
    assert len(off) == T


def test_even_stride_draws_only_kept_starts(tmp_path):
    ds = _dataset(tmp_path, _pose(jumps=[30, 90]))
    kept = set(next(iter(ds.datasets.values())).sample_indices.tolist())
    strided = EvenStrideDataset(ds, frames_per_episode=10)
    for i in range(len(strided)):
        _, local = ds.index_map[strided.indices[i]]
        assert local in kept


def test_video_subset_plays_through_the_mask_but_metrics_keep_it(tmp_path):
    from egomimic.rldb.zarr.zarr_dataset_multi import pinned_episode_subset

    ds = _dataset(tmp_path, _pose(jumps=[60]))
    (name,) = ds.datasets
    video = pinned_episode_subset(ds, [name])
    assert len(video) == T
    assert len(ds) < T and ds.datasets[name].sample_indices is not None
