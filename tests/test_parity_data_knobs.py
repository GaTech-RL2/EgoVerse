"""Unit tests for the DFoT-parity data knobs on the packed zarr loader path.

Covers the three knobs added to ``ZarrEpisodePackedDataset`` (and the
``stride`` support added to ``ZarrDataset._read_span``):

  1. ``temporal_stride`` (D07) — every-Nth-frame keyframe subsampling at read
     time. ``max_seq_len`` / ``min_seq_len`` are interpreted in post-stride
     frames; index spans stay in raw-frame coordinates.
  2. ``tail_mode`` (D17) — ``"hold_pad"`` keeps samples shorter than
     ``min_seq_len`` and repeat-pads them at read time by holding the final
     frame (observation AND action row), instead of silently dropping them.
  3. ``val_holdout_first_n`` + ``holdout_split`` (D14) — config-expressible
     held-out val split: the first n sorted episodes go exclusively to the
     ``"val"`` side and are excluded from the ``"train"`` side.

Every test also pins the bit-compat contract: at the default knob values
(stride 1, ``"drop"``, 0) the dataset reproduces the pre-knob behavior
exactly (identical index, ``torch.equal`` samples).

All fixtures are tiny synthetic zarr episodes written to ``tmp_path`` —
pure-CPU, no cluster data, no JPEG keys (so no simplejpeg encode needed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.rldb.zarr.zarr_dataset_packed import (
    ZarrEpisodePackedDataset,
    pack_collate,
)

# --------------------------------------------------------------------------- #
# Synthetic zarr fixtures
# --------------------------------------------------------------------------- #

_EMBODIMENT = "pushshapes_sim"

# Per-frame numeric keys only — no "jpeg"/"json" dtypes, so both loader paths
# skip the decode branches entirely and the tests stay pure-CPU.
KEY_MAP = {
    "state": {"zarr_key": "state", "key_type": "proprio_keys"},
    "actions": {"zarr_key": "actions", "key_type": "action_keys"},
}


def _write_episode(root: Path, name: str, total_frames: int, base: float) -> None:
    """Write one synthetic episode. Frame t carries the value ``base + t`` in
    every channel, so frame identity survives striding/padding checks."""
    g = zarr.open_group(str(root / f"{name}.zarr"), mode="w")
    g.attrs["total_frames"] = total_frames
    g.attrs["embodiment"] = _EMBODIMENT
    g.attrs["features"] = {
        "state": {"dtype": "float32"},
        "actions": {"dtype": "float32"},
    }
    t = np.arange(total_frames, dtype=np.float32) + base
    g.create_array("state", data=np.stack([t, t, t], axis=1))
    g.create_array("actions", data=np.stack([t, -t], axis=1))


def _load_episodes(root: Path) -> dict[str, ZarrDataset]:
    return {
        p.name[: -len(".zarr")]: ZarrDataset(p, key_map=KEY_MAP, transform_list=None)
        for p in sorted(root.iterdir())
        if p.name.endswith(".zarr")
    }


@pytest.fixture()
def episodes(tmp_path: Path) -> dict[str, ZarrDataset]:
    """Four episodes, sorted keys ep_a < ep_b < ep_c < ep_d, with one episode
    (ep_c) shorter than the min_seq_len=8 used throughout the tests."""
    _write_episode(tmp_path, "ep_a", total_frames=20, base=0.0)
    _write_episode(tmp_path, "ep_b", total_frames=17, base=100.0)
    _write_episode(tmp_path, "ep_c", total_frames=5, base=200.0)  # < min_seq_len
    _write_episode(tmp_path, "ep_d", total_frames=12, base=300.0)
    return _load_episodes(tmp_path)


def _packed(episodes, **kwargs) -> ZarrEpisodePackedDataset:
    defaults = dict(chunking="none", min_seq_len=8, max_seq_len=64)
    defaults.update(kwargs)
    return ZarrEpisodePackedDataset(datasets=episodes, **defaults)


def _sample_by_key(ds: ZarrEpisodePackedDataset, key: str) -> dict:
    for i, (k, _start, _end) in enumerate(ds.index):
        if k == key:
            return ds[i]
    raise KeyError(key)


# --------------------------------------------------------------------------- #
# Bit-compat: explicit defaults == implicit defaults
# --------------------------------------------------------------------------- #


class TestDefaultsBitCompat:
    def test_index_and_samples_identical(self, episodes):
        implicit = _packed(episodes)
        explicit = _packed(
            episodes,
            temporal_stride=1,
            tail_mode="drop",
            val_holdout_first_n=0,
            holdout_split="train",
        )
        assert implicit.index == explicit.index
        # Short ep_c dropped, as before the knobs existed.
        assert [k for k, _s, _e in implicit.index] == ["ep_a", "ep_b", "ep_d"]
        for i in range(len(implicit)):
            a, b = implicit[i], explicit[i]
            assert a.keys() == b.keys()
            for k in ("state", "actions"):
                assert torch.equal(a[k], b[k])
            assert a["seq_len"] == b["seq_len"]
            assert a["episode_idx"] == b["episode_idx"]

    def test_holdout_split_val_is_noop_at_zero(self, episodes):
        # The valid_datasets yaml entry pre-sets holdout_split="val" with
        # val_holdout_first_n=0 — must keep ALL episodes (current behavior).
        ds = _packed(episodes, val_holdout_first_n=0, holdout_split="val")
        assert [k for k, _s, _e in ds.index] == ["ep_a", "ep_b", "ep_d"]

    def test_read_span_default_stride_identical(self, episodes):
        ds = episodes["ep_a"]
        no_kwarg = ds._read_span(0, 20)
        explicit = ds._read_span(0, 20, stride=1)
        for k in ("state", "actions"):
            assert torch.equal(no_kwarg[k], explicit[k])
        assert no_kwarg["seq_len"] == explicit["seq_len"] == 20


# --------------------------------------------------------------------------- #
# D07 — temporal_stride
# --------------------------------------------------------------------------- #


class TestTemporalStride:
    def test_read_span_stride_matches_python_slicing(self, episodes):
        ds = episodes["ep_a"]
        full = ds._read_span(0, 20)
        for stride in (2, 3, 7):
            strided = ds._read_span(0, 20, stride=stride)
            assert strided["seq_len"] == len(range(0, 20, stride))
            for k in ("state", "actions"):
                assert torch.equal(strided[k], full[k][::stride])

    def test_read_span_rejects_bad_stride(self, episodes):
        with pytest.raises(ValueError, match="stride"):
            episodes["ep_a"]._read_span(0, 20, stride=0)

    def test_packed_stride_subsamples_frames(self, episodes):
        ds = _packed(episodes, temporal_stride=2, min_seq_len=4)
        sample = _sample_by_key(ds, "ep_b")  # 17 raw frames
        assert sample["seq_len"] == 9  # ceil(17 / 2)
        # Frame t encodes base + t: every 2nd raw frame of ep_b (base=100).
        expected = torch.arange(0, 17, 2, dtype=torch.float32) + 100.0
        assert torch.equal(sample["state"][:, 0], expected)
        assert torch.equal(sample["actions"][:, 1], -expected)

    def test_min_seq_len_is_post_stride(self, episodes):
        # ep_d has 12 raw frames -> 3 strided frames at stride 4: dropped at
        # min_seq_len=4, kept at min_seq_len=3.
        ds = _packed(episodes, temporal_stride=4, min_seq_len=4)
        assert "ep_d" not in {k for k, _s, _e in ds.index}
        ds = _packed(episodes, temporal_stride=4, min_seq_len=3)
        assert "ep_d" in {k for k, _s, _e in ds.index}

    def test_sequential_chunking_chunks_in_strided_units(self, episodes):
        # max_seq_len=5 post-stride frames @ stride 2 -> raw chunks of 10.
        ds = _packed(
            episodes,
            chunking="sequential",
            temporal_stride=2,
            min_seq_len=2,
            max_seq_len=5,
        )
        ep_a_spans = [(s, e) for k, s, e in ds.index if k == "ep_a"]
        assert ep_a_spans == [(0, 10), (10, 20)]
        for i, (k, _s, _e) in enumerate(ds.index):
            if k == "ep_a":
                assert ds[i]["seq_len"] == 5

    def test_chunking_none_cap_is_post_stride(self, episodes):
        # ep_a (20 frames) violates max_seq_len=12 raw, but fits at stride 2
        # (10 strided frames).
        with pytest.raises(ValueError, match="max_seq_len"):
            _packed(episodes, max_seq_len=12)
        ds = _packed(episodes, max_seq_len=12, temporal_stride=2, min_seq_len=3)
        assert "ep_a" in {k for k, _s, _e in ds.index}

    def test_rejects_bad_stride(self, episodes):
        with pytest.raises(ValueError, match="temporal_stride"):
            _packed(episodes, temporal_stride=0)


# --------------------------------------------------------------------------- #
# D17 — tail_mode="hold_pad"
# --------------------------------------------------------------------------- #


class TestTailModeHoldPad:
    def test_drop_is_default_and_drops(self, episodes):
        ds = _packed(episodes)
        assert ds.tail_mode == "drop"
        assert "ep_c" not in {k for k, _s, _e in ds.index}

    def test_hold_pad_keeps_and_pads_short_episode(self, episodes):
        ds = _packed(episodes, tail_mode="hold_pad")
        assert "ep_c" in {k for k, _s, _e in ds.index}
        sample = _sample_by_key(ds, "ep_c")  # 5 raw frames -> pad to 8
        assert sample["seq_len"] == 8
        assert sample["state"].shape[0] == 8
        assert sample["actions"].shape[0] == 8
        # Prefix is the untouched raw read; tail holds the final frame AND
        # the final action row (upstream pureDF repeat-pad convention).
        raw = episodes["ep_c"]._read_span(0, 5)
        for k in ("state", "actions"):
            assert torch.equal(sample[k][:5], raw[k])
            assert torch.equal(sample[k][5:], raw[k][-1:].repeat(3, 1))

    def test_hold_pad_leaves_long_episodes_untouched(self, episodes):
        padded = _packed(episodes, tail_mode="hold_pad")
        plain = _packed(episodes)
        for key in ("ep_a", "ep_b", "ep_d"):
            a = _sample_by_key(padded, key)
            b = _sample_by_key(plain, key)
            assert a["seq_len"] == b["seq_len"]
            for k in ("state", "actions"):
                assert torch.equal(a[k], b[k])

    def test_hold_pad_sequential_trailing_chunk(self, episodes):
        # ep_b (17 frames) @ max_seq_len=8: trailing chunk of 1 raw frame is
        # dropped under "drop", kept + padded to min_seq_len under hold_pad.
        drop = _packed(episodes, chunking="sequential", min_seq_len=4, max_seq_len=8)
        assert [(s, e) for k, s, e in drop.index if k == "ep_b"] == [(0, 8), (8, 16)]
        pad = _packed(
            episodes,
            chunking="sequential",
            min_seq_len=4,
            max_seq_len=8,
            tail_mode="hold_pad",
        )
        spans = [(s, e) for k, s, e in pad.index if k == "ep_b"]
        assert spans == [(0, 8), (8, 16), (16, 17)]
        tail_i = pad.index.index(("ep_b", 16, 17))
        tail = pad[tail_i]
        assert tail["seq_len"] == 4
        # All four frames hold raw frame 16 of ep_b (value base + 16).
        assert torch.equal(
            tail["state"][:, 0], torch.full((4,), 116.0, dtype=torch.float32)
        )

    def test_hold_pad_with_stride(self, episodes):
        # ep_c: 5 raw frames @ stride 2 -> 3 strided frames, padded to 4.
        ds = _packed(episodes, tail_mode="hold_pad", temporal_stride=2, min_seq_len=4)
        sample = _sample_by_key(ds, "ep_c")
        assert sample["seq_len"] == 4
        expected = torch.tensor([200.0, 202.0, 204.0, 204.0])
        assert torch.equal(sample["state"][:, 0], expected)

    def test_pack_collate_uses_padded_lengths(self, episodes):
        ds = _packed(episodes, tail_mode="hold_pad")
        batch = pack_collate([ds[i] for i in range(len(ds))])
        assert batch["seq_lens"].tolist() == [20, 17, 8, 12]
        assert batch["cu_seqlens"].tolist() == [0, 20, 37, 45, 57]
        assert batch["state"].shape[0] == 57

    def test_rejects_bad_tail_mode(self, episodes):
        with pytest.raises(ValueError, match="tail_mode"):
            _packed(episodes, tail_mode="zero_pad")


# --------------------------------------------------------------------------- #
# D14 — val_holdout_first_n + holdout_split
# --------------------------------------------------------------------------- #


class TestValHoldout:
    def test_split_is_disjoint_and_exhaustive(self, episodes):
        train = _packed(
            episodes, val_holdout_first_n=2, holdout_split="train", min_seq_len=4
        )
        val = _packed(
            episodes, val_holdout_first_n=2, holdout_split="val", min_seq_len=4
        )
        train_keys = {k for k, _s, _e in train.index}
        val_keys = {k for k, _s, _e in val.index}
        # First n in SORTED order go exclusively to val.
        assert val_keys == {"ep_a", "ep_b"}
        assert train_keys == {"ep_c", "ep_d"}
        assert train_keys & val_keys == set()
        assert train_keys | val_keys == set(episodes.keys())

    def test_episode_idx_is_split_invariant(self, episodes):
        full = _packed(episodes, min_seq_len=4)
        train = _packed(
            episodes, val_holdout_first_n=2, holdout_split="train", min_seq_len=4
        )
        val = _packed(
            episodes, val_holdout_first_n=2, holdout_split="val", min_seq_len=4
        )

        def idx_of(ds, key):
            return int(_sample_by_key(ds, key)["episode_idx"])

        for key in ("ep_a", "ep_b"):
            assert idx_of(val, key) == idx_of(full, key)
        for key in ("ep_c", "ep_d"):
            assert idx_of(train, key) == idx_of(full, key)

    def test_samples_match_full_dataset(self, episodes):
        full = _packed(episodes, min_seq_len=4)
        val = _packed(
            episodes, val_holdout_first_n=2, holdout_split="val", min_seq_len=4
        )
        for key in ("ep_a", "ep_b"):
            a = _sample_by_key(val, key)
            b = _sample_by_key(full, key)
            for k in ("state", "actions"):
                assert torch.equal(a[k], b[k])

    def test_embodiment_tagging_survives(self, episodes):
        ds = _packed(episodes, val_holdout_first_n=1, holdout_split="val")
        sample = ds[0]
        assert sample["embodiment"] == get_embodiment_id(_EMBODIMENT)

    def test_empty_side_raises(self, episodes):
        with pytest.raises(ValueError, match="no episodes"):
            _packed(episodes, val_holdout_first_n=4, holdout_split="train")

    def test_rejects_bad_args(self, episodes):
        with pytest.raises(ValueError, match="val_holdout_first_n"):
            _packed(episodes, val_holdout_first_n=-1)
        with pytest.raises(ValueError, match="holdout_split"):
            _packed(episodes, val_holdout_first_n=1, holdout_split="test")
