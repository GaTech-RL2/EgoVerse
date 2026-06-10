"""Unit tests for K action-chunk stacking on the packed zarr loader path.

Covers the three dataset-side knobs added to ``ZarrEpisodePackedDataset``
(and the ``action_stride`` support added to ``ZarrDataset._read_span``):

  1. ``action_chunk_k`` (D07) — keyframes every K post-stride frames, each
     carrying its K actions flattened TIME-MAJOR to ``K*action_dim``.
  2. ``action_pairing`` (D08) — ``"current"`` (keyframe t carries actions
     ``[t, t+K)``; the K=1-compatible generalization of frame-t ↔
     ``actions[t]``) vs ``"incoming_meanpad"`` (upstream-exact pureDF
     ``pusht_dataset.py`` convention: drop_back=1, floor keyframe count,
     keyframe i ≥ 1 carries the chunk that LED INTO it, keyframe 0 is the
     per-dim action mean tiled across K slots).
  3. ``chunk_phase_aug`` (D26) — per-sample random phase in ``[0, K)``
     (upstream ``drop_front``), default off.

Every test also pins the bit-compat contract: at the default knob values
(K=1, aug off, "current") the dataset reproduces the pre-knob behavior
exactly — identical index, ``torch.equal`` samples, and ZERO numpy RNG
consumption.

All fixtures are tiny synthetic zarr episodes written to ``tmp_path`` —
pure-CPU, no cluster data, no JPEG keys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.rldb.zarr.zarr_dataset_packed import (
    ZarrEpisodePackedDataset,
    pack_collate,
)

# --------------------------------------------------------------------------- #
# Synthetic zarr fixtures
# --------------------------------------------------------------------------- #

_EMBODIMENT = "pushshapes_sim"

# Per-frame numeric keys only — no "jpeg"/"json" dtypes, so the loader skips
# the decode branches entirely and the tests stay pure-CPU.
KEY_MAP = {
    "state": {"zarr_key": "state", "key_type": "proprio_keys"},
    "actions": {"zarr_key": "actions", "key_type": "action_keys"},
}

NO_ACTION_KEY_MAP = {
    "state": {"zarr_key": "state", "key_type": "proprio_keys"},
}


def _write_episode(root: Path, name: str, total_frames: int, base: float) -> None:
    """Write one synthetic episode. Frame t carries ``base + t`` in every
    channel (actions: ``[base + t, -(base + t)]``), so frame identity
    survives striding / chunking / padding checks."""
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


def _load_episodes(root: Path, key_map: dict = KEY_MAP) -> dict[str, ZarrDataset]:
    return {
        p.name[: -len(".zarr")]: ZarrDataset(p, key_map=key_map, transform_list=None)
        for p in sorted(root.iterdir())
        if p.name.endswith(".zarr")
    }


@pytest.fixture()
def episodes(tmp_path: Path) -> dict[str, ZarrDataset]:
    _write_episode(tmp_path, "ep_a", total_frames=20, base=0.0)
    _write_episode(tmp_path, "ep_b", total_frames=17, base=100.0)
    return _load_episodes(tmp_path)


def _packed(episodes, **kwargs) -> ZarrEpisodePackedDataset:
    defaults = dict(chunking="none", min_seq_len=1, max_seq_len=64)
    defaults.update(kwargs)
    return ZarrEpisodePackedDataset(datasets=episodes, **defaults)


def _sample_by_key(ds: ZarrEpisodePackedDataset, key: str) -> dict:
    for i, (k, _start, _end) in enumerate(ds.index):
        if k == key:
            return ds[i]
    raise KeyError(key)


def _action_row(base: float, frames: list[int]) -> torch.Tensor:
    """Expected stacked action row for raw ``frames`` of an episode with
    value base ``base`` — time-major ``[a0_x, a0_y, a1_x, a1_y, ...]``."""
    vals = []
    for t in frames:
        vals += [base + t, -(base + t)]
    return torch.tensor(vals, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Bit-compat: K=1 + "current" == pre-knob behavior, zero RNG
# --------------------------------------------------------------------------- #


class TestK1BitCompat:
    def test_index_and_samples_identical(self, episodes):
        implicit = _packed(episodes)
        explicit = _packed(
            episodes,
            action_chunk_k=1,
            chunk_phase_aug=False,
            action_pairing="current",
        )
        assert implicit.index == explicit.index
        for i in range(len(implicit)):
            a, b = implicit[i], explicit[i]
            assert a.keys() == b.keys()
            for k in ("state", "actions"):
                assert torch.equal(a[k], b[k])
            assert a["seq_len"] == b["seq_len"]
            assert a["chunk_offset"] == b["chunk_offset"]

    def test_k1_actions_keep_per_frame_shape(self, episodes):
        sample = _sample_by_key(_packed(episodes, action_chunk_k=1), "ep_a")
        assert sample["actions"].shape == (20, 2)
        assert sample["seq_len"] == 20

    @pytest.mark.parametrize("phase_aug", [False, True])
    def test_no_rng_consumed_at_k1(self, episodes, phase_aug):
        # Upstream draws drop_front only at K > 1 (`randint(K) if K > 1
        # else 0`); K=1 must not advance numpy's global RNG even with the
        # aug knob on.
        ds = _packed(episodes, action_chunk_k=1, chunk_phase_aug=phase_aug)
        np.random.seed(1234)
        before = np.random.get_state()
        _ = ds[0]
        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]

    def test_no_rng_consumed_with_aug_off_at_k4(self, episodes):
        ds = _packed(episodes, action_chunk_k=4)
        np.random.seed(1234)
        before = np.random.get_state()
        _ = ds[0]
        after = np.random.get_state()
        assert np.array_equal(before[1], after[1])


# --------------------------------------------------------------------------- #
# action_pairing="current" — stacking correctness vs hand-built episode
# --------------------------------------------------------------------------- #


class TestCurrentPairing:
    def test_stacks_match_hand_built(self, episodes):
        # ep_a: 20 frames, K=4 -> 5 keyframes at raw 0,4,8,12,16; row i =
        # actions [4i, 4i+4) flattened time-major.
        ds = _packed(episodes, action_chunk_k=4)
        sample = _sample_by_key(ds, "ep_a")
        assert sample["seq_len"] == 5
        assert sample["actions"].shape == (5, 8)
        assert sample["state"].shape == (5, 3)
        expected_keyframes = torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0])
        assert torch.equal(sample["state"][:, 0], expected_keyframes)
        for i in range(5):
            expected = _action_row(0.0, [4 * i + j for j in range(4)])
            assert torch.equal(sample["actions"][i], expected)

    def test_layout_is_time_major(self, episodes):
        # Row i reshaped (K, D) must recover the K consecutive raw actions —
        # the decode contract the stage side relies on.
        sample = _sample_by_key(_packed(episodes, action_chunk_k=4), "ep_a")
        chunk = sample["actions"][1].reshape(4, 2)
        t = torch.arange(4, 8, dtype=torch.float32)
        assert torch.equal(chunk[:, 0], t)
        assert torch.equal(chunk[:, 1], -t)

    def test_trailing_partial_chunk_holds_final_action(self, episodes):
        # ep_b: 17 frames, K=4 -> 5 keyframes; last chunk has 1 real action
        # (raw 16) repeated to fill the K slots.
        sample = _sample_by_key(_packed(episodes, action_chunk_k=4), "ep_b")
        assert sample["seq_len"] == 5
        assert torch.equal(sample["actions"][4], _action_row(100.0, [16, 16, 16, 16]))
        # Penultimate chunk is fully real.
        assert torch.equal(sample["actions"][3], _action_row(100.0, [12, 13, 14, 15]))


# --------------------------------------------------------------------------- #
# action_pairing="incoming_meanpad" — upstream-exact convention
# --------------------------------------------------------------------------- #


class TestIncomingMeanpad:
    def test_stacks_match_hand_built(self, episodes):
        # ep_a: 20 frames, K=4. Upstream: drop_back=1 -> T'=19, n_key =
        # 19//4 = 4 keyframes at raw 0,4,8,12. Keyframe i >= 1 carries
        # actions [(i-1)*4, i*4); keyframe 0 is the tiled per-dim mean of
        # the trimmed action stream (raw 0..18).
        ds = _packed(episodes, action_chunk_k=4, action_pairing="incoming_meanpad")
        sample = _sample_by_key(ds, "ep_a")
        assert sample["seq_len"] == 4
        assert sample["actions"].shape == (4, 8)
        assert torch.equal(sample["state"][:, 0], torch.tensor([0.0, 4.0, 8.0, 12.0]))
        for i in range(1, 4):
            expected = _action_row(0.0, [4 * (i - 1) + j for j in range(4)])
            assert torch.equal(sample["actions"][i], expected)
        mean_xy = torch.tensor([9.0, -9.0])  # mean of 0..18 / -(0..18)
        assert torch.equal(sample["actions"][0], mean_xy.repeat(4))

    def test_k1_incoming_shifts_by_one_frame(self, episodes):
        # K=1: keyframes are frames 0..18 (drop_back=1); frame i carries
        # actions[i-1]; frame 0 carries the stream mean.
        ds = _packed(episodes, action_chunk_k=1, action_pairing="incoming_meanpad")
        sample = _sample_by_key(ds, "ep_a")
        assert sample["seq_len"] == 19
        assert sample["state"].shape == (19, 3)
        assert sample["actions"].shape == (19, 2)
        t = torch.arange(0, 18, dtype=torch.float32)
        assert torch.equal(sample["actions"][1:, 0], t)
        assert torch.equal(sample["actions"][1:, 1], -t)
        assert torch.equal(sample["actions"][0], torch.tensor([9.0, -9.0]))

    def test_obs_trimmed_to_floor_keyframes(self, episodes):
        # ep_b: 17 frames, K=4 -> T'=16, n_key=4; the ceil-len obs read (5
        # keyframes) must be trimmed to 4.
        ds = _packed(episodes, action_chunk_k=4, action_pairing="incoming_meanpad")
        sample = _sample_by_key(ds, "ep_b")
        assert sample["seq_len"] == 4
        assert sample["state"].shape == (4, 3)
        assert torch.equal(
            sample["state"][:, 0], torch.tensor([100.0, 104.0, 108.0, 112.0])
        )


# --------------------------------------------------------------------------- #
# D26 — chunk_phase_aug bounds
# --------------------------------------------------------------------------- #


class TestPhaseAug:
    def test_phase_in_bounds_and_varies(self, episodes):
        ds = _packed(episodes, action_chunk_k=4, chunk_phase_aug=True)
        np.random.seed(0)
        phases = set()
        for _ in range(64):
            sample = _sample_by_key(ds, "ep_a")
            # ep_a's index span starts at 0, so chunk_offset == phase.
            phase = int(sample["chunk_offset"])
            assert 0 <= phase < 4
            # First keyframe is raw frame `phase`; chunk 0 starts there too.
            assert sample["state"][0, 0] == float(phase)
            assert sample["actions"][0, 0] == float(phase)
            phases.add(phase)
        assert phases == {0, 1, 2, 3}

    def test_min_seq_len_gates_on_worst_case_phase(self, tmp_path):
        # 18 frames @ K=4: phase 0 emits ceil(18/4)=5 keyframes but phase 3
        # only ceil(15/4)=4 — gating must use the minimum.
        _write_episode(tmp_path, "ep_s", total_frames=18, base=0.0)
        eps = _load_episodes(tmp_path)
        ds = _packed(eps, action_chunk_k=4, chunk_phase_aug=True, min_seq_len=4)
        assert len(ds.index) == 1
        with pytest.raises(ValueError, match="empty index"):
            _packed(eps, action_chunk_k=4, chunk_phase_aug=True, min_seq_len=5)
        # Without the aug the phase is pinned to 0 and 5 keyframes are
        # guaranteed.
        ds = _packed(eps, action_chunk_k=4, min_seq_len=5)
        assert len(ds.index) == 1


# --------------------------------------------------------------------------- #
# temporal_stride composes multiplicatively
# --------------------------------------------------------------------------- #


class TestStrideComposition:
    def test_keyframes_and_chunks_on_virtual_stream(self, episodes):
        # ts=2, K=2: virtual stream = raw 0,2,..,18; keyframes every 4 raw
        # frames; chunk i = virtual actions [2i, 2i+2) = raw [4i, 4i+2].
        ds = _packed(episodes, temporal_stride=2, action_chunk_k=2)
        sample = _sample_by_key(ds, "ep_a")
        assert sample["seq_len"] == 5
        assert torch.equal(
            sample["state"][:, 0], torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0])
        )
        for i in range(5):
            assert torch.equal(
                sample["actions"][i], _action_row(0.0, [4 * i, 4 * i + 2])
            )


# --------------------------------------------------------------------------- #
# Index gating in keyframe units + hold_pad / pack_collate composition
# --------------------------------------------------------------------------- #


class TestIndexAndComposition:
    def test_max_seq_len_counts_keyframes(self, episodes):
        ds = _packed(episodes, action_chunk_k=4, max_seq_len=5)
        assert len(ds.index) == 2
        with pytest.raises(ValueError, match="max_seq_len"):
            _packed(episodes, action_chunk_k=4, max_seq_len=4)

    def test_zero_keyframe_span_dropped_even_with_hold_pad(self, tmp_path):
        # 4 frames @ K=4 incoming: (4-1)//4 = 0 keyframes — nothing to pad.
        _write_episode(tmp_path, "ep_tiny", total_frames=4, base=0.0)
        _write_episode(tmp_path, "ep_ok", total_frames=12, base=50.0)
        eps = _load_episodes(tmp_path)
        ds = _packed(
            eps,
            action_chunk_k=4,
            action_pairing="incoming_meanpad",
            tail_mode="hold_pad",
        )
        assert [k for k, _s, _e in ds.index] == ["ep_ok"]

    def test_hold_pad_pads_in_keyframe_units(self, episodes):
        # ep_a @ K=4 emits 5 keyframes; min_seq_len=6 + hold_pad repeats the
        # final keyframe AND its stacked chunk.
        ds = _packed(episodes, action_chunk_k=4, min_seq_len=6, tail_mode="hold_pad")
        sample = _sample_by_key(ds, "ep_a")
        assert sample["seq_len"] == 6
        assert torch.equal(sample["state"][5], sample["state"][4])
        assert torch.equal(sample["actions"][5], sample["actions"][4])

    def test_pack_collate_sees_keyframe_lengths(self, episodes):
        ds = _packed(episodes, action_chunk_k=4)
        batch = pack_collate([ds[i] for i in range(len(ds))])
        assert batch["seq_lens"].tolist() == [5, 5]
        assert batch["cu_seqlens"].tolist() == [0, 5, 10]
        assert batch["actions"].shape == (10, 8)
        assert batch["state"].shape == (10, 3)


# --------------------------------------------------------------------------- #
# Validation + _read_span action_stride unit behavior
# --------------------------------------------------------------------------- #


class TestValidationAndReadSpan:
    def test_rejects_bad_knobs(self, episodes):
        with pytest.raises(ValueError, match="action_chunk_k"):
            _packed(episodes, action_chunk_k=0)
        with pytest.raises(ValueError, match="action_pairing"):
            _packed(episodes, action_pairing="outgoing")

    def test_chunking_requires_an_action_key(self, tmp_path):
        _write_episode(tmp_path, "ep_a", total_frames=20, base=0.0)
        eps = _load_episodes(tmp_path, key_map=NO_ACTION_KEY_MAP)
        ds = _packed(eps, action_chunk_k=4)
        with pytest.raises(RuntimeError, match="action_keys"):
            _ = ds[0]

    def test_read_span_action_stride_splits_rates(self, episodes):
        ds = episodes["ep_a"]
        data = ds._read_span(0, 20, stride=4, action_stride=1)
        # Obs at keyframe rate, actions at full rate; seq_len follows obs.
        assert data["seq_len"] == 5
        assert data["state"].shape == (5, 3)
        assert data["actions"].shape == (20, 2)
        full = ds._read_span(0, 20)
        assert torch.equal(data["actions"], full["actions"])
        assert torch.equal(data["state"], full["state"][::4])

    def test_read_span_action_stride_default_identical(self, episodes):
        ds = episodes["ep_a"]
        a = ds._read_span(0, 20, stride=2)
        b = ds._read_span(0, 20, stride=2, action_stride=None)
        for k in ("state", "actions"):
            assert torch.equal(a[k], b[k])

    def test_read_span_rejects_bad_action_stride(self, episodes):
        with pytest.raises(ValueError, match="action_stride"):
            episodes["ep_a"]._read_span(0, 20, action_stride=0)
