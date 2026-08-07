"""Behavioral-equality suite for the two zarr loader paths (dedup collapse 3).

Collapse 3 factors the shared read/decode logic of the two near-duplicate
zarr loaders — the **padded/windowed** path (``ZarrDataset.__getitem__``) and
the **packed/span** path (``ZarrDataset._read_span`` consumed by
``ZarrEpisodePackedDataset``) — into ``egomimic/rldb/zarr/_common.py``. Each
loader keeps only its genuine differences (windowing+padding+JPEG-fail
resample loop vs. exact-span read + ``seq_len`` metadata).

This suite is the PROOF that the refactor is behavior-preserving. It is meant
to pass BOTH before the refactor (anchoring the reference behavior captured
from the pre-collapse code at tag ``dedup-c3-pre``) and after (proving the
factored helpers reproduce that behavior bit-for-bit).

What it proves
--------------
1. ``test_reference_hashes`` — both loaders reproduce frozen sha256 reference
   hashes of the per-frame ``front_img_1`` / ``state_agent_obj`` / ``actions``
   tensors for a fixed fixture episode. These hashes were captured from the
   pre-collapse code (commit at tag ``dedup-c3-pre``). A refactor that changes
   any decoded byte fails here.
2. ``test_cross_loader_per_frame_equality`` — for several fixture episodes,
   the padded full-window read and the packed span read produce
   ``torch.equal`` per-frame tensors (images, state, actions) AND identical
   embodiment ids. This is loader-vs-loader equality independent of any frozen
   constant.
3. ``test_normalization_path_equality`` — applying the same
   ``MultiDataset.normalize`` (per-feature z-score) to BOTH loaders' outputs
   yields ``torch.equal`` normalized tensors. Proves the normalization path is
   identical across loaders.
4. ``test_cu_seqlens_metadata`` — ``pack_collate`` over packed samples emits
   the documented ``seq_lens`` / ``cu_seqlens`` / ``max_seq_len`` /
   ``batch_size`` metadata, and the concatenated per-frame stream matches the
   sum of the individual padded reads.

The suite auto-skips when the fixture dataset is unavailable (off-cluster /
CI), so it degrades gracefully.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes import get_keymap
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset
from egomimic.rldb.zarr.zarr_dataset_packed import (
    ZarrEpisodePackedDataset,
    pack_collate,
)

# --------------------------------------------------------------------------- #
# Fixture dataset
# --------------------------------------------------------------------------- #

DATA_DIR = "/coc/flash7/paphiwetsa3/datasets/new_circle_3"
_HAS_DATA = os.path.isdir(DATA_DIR)
requires_data = pytest.mark.skipif(
    not _HAS_DATA,
    reason=f"requires fixture zarr dataset at {DATA_DIR}",
)

# The single fixture episode the frozen reference hashes were captured from.
_REF_EPISODE = "episode_T_circle_obs0_000000.zarr"

# Frozen sha256(first16) reference hashes of the per-frame decoded tensors for
# _REF_EPISODE, captured from the pre-collapse code at tag dedup-c3-pre on an
# a40 (overcap). They pin the exact decoded bytes; a refactor that perturbs the
# JPEG decode / normalize-by-255 / dtype must fail test_reference_hashes.
_REF_HASHES = {
    "front_img_1": "68f20e3c2c5f72b0|shape=(290, 3, 96, 96)|dtype=float32",
    "state_agent_obj": "26a652f406d275d2|shape=(290, 5)|dtype=float32",
    "actions": "6f7a2b1ab531506b|shape=(290, 2)|dtype=float32",
}

_PER_FRAME_KEYS = ("front_img_1", "state_agent_obj", "actions")


def _tensor_hash(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(a.tobytes()).hexdigest()[:16]
    return f"{digest}|shape={tuple(a.shape)}|dtype={a.dtype}"


def _list_episodes(limit: int | None = None) -> list[Path]:
    eps = sorted(
        p for p in Path(DATA_DIR).iterdir() if p.name.endswith(".zarr")
    )
    return eps if limit is None else eps[:limit]


def _padded_full_episode(ep: Path) -> dict:
    """Read the whole episode as one window from frame 0 via the padded path.

    Using horizon == total_frames means __getitem__ reads exactly [0,
    total_frames) with no padding triggered, so its per-frame output is
    directly comparable to a packed full-episode span read.
    """
    probe = ZarrDataset(ep, key_map=get_keymap(action_horizon=1), transform_list=None)
    T = int(probe.total_frames)
    ds = ZarrDataset(ep, key_map=get_keymap(action_horizon=T), transform_list=None)
    return ds[0]


def _packed_full_episode(ep: Path) -> dict:
    """Read the whole episode as one span via the packed path (_read_span)."""
    ds = ZarrDataset(ep, key_map=get_keymap(action_horizon=1024), transform_list=None)
    return ds._read_span(0, int(ds.total_frames))


# --------------------------------------------------------------------------- #
# 1. Frozen reference hashes (anchored to pre-collapse code)
# --------------------------------------------------------------------------- #


@requires_data
class TestReferenceHashes:
    def test_padded_reference_hashes(self):
        ep = Path(DATA_DIR) / _REF_EPISODE
        if not ep.is_dir():
            pytest.skip(f"reference episode {_REF_EPISODE} not present")
        sample = _padded_full_episode(ep)
        for k in _PER_FRAME_KEYS:
            assert _tensor_hash(sample[k]) == _REF_HASHES[k], (
                f"padded loader hash drift on {k}"
            )

    def test_packed_reference_hashes(self):
        ep = Path(DATA_DIR) / _REF_EPISODE
        if not ep.is_dir():
            pytest.skip(f"reference episode {_REF_EPISODE} not present")
        span = _packed_full_episode(ep)
        for k in _PER_FRAME_KEYS:
            assert _tensor_hash(span[k]) == _REF_HASHES[k], (
                f"packed loader hash drift on {k}"
            )


# --------------------------------------------------------------------------- #
# 2. Cross-loader per-frame equality (loader vs loader)
# --------------------------------------------------------------------------- #


@requires_data
class TestCrossLoaderEquality:
    def test_per_frame_tensors_identical(self):
        eps = _list_episodes(limit=4)
        assert eps, "no fixture episodes found"
        for ep in eps:
            padded = _padded_full_episode(ep)
            span = _packed_full_episode(ep)
            for k in _PER_FRAME_KEYS:
                pe, sp = padded[k], span[k]
                assert pe.shape == sp.shape, (
                    f"{ep.name}:{k} shape mismatch {pe.shape} vs {sp.shape}"
                )
                assert torch.equal(pe, sp), f"{ep.name}:{k} tensors differ"

    def test_embodiment_id_identical(self):
        ep = _list_episodes(limit=1)[0]
        padded = _padded_full_episode(ep)
        span = _packed_full_episode(ep)
        assert int(padded["embodiment"]) == int(span["embodiment"])
        assert int(padded["metadata.robot_name"]) == int(span["metadata.robot_name"])


# --------------------------------------------------------------------------- #
# 3. Normalization-path equality across loaders
# --------------------------------------------------------------------------- #


def _make_norm_holder() -> tuple[MultiDataset, int]:
    """Z-score norm stats for actions + state, keyed on the pushshapes_sim id."""
    emb = get_embodiment_id("pushshapes_sim")
    ms = MultiDataset(state={}, norm_mode="zscore")
    ms.norm_stats[emb] = {
        "actions": {
            "mean": np.array([256.0, 256.0], dtype=np.float32),
            "std": np.array([50.0, 50.0], dtype=np.float32),
        },
        "state_agent_obj": {
            "mean": np.array([256.0, 256.0, 256.0, 0.0, 0.0], dtype=np.float32),
            "std": np.array([50.0, 50.0, 50.0, 1.0, 1.0], dtype=np.float32),
        },
    }
    ms.key_types.setdefault(emb, {})
    ms.zarr_keys.setdefault(emb, {})
    ms.key_types[emb]["actions"] = "action_keys"
    ms.zarr_keys[emb]["actions"] = "actions"
    ms.key_types[emb]["state_agent_obj"] = "proprio_keys"
    ms.zarr_keys[emb]["state_agent_obj"] = "state_agent_obj"
    ms.NORMALIZE_KEY_TYPES = {"action_keys", "proprio_keys"}
    return ms, emb


@requires_data
class TestNormalizationPathEquality:
    def test_normalized_outputs_identical(self):
        ep = _list_episodes(limit=1)[0]
        padded = _padded_full_episode(ep)
        span = _packed_full_episode(ep)
        ms, emb = _make_norm_holder()
        n_padded = ms.normalize(padded, emb)
        n_span = ms.normalize(span, emb)
        for k in ("actions", "state_agent_obj"):
            assert torch.equal(n_padded[k], n_span[k]), (
                f"normalized {k} differs across loaders"
            )
        # And normalization actually changed the values (guards a no-op stats bug).
        assert not torch.equal(n_padded["actions"], padded["actions"])


# --------------------------------------------------------------------------- #
# 4. Packed cu_seqlens / seq_lens metadata
# --------------------------------------------------------------------------- #


@requires_data
class TestPackMetadata:
    def test_cu_seqlens_and_stream(self):
        eps = _list_episodes(limit=3)
        spans = [_packed_full_episode(ep) for ep in eps]
        seq_lens = [int(s["seq_len"]) for s in spans]
        batch = pack_collate([dict(s) for s in spans])

        # Metadata contract.
        assert batch["seq_lens"].tolist() == seq_lens
        expected_cu = [0]
        acc = 0
        for sl in seq_lens:
            acc += sl
            expected_cu.append(acc)
        assert batch["cu_seqlens"].tolist() == expected_cu
        assert int(batch["max_seq_len"]) == max(seq_lens)
        assert int(batch["batch_size"]) == len(eps)

        # Concatenated per-frame stream equals the per-span reads, in order.
        for k in _PER_FRAME_KEYS:
            total = sum(seq_lens)
            assert batch[k].shape[0] == total, k
            off = 0
            for s, sl in zip(spans, seq_lens):
                chunk = batch[k][off : off + sl]
                assert torch.equal(chunk, s[k]), f"stream slice mismatch on {k}"
                off += sl


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
