"""
Unit tests for the packed-mode data + normalization plumbing fixes.

Covers the four issues discovered while wiring up packed training:

  1. ``MultiDataset.normalize`` is shape-agnostic — per-feature ``(D,)`` stats
     broadcast against both padded ``(B, T, D)`` and packed ``(T_total, D)``
     tensors. Round-trip with ``unnormalize`` is identity.
  2. ``MultiDataset._iter_leaves`` descends into
     ``ZarrEpisodePackedDataset.datasets`` so ``populate_from_datasets`` sees
     the inner ``ZarrDataset`` leaves (and reads their ``embodiment`` /
     ``key_map``). Without this fix the packed dataset was skipped silently
     and ``norm_stats.key_types`` came out empty.
  3. ``ZarrDataset.__getitem__`` loops per-frame when ``horizon > 1`` on an
     image key — passing the whole ``(horizon,)`` array of JPEG buffers to
     ``simplejpeg.decode_jpeg`` in one call crashed with a dtype error and
     caused the bounded resample loop to hang for ~400 seconds.
  4. ``MultiDataset.infer_norm_from_dataset`` uses ``pack_collate`` for
     packed datasets (``default_collate`` would torch.stack variable-length
     episode tensors and crash). ``sample_frac`` is reinterpreted as a frame
     count, not an episode count.

Tests 3 and 4 are integration-style and use the live pushshapes dataset at
``/coc/cedarp-dxu345-0/Tsim_datasets2/circle/``; they auto-skip when that
path is unavailable (CI / local dev machines).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from egomimic.rldb.embodiment.pushshapes import get_keymap
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset
from egomimic.rldb.zarr.zarr_dataset_packed import (
    ZarrEpisodePackedDataset,
)

PUSHT_FOLDER = "/coc/cedarp-dxu345-0/Tsim_datasets2/circle/basic"
_HAS_PUSHT = os.path.isdir(PUSHT_FOLDER)
requires_pusht = pytest.mark.skipif(
    not _HAS_PUSHT,
    reason=f"requires pushshapes dataset at {PUSHT_FOLDER}",
)


# --------------------------------------------------------------------------- #
# Issue 1 — normalize on padded vs packed shape
# --------------------------------------------------------------------------- #


class TestNormalizeShapeAgnostic:
    def _make_stats_holder(self, mean_v, std_v):
        ms = MultiDataset(state={}, norm_mode="zscore")
        emb = 0
        ms.norm_stats[emb] = {}
        ms.key_types.setdefault(emb, {})
        ms.zarr_keys.setdefault(emb, {})
        ms.NORMALIZE_KEY_TYPES = {"action_keys"}
        ms.key_types[emb]["actions"] = "action_keys"
        ms.zarr_keys[emb]["actions"] = "actions"
        ms.norm_stats[emb]["actions"] = {
            "mean": np.array(mean_v, dtype=np.float32),
            "std": np.array(std_v, dtype=np.float32),
        }
        return ms, emb

    def test_padded_per_feature_broadcast(self):
        ms, emb = self._make_stats_holder([10.0, 20.0], [2.0, 4.0])
        # Mean shifted by exactly one std along each feature.
        padded = {"actions": torch.zeros(2, 5, 2)}
        padded["actions"][..., 0] = 12.0
        padded["actions"][..., 1] = 24.0
        out = ms.normalize(padded, emb)
        assert torch.allclose(
            out["actions"], torch.ones_like(padded["actions"]), atol=1e-5
        )

    def test_packed_per_feature_broadcast(self):
        ms, emb = self._make_stats_holder([10.0, 20.0], [2.0, 4.0])
        packed = {"actions": torch.zeros(10, 2)}
        packed["actions"][:, 0] = 12.0
        packed["actions"][:, 1] = 24.0
        out = ms.normalize(packed, emb)
        assert torch.allclose(
            out["actions"], torch.ones_like(packed["actions"]), atol=1e-5
        )

    def test_round_trip_identity_packed(self):
        ms, emb = self._make_stats_holder([10.0, 20.0], [2.0, 4.0])
        torch.manual_seed(0)
        x = torch.randn(20, 2) * 5 + 10
        normed = ms.normalize({"actions": x}, emb)
        back = ms.unnormalize(normed, emb)
        assert torch.allclose(back["actions"], x, atol=1e-5)


# --------------------------------------------------------------------------- #
# Issue 2 — _iter_leaves descends into ZarrEpisodePackedDataset
# --------------------------------------------------------------------------- #


class TestIterLeavesPacked:
    def test_iter_leaves_descends_into_packed_wrapper(self):
        # Make a fake packed dataset with two fake ZarrDataset leaves; verify
        # _iter_leaves yields exactly those two.
        class _FakeLeaf:
            def __init__(self, eb, km):
                self.embodiment = eb
                self.key_map = km

        leaves = {
            "a": _FakeLeaf("pushshapes_sim", {"x": {"zarr_key": "x"}}),
            "b": _FakeLeaf("pushshapes_sim", {"x": {"zarr_key": "x"}}),
        }
        fake_packed = mock.MagicMock(spec=ZarrEpisodePackedDataset)
        fake_packed.datasets = leaves

        out = list(MultiDataset._iter_leaves(fake_packed))
        assert len(out) == 2
        assert set(o.embodiment for o in out) == {"pushshapes_sim"}


# --------------------------------------------------------------------------- #
# Issue 3 — __getitem__ multi-frame JPEG decode
# --------------------------------------------------------------------------- #


@requires_pusht
class TestZarrGetItemImageHorizon:
    """Verify ZarrDataset.__getitem__ correctly decodes per-frame for
    horizon>1 on an image key. Pre-fix, passing the whole ``(horizon,)``
    array of JPEG buffers to ``simplejpeg.decode_jpeg`` crashed with a dtype
    error, and the bounded resample loop then hung for ~400 seconds.
    """

    def _pick_episode(self):
        for p in sorted(Path(PUSHT_FOLDER).iterdir()):
            if p.name.endswith(".zarr"):
                return p
        pytest.skip("No .zarr episodes in pushshapes dir")

    def test_horizon_n_image_decode(self):
        """horizon=4 image read must produce (4, 3, H, W) float32 in [0,1]."""
        H = 4
        key_map = {
            "front_img_1": {
                "zarr_key": "observations.images.front_img_1",
                "key_type": "camera_keys",
                "horizon": H,
            },
            "state_agent_obj": {
                "zarr_key": "observations.state",
                "key_type": "proprio_keys",
                "horizon": H,
            },
            "actions": {
                "zarr_key": "actions",
                "key_type": "action_keys",
                "horizon": H,
            },
        }
        ds = ZarrDataset(self._pick_episode(), key_map=key_map, transform_list=None)
        sample = ds[0]
        img = sample["front_img_1"]
        assert img.shape[0] == H, img.shape
        assert (
            img.dim() == 4 and img.shape[1] == 3
        ), f"expected (H, 3, H, W), got {img.shape}"
        assert img.dtype == torch.float32
        assert img.max().item() <= 1.0
        assert img.min().item() >= 0.0

    def test_horizon_1_image_still_works(self):
        """Regression: single-frame image read (no horizon) still works."""
        key_map = {
            "front_img_1": {
                "zarr_key": "observations.images.front_img_1",
                "key_type": "camera_keys",  # no horizon
            },
            "actions": {
                "zarr_key": "actions",
                "key_type": "action_keys",
                "horizon": 2,
            },
        }
        ds = ZarrDataset(self._pick_episode(), key_map=key_map, transform_list=None)
        sample = ds[0]
        img = sample["front_img_1"]
        # No horizon on image key → single frame (3, H, W).
        assert img.dim() == 3 and img.shape[0] == 3, img.shape

    def test_decode_called_once_per_frame(self):
        """Verify the per-frame loop actually runs (no batched call)."""
        H = 5
        key_map = {
            "front_img_1": {
                "zarr_key": "observations.images.front_img_1",
                "key_type": "camera_keys",
                "horizon": H,
            },
            "actions": {
                "zarr_key": "actions",
                "key_type": "action_keys",
                "horizon": H,
            },
        }
        ds = ZarrDataset(self._pick_episode(), key_map=key_map, transform_list=None)
        import simplejpeg as _sjpg

        real_decode = _sjpg.decode_jpeg
        with mock.patch.object(
            _sjpg,
            "decode_jpeg",
            wraps=real_decode,
        ) as decode_spy:
            _ = ds[0]
        assert (
            decode_spy.call_count == H
        ), f"expected {H} per-frame decode calls, got {decode_spy.call_count}"


# --------------------------------------------------------------------------- #
# Issue 4 — infer_norm_from_dataset uses pack_collate for packed datasets
# --------------------------------------------------------------------------- #


@requires_pusht
class TestInferNormFromPacked:
    """Integration test on the live pushshapes dataset:
    (a) ``populate_from_datasets`` finds the inner ZarrDataset leaves
        and registers their key_types (issue 2).
    (b) ``infer_norm_from_dataset`` uses ``pack_collate`` so it doesn't
        crash on variable-length episode tensors (issue 4).
    (c) Per-feature stats are produced for BOTH ``actions`` and
        ``state_agent_obj`` (would only have ``actions`` pre-fix because
        the pipeline was crashing silently after collecting the first key).
    """

    def test_full_pipeline_collects_per_feature_stats(self):
        ds = ZarrEpisodePackedDataset.from_local_folder(
            folder_path=PUSHT_FOLDER,
            key_map=get_keymap(action_horizon=1024),
            max_seq_len=None,
            chunking="none",
            min_seq_len=64,
            transform_list=None,
        )
        assert len(ds) > 0

        ns = MultiDataset(state={}, norm_mode="quantile")
        ns.populate_from_datasets({"pushshapes_sim": ds})
        # Issue 2: _iter_leaves descent populates key_types via inner leaves.
        assert 15 in ns.key_types, ns.key_types
        kts = ns.key_types[15]
        assert kts.get("actions") == "action_keys"
        assert kts.get("state_agent_obj") == "proprio_keys"
        assert kts.get("front_img_1") == "camera_keys"

        ns.infer_shapes_from_batch(ds[0])
        # Small sample_frac so the test stays under a few seconds.
        ns.infer_norm_from_dataset(
            ds,
            "pushshapes_sim",
            sample_frac=0.02,
            num_workers=0,
            batch_size=4,
        )

        stats = ns.norm_stats[15]
        # Both action_keys and proprio_keys must have stats after the fix.
        assert "actions" in stats, stats.keys()
        assert "state_agent_obj" in stats, stats.keys()

        # Per-feature shape: actions is 2-D, state_agent_obj is 5-D.
        assert stats["actions"]["mean"].shape == (2,)
        assert stats["actions"]["std"].shape == (2,)
        assert stats["state_agent_obj"]["mean"].shape == (5,)
        assert stats["state_agent_obj"]["std"].shape == (5,)

        # Stats should be in plausible pixel-coord ranges (0-512 world).
        for v in stats["actions"]["mean"]:
            assert 0.0 < v < 512.0, v
        # Std non-zero (otherwise normalize would divide-by-near-zero).
        assert (stats["actions"]["std"] > 1.0).all()
        assert (stats["state_agent_obj"]["std"] > 0.5).all()


class TestNormStatsAcceptsPackedShape:
    """A focused unit test that normalize/unnormalize used with the *stats*
    a real run produces handles packed (T_total, D) tensors. Catches future
    regressions that might break broadcasting (e.g. someone adding a
    ``.view(B, T, D)`` somewhere)."""

    def test_normalize_on_packed_two_dim(self):
        ms = MultiDataset(state={}, norm_mode="zscore")
        emb = 0
        ms.norm_stats[emb] = {}
        ms.key_types.setdefault(emb, {})
        ms.zarr_keys.setdefault(emb, {})
        ms.NORMALIZE_KEY_TYPES = {"action_keys", "proprio_keys"}
        ms.key_types[emb]["actions"] = "action_keys"
        ms.zarr_keys[emb]["actions"] = "actions"
        ms.key_types[emb]["state"] = "proprio_keys"
        ms.zarr_keys[emb]["state"] = "state"
        ms.norm_stats[emb] = {
            "actions": {
                "mean": np.array([100.0, 200.0], dtype=np.float32),
                "std": np.array([10.0, 20.0], dtype=np.float32),
            },
            "state": {
                "mean": np.array([0.0, 5.0, 10.0], dtype=np.float32),
                "std": np.array([1.0, 2.0, 5.0], dtype=np.float32),
            },
        }
        # Packed (T_total, D) for both keys.
        T = 17
        data = {
            "actions": torch.full(
                (T, 2), 110.0
            ),  # 1 std above mean dim 0; 0.5 std above dim 1
            "state": torch.full((T, 3), 1.0),
        }
        data["actions"][:, 1] = 210.0
        out = ms.normalize(data, emb)
        # Expect actions[:, 0] = 1.0, actions[:, 1] = 0.5
        assert torch.allclose(out["actions"][:, 0], torch.ones(T), atol=1e-5)
        assert torch.allclose(out["actions"][:, 1], 0.5 * torch.ones(T), atol=1e-5)
        # state: (1.0 - mean) / std broadcast across T.
        expected_state = (
            torch.ones(T, 3) - torch.tensor([0.0, 5.0, 10.0])
        ) / torch.tensor([1.0, 2.0, 5.0])
        assert torch.allclose(out["state"], expected_state, atol=1e-5)


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
