"""Tests for the annotation-override hook (LLM-relabeled segments at load time).

The override lets a data config swap the in-zarr ``annotations`` for a sidecar
relabel map without rewriting any zarr store. These tests cover the three pieces:
the resolver-side JSON loader, the resolver __init__ threading, and the
``ZarrDataset`` short-circuit that feeds the per-frame language prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from egomimic.rldb.zarr.zarr_dataset_multi import (
    EpisodeResolver,
    ZarrDataset,
    _load_annotation_overrides,
)

# A known mecka folding_clothes episode on the shared cluster store (integration test).
_SHARED_DIR = Path("/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets")
_SAMPLE_HASH = "696d0d524b2b73eea2f1cc51"


# ---------------------------------------------------------------------------
# _load_annotation_overrides
# ---------------------------------------------------------------------------
def test_load_annotation_overrides_none_returns_empty() -> None:
    assert _load_annotation_overrides(None) == {}
    assert _load_annotation_overrides("") == {}


def test_load_annotation_overrides_reads_map(tmp_path) -> None:
    p = tmp_path / "relabel_map.json"
    payload = {"ep1": [{"start_idx": 0, "end_idx": 10, "text": "fold_tshirt"}]}
    p.write_text(json.dumps(payload))

    assert _load_annotation_overrides(p) == payload


def test_load_annotation_overrides_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_annotation_overrides(tmp_path / "nope.json")


def test_load_annotation_overrides_rejects_non_dict(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(["not", "a", "map"]))
    with pytest.raises(ValueError, match="must be a JSON object"):
        _load_annotation_overrides(p)


# ---------------------------------------------------------------------------
# resolver threads the path into self.annotation_overrides
# ---------------------------------------------------------------------------
def test_resolver_init_loads_overrides(tmp_path) -> None:
    p = tmp_path / "relabel_map.json"
    payload = {"epA": [{"start_idx": 0, "end_idx": 5, "text": "smooth_tshirt"}]}
    p.write_text(json.dumps(payload))

    resolver = EpisodeResolver(tmp_path, annotation_override_path=p)
    assert resolver.annotation_overrides == payload

    # No override configured -> empty dict (so .get(name) is always safe).
    assert EpisodeResolver(tmp_path).annotation_overrides == {}


# ---------------------------------------------------------------------------
# ZarrDataset override short-circuit (isolated, no zarr store needed)
# ---------------------------------------------------------------------------
def _dataset_with_override(override):
    """A ZarrDataset with only the attributes the annotation path touches."""
    ds = ZarrDataset.__new__(ZarrDataset)
    ds._annotations = None
    ds._annotation_override = override
    return ds


def test_load_annotations_uses_override() -> None:
    spans = [
        {"start_idx": 0, "end_idx": 100, "text": "fold_tshirt"},
        {"start_idx": 100, "end_idx": 200, "text": "place_on_stack"},
    ]
    ds = _dataset_with_override(spans)

    assert ds._load_annotations() == spans
    # Per-frame prompt resolves to the override label, span-aware.
    assert ds._annotation_text_for_frame(50) == ["fold_tshirt"]
    assert ds._annotation_text_for_frame(150) == ["place_on_stack"]
    # Frame outside any span -> no annotation (model falls back to default prompt).
    assert ds._annotation_text_for_frame(999) == []


def test_load_annotations_override_filters_non_dict() -> None:
    ds = _dataset_with_override(
        [{"start_idx": 0, "end_idx": 10, "text": "fold_tshirt"}, "garbage", None]
    )
    assert ds._load_annotations() == [
        {"start_idx": 0, "end_idx": 10, "text": "fold_tshirt"}
    ]


def test_no_override_does_not_short_circuit() -> None:
    # With override=None the method must fall through to the (unset) store reader,
    # proving the override branch is gated on `is not None` rather than truthiness.
    ds = _dataset_with_override(None)
    with pytest.raises(AttributeError):
        ds._load_annotations()


# ---------------------------------------------------------------------------
# Integration: real episode + override flows through the actual load path
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (_SHARED_DIR / _SAMPLE_HASH).is_dir(),
    reason="shared cluster zarr store not available",
)
def test_override_on_real_episode_changes_prompt() -> None:
    from egomimic.rldb.embodiment.human import Mecka

    key_map = Mecka.get_keymap(mode="cartesian_pi", annotation_key="annotations")
    path = _SHARED_DIR / _SAMPLE_HASH

    baseline = ZarrDataset(path, key_map=key_map)
    mid = len(baseline) // 2
    original = baseline._annotation_text_for_frame(mid)
    assert original, "expected the real episode to have annotation text at mid-frame"

    override = [{"start_idx": 0, "end_idx": len(baseline), "text": "UNIT_TEST_LABEL"}]
    relabeled = ZarrDataset(path, key_map=key_map, annotation_override=override)
    assert relabeled._annotation_text_for_frame(mid) == ["UNIT_TEST_LABEL"]
    assert relabeled._annotation_text_for_frame(mid) != original
