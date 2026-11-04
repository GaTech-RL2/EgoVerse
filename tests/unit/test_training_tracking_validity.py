"""Missing MANO tracking never reaches opted-in training or norm transforms."""

import copy
import json

import numpy as np
import pytest
import torch
from test_evaluation_tracking_validity import _leaf

from egomimic.rldb.zarr import episode_norm_samples, norm_cache
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

POLICY = "raw_tracking_valid_windows_v1"


def guarded(leaf):
    return MultiDataset(
        datasets={"tracking": leaf}, mode="total", training_tracking_policy=POLICY
    )


@pytest.mark.parametrize(
    "history,stride,horizon", [(1, 1, 1), (4, 1, 3), (4, 3, 2), (2, 2, 7)]
)
@pytest.mark.parametrize("missing", [[0], [4], [11]])
def test_training_support_equals_exact_raw_support(
    tmp_path, history, stride, horizon, missing
):
    leaf = _leaf(
        tmp_path, missing=missing, history=history, stride=stride, horizon=horizon
    )
    expected = np.array(
        [leaf.evaluation_tracking_validity(i) for i in range(len(leaf))]
    )
    ds = guarded(leaf)
    trained = ds.datasets["tracking"]
    np.testing.assert_array_equal(trained._training_tracking_valid_mask, expected)
    np.testing.assert_array_equal(
        trained._training_tracking_valid_indices, np.flatnonzero(expected)
    )
    assert ds.training_tracking_coverage == {
        "tracking": {
            "eligible_starts": 12,
            "valid_starts": int(expected.sum()),
            "rejected_starts": int((~expected).sum()),
        }
    }


def test_guard_precedes_transforms_keeps_batch64_and_episode_weight(tmp_path):
    seen = []

    class AssertTracked:
        def transform(self, data):
            assert all(
                not np.any(np.abs(v) >= 1e8)
                for v in data.values()
                if isinstance(v, np.ndarray)
            )
            seen.append(True)
            return data

    original = _leaf(tmp_path, missing=[4], transform_list=[AssertTracked()])
    ds = guarded(original)
    assert ds.index_map == [("tracking", i) for i in range(12)]
    assert not hasattr(original, "_training_tracking_valid_mask")
    indices = torch.utils.data.Subset(ds, [4] * 64)
    batch = next(
        iter(torch.utils.data.DataLoader(indices, batch_size=64, num_workers=0))
    )
    assert len(batch["frame_idx"]) == 64
    assert len(seen) == 64
    assert len(set(batch["episode_hash"])) == 1
    assert (batch["frame_idx"] != 4).all()
    assert set(batch["frame_idx"].tolist()) <= set(
        ds.datasets["tracking"]._training_tracking_valid_indices
    )
    assert "evaluation_tracking_valid" not in batch


def test_pose_glitch_eligibility_and_original_leaf_are_preserved(tmp_path):
    leaf = _leaf(tmp_path, missing=[4])
    leaf.sample_indices = np.array([0, 1, 4, 9])
    ds = guarded(leaf)
    assert ds.index_map == [("tracking", int(i)) for i in leaf.sample_indices]
    assert ds.training_tracking_coverage["tracking"] == {
        "eligible_starts": 4,
        "valid_starts": 3,
        "rejected_starts": 1,
    }
    assert set(ds.datasets["tracking"]._training_tracking_valid_indices) == {0, 1, 9}
    assert set(int(ds[2]["frame_idx"]) for _ in range(20)) <= {0, 1, 9}
    np.testing.assert_array_equal(leaf.sample_indices, [0, 1, 4, 9])


def test_selected_episode_without_valid_start_fails_before_transforms(tmp_path):
    class Never:
        def transform(self, data):
            pytest.fail("Invalid sample reached transforms")

    leaf = _leaf(tmp_path, missing=range(12), transform_list=[Never()])
    with pytest.raises(ValueError, match="no tracking-valid training windows"):
        guarded(leaf)


def test_empty_existing_eligibility_fails_closed(tmp_path):
    leaf = _leaf(tmp_path)
    leaf.sample_indices = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="no tracking-valid training windows"):
        guarded(leaf)


def test_malformed_tracking_fails_closed_during_preparation(tmp_path):
    leaf = _leaf(tmp_path, malformed=(np.full(7, 1e9), np.ones(63)))
    with pytest.raises(ValueError, match="nonzero keypoints"):
        guarded(leaf)


def test_norm_current_step_keymap_inherits_real_training_history(tmp_path):
    leaf = _leaf(tmp_path, missing=[4], history=4, horizon=2)
    train = guarded(leaf)
    norm_leaf = copy.copy(leaf)
    norm_leaf.key_map = copy.deepcopy(leaf.key_map)
    for spec in norm_leaf.key_map.values():
        if "history" in spec:
            spec["history"] = 1
    norm = guarded(norm_leaf)
    assert norm.datasets["tracking"]._training_tracking_valid_mask[6]
    assert not train.datasets["tracking"]._training_tracking_valid_mask[6]
    norm.use_training_tracking_reference(train)
    assert not norm.datasets["tracking"]._training_tracking_valid_mask[6]
    assert norm.training_tracking_coverage == train.training_tracking_coverage
    assert norm.datasets["tracking"].key_map["left.history_wrist_pose"]["history"] == 1
    sample = norm[6]
    assert int(sample["frame_idx"]) != 6
    assert sample["left.history_wrist_pose"].shape == (7,)


def test_normalization_uses_guarded_rows_and_rejects_old_stats(tmp_path, monkeypatch):
    leaf = _leaf(tmp_path, missing=[4])
    ds = guarded(leaf)
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"human_bimanual": ds})

    def forbidden_cache(*args, **kwargs):
        pytest.fail("Guarded normalization read historical per-episode cached rows")

    monkeypatch.setattr(episode_norm_samples, "collect", forbidden_cache)
    stats.infer_norm_from_dataset(
        ds,
        "human_bimanual",
        sample_frac=1,
        max_samples=12,
        batch_size=4,
        num_workers=0,
        episode_cache=tmp_path / "old-rows",
    )
    assert stats._norm_run_metadata["training_tracking_policy"] == POLICY
    assert (
        stats._norm_run_metadata["tracking_coverage"] == ds.training_tracking_coverage
    )
    for values in next(iter(stats.norm_stats.values())).values():
        assert all(np.abs(value).max() < 10 for value in values.values())
    stats.cache_stats(str(tmp_path / "new"))
    path = tmp_path / "new/norm_stats/norm_stats.json"
    payload = json.loads(path.read_text())
    assert payload["norm_run_metadata"]["training_tracking_policy"] == POLICY
    stats.infer_norm_from_dataset(ds, "human_bimanual", precomputed_norm_path=str(path))
    payload.pop("norm_run_metadata")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="required training tracking policy"):
        stats.infer_norm_from_dataset(
            ds, "human_bimanual", precomputed_norm_path=str(path)
        )


def test_tracking_policy_changes_both_norm_cache_identities(monkeypatch):
    monkeypatch.setattr(norm_cache, "code_hash", lambda recipe: "fixed-source")
    cfg = {"resolver": {"key_map": {"history": 4}}}
    enabled = {**cfg, "training_tracking_policy": POLICY}
    assert episode_norm_samples.recipe_key(
        "human_bimanual", cfg
    ) != episode_norm_samples.recipe_key("human_bimanual", enabled)
    old = norm_cache.cache_inputs(
        "human_bimanual", {"ep": "fingerprint"}, cfg, 1, False, 200000
    )
    new = norm_cache.cache_inputs(
        "human_bimanual", {"ep": "fingerprint"}, enabled, 1, False, 200000, seed=42026
    )
    assert norm_cache.norm_cache_key(old) != norm_cache.norm_cache_key(new)


def test_all_tracked_numeric_outputs_unchanged(tmp_path):
    leaf = _leaf(tmp_path)
    ds = guarded(leaf)
    for i in (0, 4, 11):
        before, after = leaf[i], ds[i]
        assert before.keys() == after.keys()
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    after[key], value, rtol=0, atol=0, equal_nan=True
                )
            else:
                assert after[key] == value


def test_unknown_training_policy_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown training tracking policy"):
        MultiDataset(
            datasets={"tracking": _leaf(tmp_path)},
            mode="total",
            training_tracking_policy="typo",
        )


def test_decoder_fallback_is_limited_to_tracking_valid_starts(tmp_path, monkeypatch):
    from egomimic.rldb.zarr import zarr_dataset_multi as module

    leaf = _leaf(tmp_path, missing=[4])
    # Reuse a numeric array as a fake camera input; decoding itself is mocked.
    leaf.key_map = {
        key: spec for key, spec in leaf.key_map.items() if "keypoints" not in key
    }
    leaf.key_map["front"] = {
        "zarr_key": "left.obs_keypoints",
        "key_type": "camera_keys",
    }
    leaf._image_keys = {"left.obs_keypoints"}
    ds = guarded(leaf)
    calls = []

    def decode(*args, **kwargs):
        if not calls:
            calls.append("failed")
            raise ValueError("synthetic JPEG failure")
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def fallback(**kwargs):
        assert set(kwargs["candidates"]) == set(
            ds.datasets["tracking"]._training_tracking_valid_indices
        )
        calls.append("fallback")
        return 9, 1

    monkeypatch.setattr(module.simplejpeg, "decode_jpeg", decode)
    monkeypatch.setattr(module, "get_fallback_idx", fallback)
    assert int(ds[0]["frame_idx"]) == 9
    assert calls == ["failed", "fallback"]


def test_annotation_cutoff_and_lagged_history_use_actual_support(tmp_path):
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

    class Cutoff(ZarrDataset):
        def _chunk_end_idx(self, idx, horizon, key_type):
            return min(idx + horizon, 5 if idx < 5 else self.total_frames)

    leaf = _leaf(tmp_path, missing=[5], history=3, stride=2, horizon=4)
    leaf.__class__ = Cutoff
    for spec in leaf.key_map.values():
        if "history" in spec:
            spec["lag_s"] = 0.1
    expected = [leaf.evaluation_tracking_validity(i) for i in range(len(leaf))]
    ds = guarded(leaf)
    np.testing.assert_array_equal(
        ds.datasets["tracking"]._training_tracking_valid_mask, expected
    )
    assert expected[4]  # Future read stops before missing row5, per actual EOS.


def test_guarded_norm_seed_is_required_in_cache_identity(monkeypatch):
    monkeypatch.setattr(norm_cache, "code_hash", lambda recipe: "fixed-source")
    cfg = {"resolver": {}, "training_tracking_policy": POLICY}
    with pytest.raises(ValueError, match="requires the sampling seed"):
        norm_cache.cache_inputs("human_bimanual", {}, cfg, 1)
    one = norm_cache.cache_inputs("human_bimanual", {}, cfg, 1, seed=42026)
    two = norm_cache.cache_inputs("human_bimanual", {}, cfg, 1, seed=42)
    assert one["sampling_seed"] == 42026
    assert norm_cache.norm_cache_key(one) != norm_cache.norm_cache_key(two)


def test_precomputed_norm_wrong_seed_or_coverage_rejected(tmp_path):
    ds = guarded(_leaf(tmp_path, missing=[4]))
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"human_bimanual": ds})
    stats.infer_norm_from_dataset(
        ds, "human_bimanual", sample_frac=1, num_workers=0, seed=42026
    )
    stats.cache_stats(str(tmp_path / "norm"))
    path = tmp_path / "norm/norm_stats/norm_stats.json"
    payload = json.loads(path.read_text())
    assert payload["norm_run_metadata"]["sampling_seed"] == 42026
    with pytest.raises(ValueError, match="sampling seed"):
        stats.infer_norm_from_dataset(
            ds, "human_bimanual", precomputed_norm_path=str(path), seed=42
        )
    payload["norm_run_metadata"]["tracking_coverage"]["tracking"]["valid_starts"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="tracking coverage"):
        stats.infer_norm_from_dataset(
            ds, "human_bimanual", precomputed_norm_path=str(path), seed=42026
        )


def test_zero_worker_guarded_norm_uses_local_seed_and_restores_callers(
    tmp_path, monkeypatch
):
    import random

    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

    ds = guarded(_leaf(tmp_path, missing=[4]))
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"human_bimanual": ds})
    selected = []
    original = ZarrDataset.__getitem__

    def read(leaf, *args, **kwargs):
        sample = original(leaf, *args, **kwargs)
        selected.append(int(sample["frame_idx"]))
        return sample

    monkeypatch.setattr(ZarrDataset, "__getitem__", read)
    sequences = []
    for caller_seed in (123, 987):
        random.seed(caller_seed)
        np.random.seed(caller_seed)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        selected.clear()
        stats.infer_norm_from_dataset(
            ds,
            "human_bimanual",
            sample_frac=1,
            max_samples=12,
            batch_size=4,
            num_workers=0,
            seed=42026,
        )
        sequences.append(list(selected))
        assert random.getstate() == python_before
        after = np.random.get_state()
        assert after[0] == numpy_before[0]
        np.testing.assert_array_equal(after[1], numpy_before[1])
        assert after[2:] == numpy_before[2:]
        assert stats._norm_run_metadata["sampling_seed"] == 42026
    assert len(sequences[0]) == 12
    assert sequences[0] == sequences[1]
    assert set(sequences[0]) <= set(
        ds.datasets["tracking"]._training_tracking_valid_indices
    )


def test_guarded_norm_restores_random_streams_when_sampling_fails(
    tmp_path, monkeypatch
):
    import random

    ds = guarded(_leaf(tmp_path, missing=[4]))
    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"human_bimanual": ds})
    python_before = random.getstate()
    numpy_before = np.random.get_state()

    def fail(*args, **kwargs):
        random.random()
        np.random.rand()
        raise RuntimeError("synthetic sampler failure")

    monkeypatch.setattr(stats, "_collect_norm_samples", fail)
    with pytest.raises(RuntimeError, match="synthetic sampler failure"):
        stats.infer_norm_from_dataset(ds, "human_bimanual", num_workers=0, seed=42026)
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert after[0] == numpy_before[0]
    np.testing.assert_array_equal(after[1], numpy_before[1])
    assert after[2:] == numpy_before[2:]
