import json
from pathlib import Path

import numpy as np
import pytest
import torch

import egomimic.rldb.zarr.zarr_dataset_multi as zarr_dataset_multi
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

STAT_FUNCTIONS = {
    "mean": lambda value, axis: np.mean(value, axis=axis),
    "std": lambda value, axis: np.std(value, axis=axis),
    "min": lambda value, axis: np.min(value, axis=axis),
    "max": lambda value, axis: np.max(value, axis=axis),
    "median": lambda value, axis: np.median(value, axis=axis),
    "quantile_1": lambda value, axis: np.percentile(value, 1, axis=axis),
    "quantile_99": lambda value, axis: np.percentile(value, 99, axis=axis),
    "quantile_0_01": lambda value, axis: np.percentile(value, 0.01, axis=axis),
    "quantile_99_99": lambda value, axis: np.percentile(value, 99.99, axis=axis),
}


@pytest.mark.parametrize(
    ("reduce_all_but_last", "axis", "expected_shape"),
    [(False, 0, (3, 4)), (True, (0, 1), (4,))],
)
def test_norm_stats_reduction_shape_and_values(
    reduce_all_but_last, axis, expected_shape
):
    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    norm_stats = MultiDataset(state={}, reduce_all_but_last=reduce_all_but_last)

    actual = norm_stats._compute_stats_for_array(values)

    assert set(actual) == set(STAT_FUNCTIONS)
    for stat_name, expected_fn in STAT_FUNCTIONS.items():
        assert actual[stat_name].shape == expected_shape
        np.testing.assert_allclose(
            actual[stat_name], expected_fn(values, axis), rtol=1e-6, atol=1e-6
        )


def _usocket_norm_stats(reduce_all_but_last=True):
    embodiment = 19
    norm_stats = MultiDataset(
        state={},
        norm_mode="minmax",
        reduce_all_but_last=reduce_all_but_last,
    )
    norm_stats.embodiments = {embodiment}
    norm_stats.key_types = {
        embodiment: {
            "actions": "action_keys",
            "state_agent_obj": "proprio_keys",
        }
    }
    norm_stats.zarr_keys = {
        embodiment: {
            "actions": "actions",
            "state_agent_obj": "state_agent_obj",
        }
    }
    norm_stats.shapes = {embodiment: {"actions": (16, 4), "state_agent_obj": (3,)}}
    norm_stats.norm_stats = {
        embodiment: {
            "actions": {
                "min": np.array([-2.0, -4.0, -1.0, -1.0], dtype=np.float32),
                "max": np.array([2.0, 4.0, 1.0, 1.0], dtype=np.float32),
            },
            "state_agent_obj": {
                "min": np.array([-2.0, -3.0, -4.0], dtype=np.float32),
                "max": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            },
        }
    }
    return norm_stats


@pytest.mark.parametrize("batch_size", [None, 2])
def test_reduced_usocket_stats_broadcast_and_roundtrip(batch_size):
    norm_stats = _usocket_norm_stats()
    actions = torch.tensor(
        [[-1.0, -2.0, -0.5, 0.5], [1.0, 2.0, 0.5, -0.5]],
        dtype=torch.float32,
    ).repeat(8, 1)
    state = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    if batch_size is not None:
        actions = actions.unsqueeze(0).repeat(batch_size, 1, 1)
        state = state.unsqueeze(0).repeat(batch_size, 1)
    raw = {"actions": actions, "state_agent_obj": state}

    normalized = norm_stats.normalize(raw, 19)
    restored = norm_stats.unnormalize(normalized, 19)

    assert normalized["actions"].shape == actions.shape
    assert normalized["state_agent_obj"].shape == state.shape
    torch.testing.assert_close(restored["actions"], actions)
    torch.testing.assert_close(restored["state_agent_obj"], state)


def test_reduction_flag_survives_state_and_dataset_propagation():
    source = _usocket_norm_stats()

    restored = MultiDataset.from_state(source.to_state())
    target = MultiDataset(state={})
    target.set_norm_stats_from(source)

    assert restored.reduce_all_but_last is True
    assert target.reduce_all_but_last is True
    assert restored.norm_stats[19]["actions"]["min"].shape == (4,)


def _action_only_norm_stats(reduce_all_but_last, stat_shape):
    norm_stats = MultiDataset(
        state={},
        norm_mode="minmax",
        reduce_all_but_last=reduce_all_but_last,
    )
    norm_stats.embodiments = {19}
    norm_stats.key_types = {19: {"actions": "action_keys"}}
    norm_stats.zarr_keys = {19: {"actions": "actions"}}
    norm_stats.shapes = {19: {"actions": (16, 4)}}
    norm_stats.norm_stats = {
        19: {
            "actions": {
                "min": np.zeros(stat_shape, dtype=np.float32),
                "max": np.ones(stat_shape, dtype=np.float32),
            }
        }
    }
    return norm_stats


def test_precomputed_stats_require_matching_reduction_semantics(tmp_path):
    slotwise = _action_only_norm_stats(False, (16, 4))
    slotwise.cache_stats(str(tmp_path))
    cache_path = tmp_path / "norm_stats" / "norm_stats.json"
    reduced = _action_only_norm_stats(True, (4,))

    with pytest.raises(ValueError, match="was computed with reduce_all_but_last=False"):
        reduced.infer_norm_from_dataset(
            object(), 19, precomputed_norm_path=str(cache_path)
        )

    legacy_payload = json.loads(cache_path.read_text())
    legacy_payload.pop("reduce_all_but_last")
    cache_path.write_text(json.dumps(legacy_payload))
    with pytest.raises(ValueError, match=r"has shape \(16, 4\).+expected \(4,\)"):
        reduced.infer_norm_from_dataset(
            object(), 19, precomputed_norm_path=str(cache_path)
        )


def test_matching_reduced_precomputed_stats_load_as_float32(tmp_path):
    source = _action_only_norm_stats(True, (4,))
    source.cache_stats(str(tmp_path))
    cache_path = tmp_path / "norm_stats" / "norm_stats.json"
    target = _action_only_norm_stats(True, (4,))
    target.norm_stats = {19: {}}

    target.infer_norm_from_dataset(object(), 19, precomputed_norm_path=str(cache_path))

    assert target.norm_stats[19]["actions"]["min"].shape == (4,)
    assert target.norm_stats[19]["actions"]["min"].dtype == np.float32


def test_precomputed_stats_reject_missing_required_key_and_mode(tmp_path):
    source = _action_only_norm_stats(True, (4,))
    source.cache_stats(str(tmp_path))
    cache_path = tmp_path / "norm_stats" / "norm_stats.json"
    payload = json.loads(cache_path.read_text())

    payload["stats"]["19"].pop("actions")
    cache_path.write_text(json.dumps(payload))
    target = _action_only_norm_stats(True, (4,))
    target.norm_stats = {19: {}}
    with pytest.raises(ValueError, match="missing required keys"):
        target.infer_norm_from_dataset(
            object(), 19, precomputed_norm_path=str(cache_path)
        )

    payload["stats"]["19"]["actions"] = {
        "min": [0.0] * 4,
        "max": [1.0] * 4,
    }
    payload["norm_mode"] = "zscore"
    cache_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="norm_mode='zscore'"):
        target.infer_norm_from_dataset(
            object(), 19, precomputed_norm_path=str(cache_path)
        )


@pytest.mark.parametrize("as_directory", [False, True])
def test_configured_precomputed_stats_path_fails_closed(tmp_path, as_directory):
    target = _action_only_norm_stats(True, (4,))
    target.norm_stats = {19: {}}
    configured = tmp_path / "missing"
    if as_directory:
        configured.mkdir()

    with pytest.raises(FileNotFoundError, match="precomputed|norm_stats.json"):
        target.infer_norm_from_dataset(
            object(), 19, precomputed_norm_path=str(configured)
        )

    assert target.norm_stats == {19: {}}


class _ActionDataset(torch.utils.data.Dataset):
    def __init__(self, size, width):
        self.size = size
        self.width = width

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "actions": torch.full(
                (2, self.width), float(index), dtype=torch.float32
            )
        }


def _two_domain_action_stats():
    norm_stats = MultiDataset(
        state={}, norm_mode="minmax", reduce_all_but_last=True
    )
    norm_stats.embodiments = {19, 20}
    norm_stats.key_types = {
        19: {"actions": "action_keys"},
        20: {"actions": "action_keys"},
    }
    norm_stats.zarr_keys = {
        19: {"actions": "actions"},
        20: {"actions": "actions"},
    }
    norm_stats.shapes = {
        19: {"actions": (2, 4)},
        20: {"actions": (2, 6)},
    }
    norm_stats.norm_stats = {19: {}, 20: {}}
    return norm_stats


def test_shape_inference_is_scoped_to_the_named_embodiment():
    norm_stats = _two_domain_action_stats()
    norm_stats.shapes = {19: {}, 20: {}}

    norm_stats.infer_shapes_from_batch(
        {"actions": torch.zeros(100, 4)}, "pushshapes_sim_u_socket"
    )
    norm_stats.infer_shapes_from_batch(
        {"actions": torch.zeros(100, 6)}, "pushshapes_sim_chain_gripper"
    )

    assert norm_stats.shapes[19]["actions"] == (100, 4)
    assert norm_stats.shapes[20]["actions"] == (100, 6)


def test_two_domain_metadata_and_atomic_cache(tmp_path):
    source = _two_domain_action_stats()
    source.infer_norm_from_dataset(
        _ActionDataset(5, 4), 19, sample_frac=1.0, num_workers=0, batch_size=2
    )
    source.infer_norm_from_dataset(
        _ActionDataset(7, 6), 20, sample_frac=1.0, num_workers=0, batch_size=2
    )
    source.cache_stats(str(tmp_path))

    cache_path = tmp_path / "norm_stats" / "norm_stats.json"
    payload = json.loads(cache_path.read_text())
    metadata = payload["norm_run_metadata"]

    assert payload["frames"] == 12
    assert metadata["total_dataset_frames"] == 12
    assert metadata["total_sampled_frames"] == 12
    assert set(metadata["embodiments"]) == {"19", "20"}
    assert metadata["embodiments"]["19"]["dataset_size"] == 5
    assert metadata["embodiments"]["19"]["sampled_frames"] == 5
    assert metadata["embodiments"]["20"]["dataset_size"] == 7
    assert metadata["embodiments"]["20"]["sampled_frames"] == 7
    assert not list(cache_path.parent.glob(".norm_stats.*.tmp"))


def test_atomic_cache_failure_preserves_previous_artifact(monkeypatch, tmp_path):
    source = _action_only_norm_stats(True, (4,))
    source.cache_stats(str(tmp_path))
    cache_path = tmp_path / "norm_stats" / "norm_stats.json"
    original = cache_path.read_bytes()

    def fail_replace(source_path, destination_path):
        assert Path(destination_path) == cache_path
        raise OSError("injected replace failure")

    monkeypatch.setattr(zarr_dataset_multi.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        source.cache_stats(str(tmp_path))

    assert cache_path.read_bytes() == original
    assert not list(cache_path.parent.glob(".norm_stats.*.tmp"))


def test_shared_precomputed_artifact_loads_each_embodiment(tmp_path):
    source = _two_domain_action_stats()
    source.infer_norm_from_dataset(
        _ActionDataset(5, 4), 19, sample_frac=1.0, num_workers=0, batch_size=2
    )
    source.infer_norm_from_dataset(
        _ActionDataset(7, 6), 20, sample_frac=1.0, num_workers=0, batch_size=2
    )
    source.cache_stats(str(tmp_path))

    target = _two_domain_action_stats()
    cache_path = tmp_path / "norm_stats" / "norm_stats.json"
    target.infer_norm_from_dataset(object(), 19, precomputed_norm_path=cache_path)
    target.infer_norm_from_dataset(object(), 20, precomputed_norm_path=cache_path)

    assert target.norm_stats[19]["actions"]["min"].shape == (4,)
    assert target.norm_stats[20]["actions"]["min"].shape == (6,)
