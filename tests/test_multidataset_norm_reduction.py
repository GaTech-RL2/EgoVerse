import json

import numpy as np
import pytest
import torch

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
