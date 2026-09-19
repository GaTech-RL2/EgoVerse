from collections import Counter

import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper, annotation_collate
from egomimic.rldb.weighted_dataset import WeightedDataset


def _mixture(weights=None):
    return WeightedDataset(
        {
            "small": TensorDataset(torch.arange(10)),
            "large": TensorDataset(torch.arange(1000)),
        },
        weights,
    )


def test_weights_control_dataset_probability_not_dataset_length():
    dataset = _mixture({"small": 3, "large": 1})
    sampler = dataset.sampler(num_samples=20000, seed=5)
    indices = list(sampler)
    counts = Counter(dataset[index][0] for index in indices)
    assert 0.73 < counts["small"] / len(indices) < 0.77
    # Sampling inside the small dataset remains uniform.
    local_counts = Counter(index for index in indices if index < 10)
    assert len(local_counts) == 10
    assert all(1300 < count < 1700 for count in local_counts.values())


def test_default_weights_are_equal_and_zero_disables_a_dataset():
    indices = list(_mixture().sampler(num_samples=10000))
    assert 0.48 < sum(index < 10 for index in indices) / len(indices) < 0.52
    dataset = _mixture({"small": 1, "large": 0})
    assert len(dataset.sampler()) == 10
    assert all(index < 10 for index in dataset.sampler(num_samples=1000))


@pytest.mark.parametrize(
    "weights",
    [
        {"small": -1, "large": 1},
        {"small": float("nan"), "large": 1},
        {"small": float("inf"), "large": 1},
        {"small": 0, "large": 0},
        {"small": 1},
        {"small": 1, "large": 1, "typo": 1},
    ],
)
def test_invalid_weights_fail_early(weights):
    with pytest.raises(ValueError):
        _mixture(weights)


def test_empty_datasets_and_index_bounds():
    with pytest.raises(ValueError, match="at least one"):
        WeightedDataset({})
    children = {"empty": [], "valid": [123]}
    with pytest.raises(ValueError, match="empty"):
        WeightedDataset(children)
    dataset = WeightedDataset(children, {"empty": 0, "valid": 1})
    assert dataset[0] == dataset[-1] == ("valid", 123)
    assert list(dataset.sampler(num_samples=2)) == [0, 0]
    for index in [1, -2]:
        with pytest.raises(IndexError):
            dataset[index]


def test_sampling_is_reproducible_sharded_and_changes_each_epoch():
    dataset = _mixture()
    serial = dataset.sampler(num_samples=102, seed=17)
    ranks = [
        dataset.sampler(num_samples=101, seed=17, num_replicas=2, rank=rank)
        for rank in range(2)
    ]
    first = list(serial)
    assert list(serial) == first
    for rank, sampler in enumerate(ranks):
        assert len(sampler) == 51
        assert list(sampler) == first[rank::2]
        sampler.set_epoch(1)
    serial.set_epoch(1)
    second = list(serial)
    assert second != first
    assert list(ranks[0]) == second[::2]
    assert list(ranks[1]) == second[1::2]


@pytest.mark.parametrize("num_samples", [0, -1, 1.5, True])
def test_invalid_epoch_size(num_samples):
    with pytest.raises(ValueError, match="positive integer"):
        _mixture().sampler(num_samples=num_samples)


class _Samples(Dataset):
    def __init__(self, dim, length=20):
        self.dim = dim
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "actions": torch.full((2, self.dim), float(index)),
            "annotations": ["one"] * (index % 3),
        }


def _datamodule(**kwargs):
    datasets = {"eva_bimanual": _Samples(14), "human_bimanual": _Samples(12)}
    params = {name: {"batch_size": 2, "num_workers": 0} for name in datasets}
    return MultiDataModuleWrapper(datasets, datasets, params, params, **kwargs)


def test_weighted_loader_keeps_heterogeneous_shapes_and_validation_contract():
    datamodule = _datamodule(
        dataset_weights={"eva_bimanual": 1, "human_bimanual": 3},
        weighted_dataloader_params={"batch_size": 16, "num_workers": 0},
        samples_per_epoch=32,
    )
    loader = datamodule.train_dataloader()
    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        assert sum(value["actions"].shape[0] for value in batch.values()) == 16
        for name, values in batch.items():
            assert values["actions"].shape[-1] == (14 if name == "eva_bimanual" else 12)
            assert isinstance(values["annotations"], list)
            assert len(values["annotations"]) == values["actions"].shape[0]
    valid_batch, _, _ = next(iter(datamodule.val_dataloader()))
    assert set(valid_batch) == {"eva_bimanual", "human_bimanual"}
    assert all(value["actions"].shape[0] == 2 for value in valid_batch.values())


def test_legacy_loader_still_returns_every_embodiment():
    batch, _, _ = next(iter(_datamodule().train_dataloader()))
    assert set(batch) == {"eva_bimanual", "human_bimanual"}


def test_collation_does_not_mutate_cached_samples():
    sample = {"actions": torch.zeros(2, 3), "annotations": ["a", "b"]}
    assert annotation_collate([sample, sample])["annotations"] == [
        ["a", "b"],
        ["a", "b"],
    ]
    assert "annotations" in sample


def test_weighted_loader_requires_explicit_shared_batch_settings():
    with pytest.raises(ValueError, match="requires weighted_dataloader_params"):
        _datamodule(dataset_weights={"eva_bimanual": 1, "human_bimanual": 1})
