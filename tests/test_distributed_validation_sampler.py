from types import SimpleNamespace

import torch
from torch.utils.data import DistributedSampler, TensorDataset

from egomimic.pl_utils.pl_data_utils import (
    MultiDataModuleWrapper,
    NonPaddingDistributedSampler,
)


def test_nonpadding_sampler_partitions_odd_dataset_exactly_once():
    dataset = list(range(11))
    shards = [
        list(NonPaddingDistributedSampler(dataset, num_replicas=2, rank=rank))
        for rank in range(2)
    ]

    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == list(range(11))
    assert [len(shard) for shard in shards] == [6, 5]


def test_datamodule_owns_train_and_exact_validation_shards_when_enabled():
    dataset = TensorDataset(torch.arange(11))
    modules = []
    for rank in range(2):
        module = MultiDataModuleWrapper(
            train_datasets={"domain": dataset},
            valid_datasets={"domain": dataset},
            train_dataloader_params={"domain": {"batch_size": 2}},
            valid_dataloader_params={"domain": {"batch_size": 2}},
            valid_combined_mode="max_size",
            manage_distributed_samplers=True,
        )
        module.trainer = SimpleNamespace(world_size=2, global_rank=rank)
        modules.append(module)

    train_sampler = modules[0].train_dataloader().iterables["domain"].sampler
    assert isinstance(train_sampler, DistributedSampler)

    valid_samplers = [
        module.val_dataloader().iterables["domain"].sampler for module in modules
    ]
    assert all(
        isinstance(item, NonPaddingDistributedSampler) for item in valid_samplers
    )
    shards = [list(item) for item in valid_samplers]
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == list(range(11))
