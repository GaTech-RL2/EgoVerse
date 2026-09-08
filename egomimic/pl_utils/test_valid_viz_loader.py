"""The sequential validation-video loader (MultiDataModuleWrapper.valid_viz_params):
episode selection, contiguous windows, the [metrics, viz] loader pair, and the
index repetition that keeps every DDP rank on the full sequence."""

import pytest
import torch
from torch.utils.data import Dataset, DistributedSampler

from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper, viz_indices


class _FakeMulti(Dataset):
    """Minimal stand-in for a MultiDataset: 4 episodes of unequal length,
    two operators (one held out), samples carry their global index."""

    def __init__(self, tags=True):
        lengths = {"ep_b": 50, "ep_a": 40, "ep_d": 30, "ep_c": 20}
        self.index_map = []
        self._global_indices_by_dataset = {}
        for name, n in lengths.items():
            self._global_indices_by_dataset[name] = []
            for local in range(n):
                self._global_indices_by_dataset[name].append(len(self.index_map))
                self.index_map.append((name, local))
        self.episode_names = sorted(lengths)
        if tags:
            self._operator_seen = {
                "ep_a": True,
                "ep_b": True,
                "ep_c": False,
                "ep_d": False,
            }
            self._episodes_by_group = {"op1": ["ep_a", "ep_b"], "op2": ["ep_c", "ep_d"]}

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        name, local = self.index_map[i]
        return {
            "gidx": torch.tensor(i),
            "local": torch.tensor(local),
            "x": torch.zeros(2),
        }


def test_viz_indices_auto_picks_first_seen_and_first_unseen_windows():
    ds = _FakeMulti()
    idx = viz_indices(ds, "auto", frames_per_episode=10, start_frac=0.5)
    a = ds._global_indices_by_dataset["ep_a"]  # first seen by name
    c = ds._global_indices_by_dataset["ep_c"]  # first unseen by name
    assert idx == a[20:30] + c[10:20]


def test_viz_indices_all_and_explicit_and_whole_episode():
    ds = _FakeMulti()
    by = ds._global_indices_by_dataset
    assert (
        viz_indices(ds, "all", frames_per_episode=None)
        == by["ep_a"] + by["ep_b"] + by["ep_c"] + by["ep_d"]
    )
    assert (
        viz_indices(ds, ["ep_d"], frames_per_episode=5, start_frac=0.0)
        == by["ep_d"][:5]
    )
    # window longer than the episode: clamps to the episode
    assert (
        viz_indices(ds, ["ep_c"], frames_per_episode=100, start_frac=0.9) == by["ep_c"]
    )
    with pytest.raises(ValueError, match="not in the validation set"):
        viz_indices(ds, ["nope"])
    with pytest.raises(ValueError, match="start_frac"):
        viz_indices(ds, "all", 5, start_frac=1.0)


def test_viz_indices_per_group_takes_first_episode_of_each_operator():
    ds = _FakeMulti()
    by = ds._global_indices_by_dataset
    assert (
        viz_indices(ds, "per_group", frames_per_episode=None) == by["ep_a"] + by["ep_c"]
    )
    with pytest.raises(ValueError, match="per_group"):
        viz_indices(_FakeMulti(tags=False), "per_group")


def test_viz_indices_without_tags_takes_first_episode():
    ds = _FakeMulti(tags=False)
    assert (
        viz_indices(ds, "auto", frames_per_episode=4, start_frac=0.0)
        == ds._global_indices_by_dataset["ep_a"][:4]
    )


def _dm(viz):
    ds = _FakeMulti()
    return MultiDataModuleWrapper(
        train_datasets={"h": ds},
        valid_datasets={"h": ds},
        train_dataloader_params={"h": {"batch_size": 8}},
        valid_dataloader_params={"h": {"batch_size": 8, "shuffle": True}},
        valid_viz_params=viz,
    ), ds


def test_val_dataloader_without_viz_is_unchanged():
    dm, _ = _dm(None)
    assert dm.viz_dataloader_idx is None
    assert not isinstance(dm.val_dataloader(), list)


def test_val_dataloader_with_viz_returns_pair_and_viz_is_contiguous():
    dm, ds = _dm(
        {
            "h": {
                "episodes": "auto",
                "frames_per_episode": 12,
                "start_frac": 0.25,
                "batch_size": 5,
            }
        }
    )
    assert dm.viz_dataloader_idx == 1
    loaders = dm.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 2
    seen = []
    for batch in loaders[1]:
        seen.extend(batch["h"]["gidx"].tolist())
    a = ds._global_indices_by_dataset["ep_a"]
    c = ds._global_indices_by_dataset["ep_c"]
    assert seen == a[10:22] + c[5:17]


def test_viz_only_returns_just_the_viz_loader_at_index_zero():
    dm, ds = _dm(
        {
            "h": {
                "episodes": ["ep_d"],
                "frames_per_episode": 6,
                "start_frac": 0.0,
                "batch_size": 3,
                "viz_only": True,
            }
        }
    )
    assert dm.viz_only and dm.viz_dataloader_idx == 0
    loaders = dm.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 1
    got = [i for b in loaders[0] for i in b["h"]["gidx"].tolist()]
    assert got == ds._global_indices_by_dataset["ep_d"][:6]


def test_viz_indices_repeated_per_rank_survive_distributed_sharding():
    # Lightning wraps val loaders in DistributedSampler(shuffle=False), which
    # hands rank r the positions r, r+W, ...; repeating every index W times
    # gives each rank the whole contiguous sequence, in order.
    ds = _FakeMulti()
    idx = viz_indices(ds, ["ep_d"], frames_per_episode=6, start_frac=0.0)
    world = 3
    repeated = [i for i in idx for _ in range(world)]
    for rank in range(world):
        sampler = DistributedSampler(
            repeated, num_replicas=world, rank=rank, shuffle=False
        )
        got = [repeated[p] for p in sampler]
        assert got == idx


def test_viz_pair_are_plain_loaders_lightning_can_shard():
    # Lightning only shards / set_epoch()s plain DataLoaders it can see, so with
    # a viz loader both val loaders must be DataLoaders (not nested
    # CombinedLoaders) that already yield {dataset_name: batch} dicts.
    from types import SimpleNamespace

    from lightning.pytorch.utilities.combined_loader import CombinedLoader
    from torch.utils.data import DataLoader, DistributedSampler

    from egomimic.pl_utils.pl_model import unwrap_combined_batch

    dm, ds = _dm(
        {
            "h": {
                "episodes": ["ep_c"],
                "frames_per_episode": 6,
                "start_frac": 0.0,
                "batch_size": 3,
            }
        }
    )
    dm.trainer = SimpleNamespace(world_size=2)  # as under 2-GPU DDP
    loaders = dm.val_dataloader()
    assert all(isinstance(ld, DataLoader) for ld in loaders)
    outer = CombinedLoader(loaders, "sequential")
    seen_idx = set()
    for batch, _bidx, didx in outer:
        seen_idx.add(didx)
        assert isinstance(batch, dict) and "h" in batch and batch["h"]["gidx"].ndim == 1
        assert unwrap_combined_batch(batch) is batch
    assert seen_idx == {0, 1}
    # What lightning does under DDP: swap in a DistributedSampler. With the
    # index repetition every rank then gets the full contiguous window once.
    viz = loaders[1]
    idx = viz_indices(ds, ["ep_c"], frames_per_episode=6, start_frac=0.0)
    for rank in range(2):
        sharded = DataLoader(
            viz.dataset,
            batch_size=3,
            collate_fn=viz.collate_fn,
            sampler=DistributedSampler(
                viz.dataset, num_replicas=2, rank=rank, shuffle=False
            ),
        )
        got = [i for b in sharded for i in b["h"]["gidx"].tolist()]
        assert got == idx, (rank, got, idx)


def test_on_load_checkpoint_drops_policy_extra_keys():
    from types import SimpleNamespace

    from egomimic.pl_utils.pl_model import ModelWrapper

    # The hook only touches the checkpoint dict; call it unbound with a stub self.
    stub = SimpleNamespace(EXTRA_STATE_DICT_KEYS=ModelWrapper.EXTRA_STATE_DICT_KEYS)
    ckpt = {
        "state_dict": {
            "model.w": torch.zeros(1),
            "_extra_training_split_info": {"a": True},
        },
        "epoch": 3,
    }
    ModelWrapper.on_load_checkpoint(stub, ckpt)
    assert "_extra_training_split_info" not in ckpt["state_dict"]
    assert "model.w" in ckpt["state_dict"] and ckpt["epoch"] == 3
    ModelWrapper.on_load_checkpoint(stub, {"epoch": 1})  # no state_dict: no-op
