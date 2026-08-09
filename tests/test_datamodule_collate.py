"""Regression guard: the training dataloader must use pack_collate for packed
datasets.

Two call sites need pack_collate. zarr_dataset_multi's norm-stat inference had
it and was tested; MultiDataModuleWrapper -- the *training* dataloader -- never
wired it and had no test at all, so every packed_episode config died on its
first batch with "Trying to resize storage that is not resizable".
"""
from unittest import mock

import pytest

from egomimic.pl_utils.pl_data_utils import (
    MultiDataModuleWrapper,
    _collate_fn_for,
    annotation_collate,
)
from egomimic.rldb.zarr.zarr_dataset_packed import ZarrEpisodePackedDataset, pack_collate


def _packed():
    ds = mock.MagicMock(spec=ZarrEpisodePackedDataset)
    ds.__len__.return_value = 4
    return ds


def _unpacked():
    ds = mock.MagicMock()
    ds.__len__.return_value = 4
    return ds


def test_collate_dispatch_picks_pack_collate_for_packed_datasets():
    assert _collate_fn_for(_packed()) is pack_collate


def test_collate_dispatch_falls_back_for_everything_else():
    assert _collate_fn_for(_unpacked()) is annotation_collate


@pytest.mark.parametrize("loader_attr", ["train_dataloader", "val_dataloader"])
def test_datamodule_wires_pack_collate_for_packed_datasets(loader_attr):
    """The dispatch must reach the DataLoader, not just exist as a helper."""
    ds = _packed()
    params = {"pushshapes_sim": {"batch_size": 2, "num_workers": 0}}
    dm = MultiDataModuleWrapper(
        train_datasets={"pushshapes_sim": ds},
        valid_datasets={"pushshapes_sim": ds},
        train_dataloader_params=params,
        valid_dataloader_params=params,
    )
    combined = getattr(dm, loader_attr)()
    loaders = combined.iterables if hasattr(combined, "iterables") else combined
    loaders = loaders.values() if isinstance(loaders, dict) else [loaders]
    for dl in loaders:
        assert dl.collate_fn is pack_collate, (
            f"{loader_attr} used {dl.collate_fn!r}; packed datasets need pack_collate "
            "or default_collate raises 'Trying to resize storage that is not resizable'"
        )


def test_datamodule_keeps_annotation_collate_for_unpacked_datasets():
    ds = _unpacked()
    params = {"eva": {"batch_size": 2, "num_workers": 0}}
    dm = MultiDataModuleWrapper(
        train_datasets={"eva": ds},
        valid_datasets={"eva": ds},
        train_dataloader_params=params,
        valid_dataloader_params=params,
    )
    combined = dm.train_dataloader()
    loaders = combined.iterables if hasattr(combined, "iterables") else combined
    loaders = loaders.values() if isinstance(loaders, dict) else [loaders]
    for dl in loaders:
        assert dl.collate_fn is annotation_collate
