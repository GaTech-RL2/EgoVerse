"""Nested validation dataset groups in MultiDataModuleWrapper."""

import pytest
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import Dataset

from egomimic.pl_utils.pl_data_utils import (
    DEFAULT_VALID_GROUP,
    MultiDataModuleWrapper,
    as_valid_groups,
)
from egomimic.pl_utils.pl_model import _unwrap_combined_loader_batch

_SOURCE_A = "pushshapes_sim_u_socket"
_SOURCE_B = "pushshapes_sim_chain_gripper"


class _Tiny(Dataset):
    def __init__(self, n: int = 4):
        self.n = n
        self.norm_stats = None

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {"actions": torch.zeros(2, 3)}

    def set_norm_stats_from(self, owner):
        self.norm_stats = owner.norm_stats


def _params(*sources):
    return {source: {"batch_size": 2, "num_workers": 0} for source in sources}


def _module(valid_datasets, valid_params=None):
    return MultiDataModuleWrapper(
        train_datasets={_SOURCE_A: _Tiny()},
        valid_datasets=valid_datasets,
        train_dataloader_params=_params(_SOURCE_A),
        valid_dataloader_params=valid_params or _params(_SOURCE_A, _SOURCE_B),
    )


def test_flat_valid_datasets_become_the_default_group():
    assert as_valid_groups({_SOURCE_A: "ds"}) == {DEFAULT_VALID_GROUP: {_SOURCE_A: "ds"}}


def test_grouped_valid_datasets_keep_their_labels_and_order():
    grouped = {"newtask": {_SOURCE_A: "x"}, "valid": {_SOURCE_B: "y"}}
    assert list(as_valid_groups(grouped)) == ["newtask", "valid"]


def test_mixing_group_keys_with_source_keys_is_rejected():
    with pytest.raises(ValueError, match="mixes embodiment keys"):
        as_valid_groups({_SOURCE_A: "ds", "newtask": {_SOURCE_B: "ds"}})


def test_a_group_of_non_source_keys_is_rejected():
    with pytest.raises(ValueError, match="non-embodiment keys"):
        as_valid_groups({"newtask": {"deeper": {_SOURCE_A: "ds"}}})


def test_a_group_whose_members_are_not_a_mapping_is_rejected():
    with pytest.raises(ValueError, match="must map source -> dataset"):
        as_valid_groups({"newtask": 5})


def test_empty_valid_datasets_yields_no_groups():
    assert as_valid_groups({}) == {}


def test_none_slots_are_dropped_and_emptied_groups_disappear():
    module = _module({"a_group": {_SOURCE_A: None}, "b_group": {_SOURCE_B: _Tiny()}})
    assert module.valid_group_names == ["b_group"]


def test_flat_config_returns_a_bare_loader_so_dataloader_idx_stays_zero():
    module = _module({_SOURCE_A: _Tiny()})
    assert module.valid_group_names == [DEFAULT_VALID_GROUP]
    assert isinstance(module.val_dataloader(), CombinedLoader)


def test_multiple_groups_return_one_loader_each_in_declaration_order():
    module = _module({"valid": {_SOURCE_A: _Tiny()}, "newtask": {_SOURCE_B: _Tiny()}})
    loaders = module.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 2
    assert module.valid_group_names == ["valid", "newtask"]


def test_valid_datasets_alias_prefers_the_default_group():
    module = _module({"newtask": {_SOURCE_B: _Tiny()}, "valid": {_SOURCE_A: _Tiny()}})
    assert list(module.valid_datasets) == [_SOURCE_A]


def test_valid_datasets_alias_falls_back_to_the_first_group():
    module = _module({"newtask": {_SOURCE_B: _Tiny()}, "other": {_SOURCE_A: _Tiny()}})
    assert list(module.valid_datasets) == [_SOURCE_B]


def test_iter_valid_datasets_reaches_every_group_the_alias_skips():
    module = _module({"valid": {_SOURCE_A: _Tiny()}, "newtask": {_SOURCE_B: _Tiny()}})
    seen = [(group, source) for group, source, _ in module.iter_valid_datasets()]
    assert seen == [("valid", _SOURCE_A), ("newtask", _SOURCE_B)]
    # This is the regression the alias caused: iterating it covers only one.
    assert len(list(module.valid_datasets)) < len(seen)


def test_per_group_dataloader_params_are_selected_by_group_name():
    module = _module(
        {"valid": {_SOURCE_A: _Tiny()}, "newtask": {_SOURCE_A: _Tiny()}},
        valid_params={
            "valid": {_SOURCE_A: {"batch_size": 2, "num_workers": 0}},
            "newtask": {_SOURCE_A: {"batch_size": 1, "num_workers": 0}},
        },
    )
    loaders = module.val_dataloader()
    assert loaders[0].flattened[0].batch_size == 2
    assert loaders[1].flattened[0].batch_size == 1


def test_source_keyed_params_are_shared_by_every_group():
    module = _module({"valid": {_SOURCE_A: _Tiny()}, "newtask": {_SOURCE_A: _Tiny()}})
    loaders = module.val_dataloader()
    assert [loader.flattened[0].batch_size for loader in loaders] == [2, 2]


def test_a_group_missing_dataloader_params_names_the_group_it_failed_in():
    module = _module(
        {"newtask": {_SOURCE_B: _Tiny()}}, valid_params=_params(_SOURCE_A)
    )
    with pytest.raises(ValueError, match="in val group 'newtask'"):
        module.val_dataloader()


def test_unwrap_drops_the_extra_tuple_lightning_leaks_for_nested_groups():
    batch = {_SOURCE_A: {"actions": torch.zeros(1)}}
    assert _unwrap_combined_loader_batch((batch, 0, 1)) is batch
    assert _unwrap_combined_loader_batch(batch) is batch


def test_unwrap_leaves_a_non_triple_tuple_alone():
    pair = ({"a": 1}, 0)
    assert _unwrap_combined_loader_batch(pair) is pair


def _bare_evaluator():
    """A PlanarActionEval instance without running its heavy __init__."""
    from egomimic.eval.planar_action_eval import PlanarActionEval

    evaluator = PlanarActionEval.__new__(PlanarActionEval)
    return evaluator


def test_default_group_keeps_the_bare_valid_prefix():
    evaluator = _bare_evaluator()
    evaluator.set_validation_group(DEFAULT_VALID_GROUP)
    metrics = {"Valid/MSE": 1.0, "Train/Loss": 2.0}
    assert evaluator._namespaced(metrics) == metrics


def test_a_named_group_namespaces_only_the_valid_metrics():
    evaluator = _bare_evaluator()
    evaluator.set_validation_group("newtask")
    out = evaluator._namespaced({"Valid/MSE": 1.0, "Valid/MSE/eva": 2.0, "Other": 3.0})
    assert out == {
        "Valid_newtask/MSE": 1.0,
        "Valid_newtask/MSE/eva": 2.0,
        "Other": 3.0,
    }


def test_no_group_set_leaves_metrics_untouched():
    evaluator = _bare_evaluator()
    metrics = {"Valid/MSE": 1.0}
    assert evaluator._namespaced(metrics) == metrics
