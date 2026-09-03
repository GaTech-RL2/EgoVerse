import logging
import random
from typing import Literal

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from termcolor import cprint
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Name of the val group that keeps the pre-groups behaviour: its metrics stay
# `Valid/...` and its videos stay in `videos/epoch_N/{embodiment}/`, so runs
# predating multi-group validation overlay on the same wandb charts. Every
# other group is namespaced (`Valid_{group}/...`, `videos/epoch_N/{group}/...`).
DEFAULT_VALID_GROUP = "valid"


def _is_embodiment_name(name) -> bool:
    """True when *name* is an embodiment (or legacy alias) the algo can route.

    This is what separates the two shapes `valid_datasets` accepts. It is the
    same lookup `process_batch_for_training` performs on the batch dict key
    (`algo.py`, `get_embodiment_id(embodiment_name)`), so a key that passes
    here is exactly a key the algo can consume.
    """
    try:
        get_embodiment_id(str(name))
    except (KeyError, AttributeError):
        return False
    return True


def _as_valid_groups(valid_datasets: dict) -> dict:
    """Normalise `valid_datasets` to `{group_name: {embodiment: dataset}}`.

    Two accepted shapes:

      * FLAT -- `{embodiment: dataset}`. Every key is an embodiment name, so
        the whole mapping becomes the single `DEFAULT_VALID_GROUP` group. This
        is the historical shape and stays byte-identical in behaviour.
      * GROUPED -- `{group_name: {embodiment: dataset}}`. Keys are arbitrary
        labels (`valid`, `newtask`, ...), each mapping to a per-embodiment
        dict. Lightning runs one val loop per group, in insertion order.

    The shapes are told apart by whether the top-level keys are embodiment
    names rather than by inspecting value types -- hydra hands us instantiated
    objects for the flat shape and DictConfig/dict for the grouped one, and
    that distinction is fragile across omegaconf versions.
    """
    if not valid_datasets:
        return {}

    keys = list(valid_datasets.keys())
    if all(_is_embodiment_name(k) for k in keys):
        return {DEFAULT_VALID_GROUP: dict(valid_datasets)}

    mixed = [k for k in keys if _is_embodiment_name(k)]
    if mixed:
        raise ValueError(
            "valid_datasets mixes embodiment keys with val-group keys "
            f"({mixed} look like embodiments, {[k for k in keys if k not in mixed]} "
            "do not). Use either {embodiment: dataset} or "
            "{group: {embodiment: dataset}}, not both."
        )

    groups = {}
    for group_name, members in valid_datasets.items():
        try:
            members = dict(members)
        except TypeError as exc:
            raise ValueError(
                f"val group {group_name!r} must map embodiment -> dataset, got "
                f"{type(members).__name__}."
            ) from exc
        bad = [k for k in members if not _is_embodiment_name(k)]
        if bad:
            raise ValueError(
                f"val group {group_name!r} has non-embodiment keys {bad}. Every "
                "key must be an embodiment because the algo routes batches by "
                "that key (process_batch_for_training -> get_embodiment_id)."
            )
        groups[group_name] = members
    return groups


def _params_for_group(valid_dataloader_params: dict, group_name: str) -> dict:
    """Per-group dataloader params, falling back to a single shared block.

    `valid_dataloader_params` may be keyed by embodiment (one block shared by
    every group) or by group name (a block per group). The former is the
    historical shape.
    """
    if not valid_dataloader_params:
        return {}
    if all(_is_embodiment_name(k) for k in valid_dataloader_params):
        return valid_dataloader_params
    return valid_dataloader_params.get(group_name, {})



class MultiDataModuleWrapper(LightningDataModule):
    """
    New functionality for dictionary based multi embodiment loading using CombinedLoader.

    Uses hydra to instantiate DataLoader objects and then wraps them in a combined loader
    """

    def __init__(
        self,
        train_datasets: dict,
        valid_datasets: dict,
        train_dataloader_params: dict,
        valid_dataloader_params: dict,
    ):
        """
        Args:
            train_datasets: dictionary of train datasets
            valid_datasets: dictionary of valid datasets
            train_dataloader_params: dictionary of train dataloader parameters
            valid_dataloader_params: dictionary of valid dataloader parameters

        Tokenization (sampling a prompt from per-sample annotation lists,
        splicing in embodiment / control-mode / proprio blocks, and running
        the HF tokenizer) lives on the algo side now — see
        ``PI.process_batch_for_training``. The collate here only stacks
        tensors and preserves variable-length list-valued keys (e.g. raw
        ``annotations``) so the algo can consume them downstream.
        """
        super().__init__()
        # Drop `None` slots so downstream iteration sites don't need null guards.
        # `None` entries arise when an inheriting data config opts out of a
        # dataset defined in a base (e.g. `aria_bimanual: null`).
        self.train_datasets = {k: v for k, v in train_datasets.items() if v is not None}
        # `valid_datasets` may be flat ({embodiment: dataset}) or grouped
        # ({group: {embodiment: dataset}}); normalise to the grouped form and
        # drop `None` slots inside each group.
        self.valid_groups = {
            group: {k: v for k, v in members.items() if v is not None}
            for group, members in _as_valid_groups(valid_datasets).items()
        }
        self.valid_groups = {g: m for g, m in self.valid_groups.items() if m}
        # Positional: Lightning hands `validation_step` a `dataloader_idx` that
        # indexes this list, and that is how the evaluator recovers the group
        # name for metric prefixes and video paths.
        self.valid_group_names = list(self.valid_groups.keys())
        # Kept for callers (and checkpoint/eval paths) that expect the flat
        # attribute; it is the default group when present, else the first.
        self.valid_datasets = self.valid_groups.get(
            DEFAULT_VALID_GROUP,
            next(iter(self.valid_groups.values()), {}),
        )
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.collate_fn = annotation_collate

    def iter_valid_datasets(self):
        """Yield ``(group, embodiment, dataset)`` for EVERY val dataset.

        Use this for anything that must touch all of validation -- norm-stats
        wiring above all. ``self.valid_datasets`` is only a back-compat alias for
        a SINGLE group (the default if present, else the first), so iterating it
        silently skips every other group. That is exactly how the `newtask` group
        shipped unnormalised: its datasets never received norm stats, the
        evaluator unnormalised them anyway, and both its overlays and its metrics
        were computed on doubly-unnormalised actions.
        """
        for group, members in self.valid_groups.items():
            for embodiment, dataset in members.items():
                yield group, embodiment, dataset

    def train_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.train_datasets.items():
            dataset_params = self.train_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config train_dataloader_params."
                )
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=True,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")

    def _val_loader_for_group(self, group_name: str) -> CombinedLoader:
        group_params = _params_for_group(self.valid_dataloader_params, group_name)
        iterables = dict()
        for dataset_name, dataset in self.valid_groups[group_name].items():
            dataset_params = group_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name} in val group {group_name!r}. Please add {dataset_name} into your data config valid_dataloader_params."
                )
            dataset_params = dict(dataset_params)
            shuffle = dataset_params.pop("shuffle", False)
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=shuffle,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")

    def val_dataloader(self):
        """One CombinedLoader per val group.

        A single group returns the bare CombinedLoader, which is exactly what
        this method returned before groups existed -- so single-group runs keep
        their old dataloader topology and `dataloader_idx` stays 0. Multiple
        groups return a list, and Lightning then runs the val loops back to
        back in list order, passing the group's index as `dataloader_idx`.
        """
        loaders = [self._val_loader_for_group(g) for g in self.valid_group_names]
        if len(loaders) == 1:
            return loaders[0]
        return loaders




def _extract_list_keys(batch):
    """Pop all list-valued keys from *batch* samples and return them separately.

    This lets ``default_collate`` handle tensors / numbers while variable-length
    annotation lists (``key_type == "annotation_keys"``) are preserved as
    ``list[list[str]]``.
    """
    list_keys = {k for k in batch[0] if isinstance(batch[0][k], list)}
    return {k: [sample.pop(k) for sample in batch] for k in list_keys}


def _extract_keys(batch, keys):
    return {k: [sample.pop(k) for sample in batch] for k in keys}


def annotation_collate(batch):
    """Collate that preserves variable-length list-valued keys (e.g. annotation_keys)."""
    extracted = _extract_list_keys(batch)
    collated = default_collate(batch)
    collated.update(extracted)
    return collated


