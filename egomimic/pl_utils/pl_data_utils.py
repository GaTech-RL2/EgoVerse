import logging

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, default_collate

from egomimic.rldb.embodiment.embodiment import get_embodiment_id

logger = logging.getLogger(__name__)

# Name of the val group that keeps the pre-groups behaviour: its metrics stay
# `Valid/...`, so runs predating multi-group validation overlay on the same
# wandb charts. Every other group is namespaced (`Valid_{group}/...`).
DEFAULT_VALID_GROUP = "valid"


def _is_embodiment_name(name) -> bool:
    """True when *name* names an embodiment the loader keys can carry.

    This is what separates the two shapes `valid_datasets` accepts. Note the
    pipeline runner itself treats source keys as opaque -- it never resolves an
    embodiment -- so this lookup is used ONLY to disambiguate config shape here,
    never to route a batch.
    """
    try:
        get_embodiment_id(str(name))
    except (KeyError, AttributeError):
        return False
    return True


def as_valid_groups(valid_datasets: dict) -> dict:
    """Normalise `valid_datasets` to `{group_name: {source: dataset}}`.

    Two accepted shapes:

      * FLAT -- `{source: dataset}`. Every key is an embodiment name, so the
        whole mapping becomes the single `DEFAULT_VALID_GROUP` group. This is
        the historical shape and stays identical in behaviour.
      * GROUPED -- `{group_name: {source: dataset}}`. Keys are arbitrary labels
        (`valid`, `newtask`, ...), each mapping to a per-source dict. Lightning
        runs one val loop per group, in insertion order.

    The shapes are told apart by whether the top-level keys are embodiment
    names rather than by inspecting value types -- hydra hands us instantiated
    objects for the flat shape and DictConfig/dict for the grouped one, and
    that distinction is fragile across omegaconf versions.
    """
    if not valid_datasets:
        return {}

    keys = list(valid_datasets.keys())
    if all(_is_embodiment_name(key) for key in keys):
        return {DEFAULT_VALID_GROUP: dict(valid_datasets)}

    mixed = [key for key in keys if _is_embodiment_name(key)]
    if mixed:
        raise ValueError(
            "valid_datasets mixes embodiment keys with val-group keys "
            f"({mixed} look like embodiments, "
            f"{[key for key in keys if key not in mixed]} do not). Use either "
            "{source: dataset} or {group: {source: dataset}}, not both."
        )

    groups = {}
    for group_name, members in valid_datasets.items():
        try:
            members = dict(members)
        except TypeError as exc:
            raise ValueError(
                f"val group {group_name!r} must map source -> dataset, got "
                f"{type(members).__name__}."
            ) from exc
        bad = [key for key in members if not _is_embodiment_name(key)]
        if bad:
            raise ValueError(
                f"val group {group_name!r} has non-embodiment keys {bad}. A "
                "group's keys name datasets, so they must be embodiment names; "
                "nesting groups inside groups is not supported."
            )
        groups[group_name] = members
    return groups


def _params_for_group(valid_dataloader_params: dict, group_name: str) -> dict:
    """Per-group dataloader params, falling back to a single shared block.

    `valid_dataloader_params` may be keyed by source (one block shared by every
    group) or by group name (a block per group). The former is the historical
    shape.
    """
    if not valid_dataloader_params:
        return {}
    if all(_is_embodiment_name(key) for key in valid_dataloader_params):
        return valid_dataloader_params
    return valid_dataloader_params.get(group_name, {})


class MultiDataModuleWrapper(LightningDataModule):
    """
    Build dictionary-based multi-source loaders with Lightning CombinedLoader.

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

        The collate function stacks ordinary values and preserves variable-length
        list-valued fields for whichever configured stage consumes them.
        """
        super().__init__()
        # Drop `None` slots so downstream iteration sites don't need null guards.
        # `None` entries arise when an inheriting data config opts out of a
        # dataset defined in a base (e.g. `aria_bimanual: null`).
        self.train_datasets = {k: v for k, v in train_datasets.items() if v is not None}
        # `valid_datasets` may be flat ({source: dataset}) or grouped
        # ({group: {source: dataset}}); normalise to the grouped form and drop
        # `None` slots inside each group.
        self.valid_groups = {
            group: {k: v for k, v in members.items() if v is not None}
            for group, members in as_valid_groups(valid_datasets).items()
        }
        self.valid_groups = {g: m for g, m in self.valid_groups.items() if m}
        # Positional: Lightning hands `validation_step` a `dataloader_idx` that
        # indexes this list, and that is how the evaluator recovers the group
        # name for its metric prefix.
        self.valid_group_names = list(self.valid_groups.keys())
        # Kept for callers that expect the flat attribute; it is the default
        # group when present, else the first. Anything that must touch ALL of
        # validation uses `iter_valid_datasets` instead.
        self.valid_datasets = self.valid_groups.get(
            DEFAULT_VALID_GROUP,
            next(iter(self.valid_groups.values()), {}),
        )
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.collate_fn = annotation_collate

    def iter_valid_datasets(self):
        """Yield ``(group, source, dataset)`` for EVERY val dataset.

        Use this for anything that must touch all of validation -- norm-stats
        wiring above all. ``self.valid_datasets`` is only a back-compat alias
        for a SINGLE group, so iterating it silently skips every other group,
        leaving those datasets unnormalised while the evaluator unnormalises
        their samples anyway.
        """
        for group, members in self.valid_groups.items():
            for source, dataset in members.items():
                yield group, source, dataset

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
