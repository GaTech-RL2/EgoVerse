import logging

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, default_collate

from egomimic.rldb.weighted_dataset import WeightedDataset

logger = logging.getLogger(__name__)


class MultiDataModuleWrapper(LightningDataModule):
    """
    Load per-embodiment batches or a weighted mixture of the training datasets.
    """

    def __init__(
        self,
        train_datasets: dict,
        valid_datasets: dict,
        train_dataloader_params: dict,
        valid_dataloader_params: dict,
        dataset_weights: dict | None = None,
        weighted_dataloader_params: dict | None = None,
        samples_per_epoch: int | None = None,
        sampling_seed: int = 42,
    ):
        """
        Args:
            train_datasets: dictionary of train datasets
            valid_datasets: dictionary of valid datasets
            train_dataloader_params: dictionary of train dataloader parameters
            valid_dataloader_params: dictionary of valid dataloader parameters
            dataset_weights: optional relative sampling probabilities per dataset
            weighted_dataloader_params: shared loader settings, including total batch_size
            samples_per_epoch: global number of mixture draws (None uses active source lengths)
            sampling_seed: seed shared across ranks; the sampler adds the current epoch

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
        self.valid_datasets = {k: v for k, v in valid_datasets.items() if v is not None}
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.collate_fn = annotation_collate
        self.weighted_dataset = (
            WeightedDataset(self.train_datasets, dataset_weights)
            if dataset_weights is not None
            else None
        )
        self.weighted_dataloader_params = weighted_dataloader_params
        self.samples_per_epoch = samples_per_epoch
        self.sampling_seed = sampling_seed
        if self.weighted_dataset is not None and not weighted_dataloader_params:
            raise ValueError("dataset_weights requires weighted_dataloader_params")
        if self.weighted_dataset is None and (
            weighted_dataloader_params is not None or samples_per_epoch is not None
        ):
            raise ValueError("Weighted loader settings require dataset_weights")

    def train_dataloader(self):
        if self.weighted_dataset is not None:
            params = dict(self.weighted_dataloader_params)
            if any(
                key in params
                for key in ("shuffle", "sampler", "batch_sampler", "collate_fn")
            ):
                raise ValueError("The weighted loader owns sampling and collation")
            trainer = self.trainer
            sampler = self.weighted_dataset.sampler(
                num_samples=self.samples_per_epoch,
                seed=self.sampling_seed,
                num_replicas=trainer.world_size if trainer is not None else 1,
                rank=trainer.global_rank if trainer is not None else 0,
            )
            return DataLoader(
                self.weighted_dataset,
                sampler=sampler,
                collate_fn=weighted_collate,
                **params,
            )
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

    def val_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.valid_datasets.items():
            dataset_params = self.valid_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config valid_dataloader_params."
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
    batch = [dict(sample) for sample in batch]
    extracted = _extract_list_keys(batch)
    collated = default_collate(batch)
    collated.update(extracted)
    return collated


def weighted_collate(batch):
    """Keep each dataset's schema intact until model-side homogeneous batching."""
    by_dataset = {}
    for name, sample in batch:
        by_dataset.setdefault(name, []).append(sample)
    return {name: annotation_collate(samples) for name, samples in by_dataset.items()}
