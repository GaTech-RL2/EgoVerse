import logging

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, DistributedSampler, Sampler, default_collate

logger = logging.getLogger(__name__)


class NonPaddingDistributedSampler(Sampler[int]):
    """Deterministically shard validation indices without padding or overlap."""

    def __init__(self, dataset, *, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}"
            )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = max(0, len(self.dataset) - self.rank)
        return (remaining + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        del epoch


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
        valid_combined_mode: str = "max_size_cycle",
        manage_distributed_samplers: bool = False,
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
        self.valid_datasets = {k: v for k, v in valid_datasets.items() if v is not None}
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.manage_distributed_samplers = bool(manage_distributed_samplers)
        if valid_combined_mode not in {"min_size", "max_size_cycle", "max_size"}:
            raise ValueError(
                "valid_combined_mode must be min_size, max_size_cycle, or "
                f"max_size; got {valid_combined_mode!r}"
            )
        self.valid_combined_mode = valid_combined_mode
        self.collate_fn = annotation_collate

    def _distributed_context(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 1, 0
        world_size = int(getattr(trainer, "world_size", 1))
        rank = int(getattr(trainer, "global_rank", 0))
        if world_size <= 0 or not 0 <= rank < world_size:
            raise RuntimeError(
                f"Invalid Trainer distributed context rank={rank} world={world_size}"
            )
        return world_size, rank

    def train_dataloader(self):
        iterables = dict()
        world_size, rank = self._distributed_context()
        for dataset_name, dataset in self.train_datasets.items():
            dataset_params = self.train_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config train_dataloader_params."
                )
            sampler = None
            shuffle = True
            if self.manage_distributed_samplers and world_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=42,
                    drop_last=False,
                )
                shuffle = False
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")

    def val_dataloader(self):
        iterables = dict()
        world_size, rank = self._distributed_context()
        for dataset_name, dataset in self.valid_datasets.items():
            dataset_params = self.valid_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config valid_dataloader_params."
                )
            dataset_params = dict(dataset_params)
            shuffle = dataset_params.pop("shuffle", False)
            sampler = None
            if self.manage_distributed_samplers and world_size > 1:
                if shuffle:
                    raise ValueError(
                        "Exact distributed validation requires shuffle=false"
                    )
                sampler = NonPaddingDistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                )
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        # ``max_size`` is useful for immutable validation corpora whose
        # embodiments contain different numbers of windows: it visits every
        # window once and yields ``None`` for an exhausted domain instead of
        # silently cycling and double-counting its first batches. Training
        # intentionally keeps the historical ``max_size_cycle`` behavior.
        return CombinedLoader(iterables, self.valid_combined_mode)


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
