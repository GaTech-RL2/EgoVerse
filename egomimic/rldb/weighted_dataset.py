"""Dataset mixtures with dataset-level (rather than frame-level) weights."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping
from itertools import accumulate

import torch
from torch.utils.data import Dataset, DistributedSampler


class WeightedDataset(Dataset):
    """Concatenate named datasets and sample them with replacement.

    ``weights`` specifies relative probabilities of choosing a *dataset*;
    frames within that dataset are uniform. For example, weights 3 and 1
    produce a 75%/25% mixture even when the dataset lengths differ. Omitted
    weights give every dataset equal probability; a zero weight disables it.

    Indexing remains deterministic and returns ``(dataset_name, sample)``.
    Use :meth:`sampler` with a DataLoader to apply the weights. Keeping the
    randomness in the sampler makes sampling independent of worker count and
    lets distributed workers partition the same seeded draw each epoch.
    """

    def __init__(self, datasets: Mapping[str, Dataset], weights=None):
        self.datasets = dict(datasets)
        if not self.datasets:
            raise ValueError("WeightedDataset requires at least one dataset")
        self.names = tuple(self.datasets)
        if weights is None:
            weights = dict.fromkeys(self.names, 1.0)
        if set(weights) != set(self.names):
            raise ValueError("weights must have exactly the same keys as datasets")
        self.weights = {name: float(weights[name]) for name in self.names}
        if any(not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError("Dataset weights must be finite and non-negative")
        if not any(self.weights.values()):
            raise ValueError("At least one dataset weight must be positive")
        self.lengths = tuple(len(self.datasets[name]) for name in self.names)
        for name, length in zip(self.names, self.lengths):
            if length == 0 and self.weights[name] > 0:
                raise ValueError(f"Positive-weight dataset {name!r} is empty")
        self.cumulative_sizes = tuple(accumulate(self.lengths))
        probabilities = torch.tensor(list(self.weights.values()), dtype=torch.double)
        # Scaling first also handles finite weights whose sum would overflow.
        probabilities /= probabilities.max()
        self.probabilities = probabilities / probabilities.sum()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self.cumulative_sizes, index)
        start = self.cumulative_sizes[dataset_index - 1] if dataset_index else 0
        name = self.names[dataset_index]
        return name, self.datasets[name][index - start]

    def sampler(self, *, num_samples=None, seed=42, num_replicas=1, rank=0):
        """Create a rank-aware sampler; call ``set_epoch`` between epochs.

        ``num_samples`` is the global epoch size, rounded up to a multiple of
        the world size so every rank takes the same number of optimizer steps.
        Lightning calls ``set_epoch`` automatically.
        """
        return WeightedDatasetSampler(
            self,
            num_samples=num_samples,
            seed=seed,
            num_replicas=num_replicas,
            rank=rank,
        )


class WeightedDatasetSampler(DistributedSampler):
    """Seeded mixture draws, sharded across ranks without worker-side RNG."""

    def __init__(self, dataset, *, num_samples=None, seed=42, num_replicas=1, rank=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, seed=seed)
        if num_samples is None:
            num_samples = sum(
                length
                for name, length in zip(dataset.names, dataset.lengths)
                if dataset.weights[name] > 0
            )
        if (
            isinstance(num_samples, bool)
            or int(num_samples) != num_samples
            or num_samples <= 0
        ):
            raise ValueError("num_samples must be a positive integer")
        self.num_samples = math.ceil(num_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        sources = torch.multinomial(
            self.dataset.probabilities,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        indices = torch.empty(self.total_size, dtype=torch.long)
        start = 0
        for source, length in enumerate(self.dataset.lengths):
            positions = (sources == source).nonzero(as_tuple=True)[0]
            if positions.numel():
                indices[positions] = start + torch.randint(
                    length, (positions.numel(),), generator=generator
                )
            start += length
        return iter(indices[self.rank : self.total_size : self.num_replicas].tolist())
