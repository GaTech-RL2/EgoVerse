import logging

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, default_collate

logger = logging.getLogger(__name__)


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
        valid_viz_params: dict | None = None,
    ):
        """
        Args:
            train_datasets: dictionary of train datasets
            valid_datasets: dictionary of valid datasets
            train_dataloader_params: dictionary of train dataloader parameters
            valid_dataloader_params: dictionary of valid dataloader parameters
            valid_viz_params: optional per-dataset params for a second,
                sequential validation loader used only for the validation
                videos (the metrics loader may be shuffled). Keys per dataset:
                ``episodes`` ("auto" = first seen + first unseen episode,
                "all", or a list of episode names), ``frames_per_episode``
                (int, or null for whole episodes), ``start_frac`` (where in
                the episode the window starts, default 0.3), plus DataLoader
                kwargs (``batch_size``, ``num_workers``). When set,
                ``val_dataloader`` returns ``[metrics_loader, viz_loader]``
                and ``viz_dataloader_idx`` is 1; otherwise a single loader
                and ``viz_dataloader_idx`` is None. ``viz_only: true`` drops
                the metrics loader entirely (``[viz_loader]``,
                ``viz_dataloader_idx`` 0): validation then only writes the
                videos, for eval runs that need clips but not numbers.

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
        self.valid_viz_params = {
            k: v for k, v in (valid_viz_params or {}).items() if v is not None
        }
        self.viz_only = any(
            bool(p.get("viz_only", False)) for p in self.valid_viz_params.values()
        )
        if self.viz_only:
            self.viz_dataloader_idx = 0
        else:
            self.viz_dataloader_idx = 1 if self.valid_viz_params else None
        self.collate_fn = annotation_collate

    def train_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.train_datasets.items():
            dataset_params = self.train_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config train_dataloader_params."
                )
            # Datasets that expose per-index sample weights (e.g.
            # EpisodePromptMultiDataset with balance_by=group) get a weighted
            # sampler instead of uniform shuffling. Lightning wraps custom
            # samplers in DistributedSamplerWrapper under DDP.
            sampler = None
            weights_fn = getattr(dataset, "sample_weights", None)
            weights = weights_fn() if callable(weights_fn) else None
            if weights is not None:
                sampler = WeightedRandomSampler(
                    weights, num_samples=len(weights), replacement=True
                )
                logger.info(
                    f"Using WeightedRandomSampler for dataset {dataset_name} "
                    f"({len(weights)} indices)."
                )
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=sampler is None,
                sampler=sampler,
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

        if not self.valid_viz_params:
            return CombinedLoader(iterables, "max_size_cycle")
        if self.viz_only:
            return [self._viz_dataloader()]
        # With a viz loader the two loaders must be plain DataLoaders: lightning
        # only shards (DistributedSampler) and set_epoch()s loaders it can see,
        # and it does not look inside a CombinedLoader nested in a list. Each
        # plain loader wraps its batch as {dataset_name: batch} via the collate
        # so the algo sees the same structure a CombinedLoader would give.
        if len(iterables) != 1:
            raise NotImplementedError(
                "valid_viz_params is only supported with a single valid dataset "
                f"(got {sorted(iterables)})."
            )
        ((name, metrics_loader),) = iterables.items()
        metrics_loader = DataLoader(
            metrics_loader.dataset,
            shuffle=shuffle,
            collate_fn=_NamedCollate(name, self.collate_fn),
            **dataset_params,
        )
        return [metrics_loader, self._viz_dataloader()]

    def _viz_dataloader(self):
        """Sequential loader over contiguous frames of a few validation
        episodes, for the validation videos. Under DDP Lightning shards any
        loader with a DistributedSampler (rank r gets positions r, r+W, ...),
        so every index is repeated ``world_size`` times: each rank then sees
        the full contiguous sequence once, in order."""
        world_size = 1
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            world_size = int(getattr(trainer, "world_size", 1) or 1)
        for dataset_name, params in self.valid_viz_params.items():
            dataset = self.valid_datasets.get(dataset_name)
            if dataset is None:
                raise ValueError(
                    f"valid_viz_params names dataset {dataset_name!r}, which is not "
                    f"in valid_datasets {sorted(self.valid_datasets)}"
                )
            params = dict(params)
            episodes = params.pop("episodes", "auto")
            frames_per_episode = params.pop("frames_per_episode", 300)
            start_frac = float(params.pop("start_frac", 0.3))
            params.pop("shuffle", None)
            params.pop("viz_only", None)
            indices = viz_indices(dataset, episodes, frames_per_episode, start_frac)
            logger.info(
                f"Validation viz loader for {dataset_name}: {len(indices)} frames "
                f"(episodes={episodes!r}, frames_per_episode={frames_per_episode}, "
                f"world_size={world_size})."
            )
            repeated = [i for i in indices for _ in range(world_size)]
            loader = DataLoader(
                Subset(dataset, repeated),
                shuffle=False,
                collate_fn=_NamedCollate(dataset_name, self.collate_fn),
                **params,
            )
            return loader  # single dataset (checked in val_dataloader)
        raise ValueError("valid_viz_params is empty")


class _NamedCollate:
    """Picklable collate wrapper returning ``{dataset_name: collate(samples)}``,
    the per-dataset dict structure CombinedLoader batches have."""

    def __init__(self, name, collate_fn):
        self.name = name
        self.collate_fn = collate_fn

    def __call__(self, samples):
        return {self.name: self.collate_fn(samples)}


def viz_indices(dataset, episodes="auto", frames_per_episode=300, start_frac=0.3):
    """Global sample indices for the validation-video loader: a contiguous
    window per chosen episode, episodes in sorted-name order.

    ``dataset`` must expose ``_global_indices_by_dataset`` (MultiDataset);
    ``episodes`` is "auto" (first episode tagged seen and first tagged unseen
    via ``_operator_seen``, or just the first episode when there are no tags),
    "all", "per_group" (first episode by name of every prompt group, i.e. one
    per operator, via ``_episodes_by_group``), or an explicit list of episode
    names. ``frames_per_episode`` None means the whole episode.
    """
    by_ep = getattr(dataset, "_global_indices_by_dataset", None)
    if not by_ep:
        raise ValueError(
            "viz loader needs a MultiDataset with _global_indices_by_dataset"
        )
    names = sorted(by_ep)
    if episodes == "all":
        chosen = names
    elif episodes == "auto":
        seen_tags = getattr(dataset, "_operator_seen", None)
        if seen_tags:
            chosen = []
            for want in (True, False):
                for n in names:
                    if bool(seen_tags.get(n, True)) == want:
                        chosen.append(n)
                        break
        else:
            chosen = names[:1]
    elif episodes == "per_group":
        by_group = getattr(dataset, "_episodes_by_group", None)
        if not by_group:
            raise ValueError(
                "viz episodes='per_group' needs a dataset with _episodes_by_group"
            )
        chosen = sorted(min(eps) for eps in by_group.values() if eps)
    else:
        chosen = list(episodes)
        missing = [n for n in chosen if n not in by_ep]
        if missing:
            raise ValueError(f"viz episodes not in the validation set: {missing}")
    if not (0.0 <= start_frac < 1.0):
        raise ValueError(f"start_frac must be in [0, 1), got {start_frac}")
    out = []
    for n in chosen:
        idxs = list(by_ep[n])
        if frames_per_episode is None:
            out.extend(idxs)
            continue
        k = int(frames_per_episode)
        start = min(int(len(idxs) * start_frac), max(len(idxs) - k, 0))
        out.extend(idxs[start : start + k])
    return out


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


def prompt_collate(prompts):
    """Pad a list of per-sample prompts (see ``prompt_dataset.py``) to the
    batch's longest ``P`` and build ``metadata.mask`` (``True`` = padded).
    Port of behavior_prompting's ``collate_prompts`` pair-prompting branch.

    Also used for the own-episode ``history`` payload, where ``P`` may be 0
    for every sample (episode start): ``pad_sequence`` then yields
    ``(B, 0, ...)`` tensors and a ``(B, 0)`` mask.
    """
    lengths = torch.tensor([int(p["length"]) for p in prompts], dtype=torch.long)
    obs = {
        key: pad_sequence([p["obs"][key] for p in prompts], batch_first=True)
        for key in prompts[0]["obs"]
    }
    action = pad_sequence([p["action"] for p in prompts], batch_first=True)
    P_max = int(action.shape[1])
    mask = torch.arange(P_max)[None, :] >= lengths[:, None]
    return {
        "obs": obs,
        "action": action,
        "metadata": {"mask": mask, "length": lengths},
    }


def annotation_collate(batch):
    """Collate that preserves variable-length list-valued keys (e.g. annotation_keys)
    and pads variable-length ``prompt`` / ``history`` dicts (whole-episode
    prompts and own-episode history, both padded by ``prompt_collate``)."""
    padded = {}
    for key in ("prompt", "history"):
        if key in batch[0]:
            if not all(key in sample for sample in batch):
                raise ValueError(
                    f"Every sample in a batch must carry a `{key}`, or none."
                )
            padded[key] = [sample.pop(key) for sample in batch]
    extracted = _extract_list_keys(batch)
    collated = default_collate(batch)
    collated.update(extracted)
    for key, payloads in padded.items():
        collated[key] = prompt_collate(payloads)
    return collated
