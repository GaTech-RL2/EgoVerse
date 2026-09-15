import logging

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, default_collate

logger = logging.getLogger(__name__)

# Val heads that can carry a metric loader, in val_dataloader() order.
VAL_HEADS = ("valid", "train_viz", "unseen_op_valid")
# Suffix of the video-only companion loader of a head (``valid`` ->
# ``valid_video``). ModelWrapper.validation_step splits on it.
VIDEO_SUFFIX = "_video"


def video_loader_name(head: str) -> str:
    return f"{head}{VIDEO_SUFFIX}"


def head_of_loader(name: str) -> tuple[str, bool]:
    """``("valid", False)`` for a metric loader, ``("valid", True)`` for its
    video-only companion."""
    if name.endswith(VIDEO_SUFFIX):
        return name[: -len(VIDEO_SUFFIX)], True
    return name, False


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
        train_viz_datasets: dict | None = None,
        train_viz_dataloader_params: dict | None = None,
        unseen_op_valid_datasets: dict | None = None,
        unseen_op_valid_dataloader_params: dict | None = None,
        valid_prefix: str | None = None,
        held_out_operators: list | None = None,
        video_datasets: dict | None = None,
        metric_frames_per_episode: dict | None = None,
        video_episodes: dict | None = None,
    ):
        """
        Args:
            train_datasets: dictionary of train datasets
            valid_datasets: dictionary of valid datasets
            train_dataloader_params: dictionary of train dataloader parameters
            valid_dataloader_params: dictionary of valid dataloader parameters
            train_viz_datasets: optional dict of datasets iterated like a
                second val loader. Used by TrainVizEvalVideo to visualize the
                policy on training data alongside the canonical validation.
            train_viz_dataloader_params: dict of per-dataset DataLoader kwargs
                for the train_viz loader.
            unseen_op_valid_datasets: optional dict of datasets iterated as a
                third val loader, evaluated by a prefixed copy of the canonical
                evaluator (metrics ``unseen_op_valid/``). The held-out-operator split configs put the
                unseen operators here, so ``valid_datasets`` can hold the seen
                operators' held-out episodes and ``train_viz`` the train split.
            unseen_op_valid_dataloader_params: dict of per-dataset DataLoader
                kwargs for the unseen_op_valid loader.
            valid_prefix: config-only. When set, trainHydra wraps the canonical
                evaluator so the valid loader logs ``<prefix>/...`` and writes
                ``videos_<prefix>/`` (``seen_op_valid`` in the opsplit configs).
            held_out_operators: config-only. The held-out-operator split
                configs (data/mecka_fold_*_opsplit_*.yaml) keep the operator
                id list once at the data-config root and interpolate it from
                the train/valid filter lambdas; hydra.instantiate(cfg.data)
                forwards every root key here, so it must be accepted. Kept as
                an attribute for provenance only.
            video_datasets: optional ``{head: {dataset_name: dataset}}`` for the
                video-ONLY companion loaders (``<head>_video``). Built by
                trainHydra from ``data.video_episodes``: the head's own split
                restricted to the pinned episode hashes, contiguous and
                unsubsampled so the overlay video stays watchable while the
                metric loader is per-episode subsampled. A head with no pins
                gets no video loader (today's behaviour). The video loader
                reuses that head's ``*_dataloader_params`` (unshuffled).
            metric_frames_per_episode: config-only ``{head: K}``. trainHydra
                wraps each val head's dataset in ``EvenStrideDataset(base,
                frames_per_episode=K)`` so the ``limit_val_batches`` window
                covers every episode of the split instead of the leading
                hash-sorted slice. Forwarded here by hydra.instantiate (it
                rejects undeclared root keys); kept for provenance only.
            video_episodes: config-only ``{head: [episode_hash, ...]}`` -- the
                pinned video episodes, one per operator per head. Provenance
                only; trainHydra is what reads it.

        Tokenization (sampling a prompt from per-sample annotation lists,
        splicing in embodiment / control-mode / proprio blocks, and running
        the HF tokenizer) lives on the algo side now; see
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
        self.train_viz_datasets = {
            k: v for k, v in (train_viz_datasets or {}).items() if v is not None
        }
        self.train_viz_dataloader_params = train_viz_dataloader_params or {}
        self.unseen_op_valid_datasets = {
            k: v for k, v in (unseen_op_valid_datasets or {}).items() if v is not None
        }
        self.unseen_op_valid_dataloader_params = unseen_op_valid_dataloader_params or {}
        self.valid_prefix = valid_prefix
        self.held_out_operators = list(held_out_operators or [])
        # {head: {dataset_name: dataset}}; heads with no pinned episodes are
        # dropped so `val_loader_names()` only grows for configs that ask for it.
        self.video_datasets = {
            head: {k: v for k, v in (datasets or {}).items() if v is not None}
            for head, datasets in (video_datasets or {}).items()
        }
        self.video_datasets = {h: d for h, d in self.video_datasets.items() if d}
        self.metric_frames_per_episode = dict(metric_frames_per_episode or {})
        self.video_episodes = dict(video_episodes or {})
        self.collate_fn = annotation_collate

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

    def _build_val_style_loader(self, datasets: dict, params: dict, kind: str):
        iterables = dict()
        for dataset_name, dataset in datasets.items():
            dataset_params = params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config {kind}_dataloader_params."
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

    def _metric_sources(self) -> dict:
        return {
            "valid": (self.valid_datasets, self.valid_dataloader_params),
            "train_viz": (self.train_viz_datasets, self.train_viz_dataloader_params),
            "unseen_op_valid": (
                self.unseen_op_valid_datasets,
                self.unseen_op_valid_dataloader_params,
            ),
        }

    def val_loader_names(self) -> list[str]:
        """Names of the val loaders in ``val_dataloader()`` order; position i is
        Lightning's ``dataloader_idx`` i. ModelWrapper dispatches on these.

        Metric loaders first, in the historical order, so a config with no
        pinned video episodes gets exactly today's list and today's indices;
        the video-only loaders are appended after them in the same head order.
        """
        metric = [
            name
            for name in VAL_HEADS
            if name == "valid" or self._metric_sources()[name][0]
        ]
        return metric + [
            video_loader_name(name) for name in metric if name in self.video_datasets
        ]

    def val_dataloader(self):
        sources = self._metric_sources()
        for head, datasets in self.video_datasets.items():
            # The video loader reuses its head's params (batch size / workers /
            # unshuffled); only the dataset differs.
            sources[video_loader_name(head)] = (datasets, sources[head][1])
        names = self.val_loader_names()
        loaders = [
            # kind names the *_dataloader_params block in the error message, and
            # a video loader borrows its head's block.
            self._build_val_style_loader(*sources[name], kind=head_of_loader(name)[0])
            for name in names
        ]
        if len(loaders) == 1:
            return loaders[0]
        # Several heads: return a list so Lightning populates dataloader_idx
        # (the position in val_loader_names()) and ModelWrapper can dispatch.
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
