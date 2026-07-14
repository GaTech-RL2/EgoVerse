import logging
import random
from typing import Literal

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from termcolor import cprint
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer

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
        train_viz_datasets: dict | None = None,
        train_viz_dataloader_params: dict | None = None,
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
        self.train_viz_datasets = {
            k: v for k, v in (train_viz_datasets or {}).items() if v is not None
        }
        self.train_viz_dataloader_params = train_viz_dataloader_params or {}
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

    def val_dataloader(self):
        valid_loader = self._build_val_style_loader(
            self.valid_datasets, self.valid_dataloader_params, kind="valid"
        )
        if not self.train_viz_datasets:
            return valid_loader
        # When train_viz_datasets is configured, return a list so Lightning
        # populates dataloader_idx (0=valid, 1=train_viz) and ModelWrapper can
        # dispatch to self.train_viz_evaluator.
        train_viz_loader = self._build_val_style_loader(
            self.train_viz_datasets,
            self.train_viz_dataloader_params,
            kind="train_viz",
        )
        return [valid_loader, train_viz_loader]




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


# Target spatial size (H, W) for all camera images before collation. ABC-130k/eva
# episodes were captured at mixed widths (640 vs 848, both 480 tall), so a shuffled
# batch can mix resolutions, which default_collate cannot stack ("Trying to resize
# storage that is not resizable"). Resizing every camera image to one fixed size
# here makes the batch stackable; the model still resize_with_pad's to 224x224
# afterwards, so this only needs to unify shapes.
_IMAGE_TARGET_HW = (480, 640)


def _resize_image_keys(batch, size=_IMAGE_TARGET_HW):
    """In-place resize of every camera-image tensor in *batch* to a fixed (H, W).

    Camera images are identified by the substring ``"images"`` in the key
    (dataset-style keymaps, ``observations.images.*``) or the ``"_rgb"`` suffix
    (PI/PaliGemma-style keymaps, ``base_0_rgb`` / ``*_wrist_0_rgb``). Each image
    is ``(C, H, W)`` or ``(T, C, H, W)``; only the trailing two spatial dims are
    resized (bilinear, per-channel — equivalent to resizing the image). Tensors
    already at the target size are skipped.
    """
    th, tw = size
    for sample in batch:
        for k in list(sample.keys()):
            if "images" not in k and not k.endswith("_rgb"):
                continue
            v = sample[k]
            if not isinstance(v, torch.Tensor) or v.ndim < 2:
                continue
            if v.shape[-2] == th and v.shape[-1] == tw:
                continue
            orig_dtype = v.dtype
            x = v.float()
            lead = x.shape[:-2]  # leading (non-spatial) dims, e.g. (C,) or (T, C)
            x4 = x.reshape(-1, 1, x.shape[-2], x.shape[-1])  # (N, 1, H, W)
            x4 = torch.nn.functional.interpolate(
                x4, size=(th, tw), mode="bilinear", align_corners=False
            )
            sample[k] = x4.reshape(*lead, th, tw).to(orig_dtype)


def annotation_collate(batch):
    """Collate that preserves variable-length list-valued keys (e.g. annotation_keys)."""
    extracted = _extract_list_keys(batch)
    _resize_image_keys(batch)
    collated = default_collate(batch)
    collated.update(extracted)
    return collated


