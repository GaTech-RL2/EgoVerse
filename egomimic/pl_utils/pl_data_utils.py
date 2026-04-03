from torch.utils.data import DataLoader, random_split, default_collate
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning import LightningDataModule
from transformers import AutoTokenizer
from egomimic.utils.egomimicUtils import nds
import json
import os
import logging
from egomimic.rldb.utils import RLDBDataset
from termcolor import cprint
import torch
import numpy as np

logger = logging.getLogger(__name__)


def zero_obs_batch(batch, use_numpy=False):
    """
    In-place replace image observation tensors with zeros_like.
    Observation keys are zeroed only if they follow the LeRobot ``obs.`` naming
    convention and contain ``image`` in the key name.
    Args:
        batch: dict from collate (keys like obs.aria_image, obs.robot0_joint_pos, etc.)
        use_numpy: if True, convert to np and use np.zeros_like; else use torch.zeros_like
    """
    for key in list(batch.keys()):
        if not key.startswith("obs.") or "image" not in key:
            continue
        val = batch[key]
        if use_numpy:
            arr = val.numpy() if torch.is_tensor(val) else np.asarray(val)
            batch[key] = np.zeros_like(arr)
        else:
            t = val if torch.is_tensor(val) else torch.from_numpy(np.asarray(val))
            batch[key] = torch.zeros_like(t)


def _wrap_collate_zero_obs(base_collate, use_numpy=False):
    """Wrap a collate_fn to zero out image obs.* keys after collation."""

    def _collate(batch):
        out = base_collate(batch)
        if isinstance(out, dict):
            zero_obs_batch(out, use_numpy=use_numpy)
        return out

    return _collate


class RLDBModule(LightningDataModule):
    """
    Deprecated and is not supported by trainHydra.py
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_kwargs,
        valid_dataloader_kwargs,
    ):
        cprint(
            "RLDBModule is deprecated and is not supported by trainHydra.py. Use MultiDataModuleWrapper instead",
            "red",
        )

        super().__init__()
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.valid_dataloader_kwargs = valid_dataloader_kwargs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, **self.train_dataloader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, shuffle=False, **self.valid_dataloader_kwargs
        )


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
        collate_max_length = 128,
        model_name = "google/paligemma-3b-mix-224",
        use_tokenizer = False,
        zero_obs = False,
        zero_obs_use_numpy = False,
    ):
        super().__init__()
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        if use_tokenizer:
            base_collate = build_tokenized_collate(
                max_length=collate_max_length,
                model_name=model_name,
            )
        else:
            base_collate = default_collate
        if zero_obs:
            self.collate_fn = _wrap_collate_zero_obs(base_collate, use_numpy=zero_obs_use_numpy)
        else:
            self.collate_fn = base_collate
        
    def train_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.train_datasets.items():
            dataset_params = self.train_dataloader_params.get(dataset_name, {})
            iterables[dataset.embodiment] = DataLoader(
                dataset,
                shuffle=True,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")

    def val_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.valid_datasets.items():
            dataset_params = self.valid_dataloader_params.get(dataset_name, {})
            iterables[dataset.embodiment] = DataLoader(
                dataset,
                shuffle=False,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")


class DualDataModuleWrapper(LightningDataModule):
    """
    Same as DataModuleWrapper but there are two train datasets and two valid datasets
    """

    """
    Deprecated and is not supported by trainHydra.py
    """

    def __init__(
        self,
        train_dataset1,
        valid_dataset1,
        train_dataset2,
        valid_dataset2,
        train_dataloader_params,
        valid_dataloader_params,
        collate_max_length = 128,
        model_name = "google/paligemma-3b-mix-224",
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        cprint(
            "DualDataModuleWrapper is deprecated and is not supported by trainHydra.py. Use MultiDataModuleWrapper instead",
            "red",
        )

        super().__init__()
        self.train_dataset1 = train_dataset1
        self.valid_dataset1 = valid_dataset1
        self.train_dataset2 = train_dataset2
        self.valid_dataset2 = valid_dataset2
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.collate_fn = build_tokenized_collate(
            max_length=collate_max_length,
            model_name=model_name,
        )

    def train_dataloader(self):
        new_dataloader1 = DataLoader(
            dataset=self.train_dataset1, collate_fn=self.collate_fn, **self.train_dataloader_params
        )
        new_dataloader2 = DataLoader(
            dataset=self.train_dataset2, collate_fn=self.collate_fn, **self.train_dataloader_params
        )
        return [new_dataloader1, new_dataloader2]

    ## to change embodiment sampling freq, just change the batch_size
    def val_dataloader(self):
        new_dataloader1 = DataLoader(
            dataset=self.valid_dataset1, collate_fn=self.collate_fn, shuffle=False, **self.valid_dataloader_params
        )
        new_dataloader2 = DataLoader(
            dataset=self.valid_dataset2, collate_fn=self.collate_fn, shuffle=False, **self.valid_dataloader_params
        )
        return [new_dataloader1, new_dataloader2]

    # def val_dataloader(self):
    #     new_dataloader1 = DataLoader(dataset=self.valid_dataset1, **self.valid_dataloader_params)
    #     new_dataloader2 = DataLoader(dataset=self.valid_dataset2, **self.valid_dataloader_params)
    #     return [new_dataloader1, new_dataloader2]


class DataModuleWrapper(LightningDataModule):
    """
    Wrapper around a LightningDataModule that allows for the data loader to be refreshed
    constantly.
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_params,
        valid_dataloader_params,
        collate_max_length = 128,
        model_name = "google/paligemma-3b-mix-224",
        
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.collate_fn = build_tokenized_collate(
            max_length=collate_max_length,
            model_name=model_name,
        )

    def train_dataloader(self):
        new_dataloader = DataLoader(
            dataset=self.train_dataset, collate_fn=self.collate_fn, **self.train_dataloader_params
        )
        return new_dataloader

    def val_dataloader_1(self):
        new_dataloader = DataLoader(
            dataset=self.valid_dataset, collate_fn=self.collate_fn, **self.valid_dataloader_params
        )
        return new_dataloader


def build_tokenized_collate(max_length=128, model_name="google/paligemma-3b-mix-224"):
    """Return a collate_fn closure that tokenizes the annotations field."""

    tok = AutoTokenizer.from_pretrained(model_name)

    def _collate(batch):
        if "annotations" not in batch[0]:
            return default_collate(batch)

        # Ensure annotations are always strings (no None) before collating/tokenizing
        sanitized_batch = []
        for sample in batch:
            sample_copy = dict(sample)
            ann = sample_copy.get("annotations")
            sample_copy["annotations"] = "" if ann is None else str(ann)
            sanitized_batch.append(sample_copy)

        prompts = [sample["annotations"] for sample in sanitized_batch]

        enc = tok(
            prompts,
            padding="max_length" if max_length is not None else "longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        collated = default_collate(sanitized_batch)
        attention_mask = enc["attention_mask"].bool()
        token_loss_mask = attention_mask.clone()
        token_loss_mask[:, -1] = False

        collated["tokenized_prompt"] = enc["input_ids"].requires_grad_(False)
        collated["tokenized_mask"] = attention_mask.requires_grad_(False)
        collated["token_loss_mask"] = token_loss_mask.requires_grad_(False)
        collated["token_ar_mask"] = attention_mask.clone().requires_grad_(False)
        return collated

    return _collate
