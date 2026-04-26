import logging
import random
from typing import Literal

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from termcolor import cprint
from torch.utils.data import DataLoader, default_collate

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - tokenizer only needed on request
    AutoTokenizer = None

logger = logging.getLogger(__name__)


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
        collate_max_length=128,
        model_name="google/paligemma-3b-mix-224",
        sampling_mode: Literal["first", "random"] = "random",
        annotation_key=None,
        use_tokenizer=False,
        default_prompt="",
    ):
        """
        Args:
            train_datasets: dictionary of train datasets
            valid_datasets: dictionary of valid datasets
            train_dataloader_params: dictionary of train dataloader parameters
            valid_dataloader_params: dictionary of valid dataloader parameters
            model_name: name of the model to use for the tokenizer
            sampling_mode: "first" to sample the first prompt from the list of prompts, "random" to sample a random prompt from the list of prompts
            annotation_key: key of the annotation to use for the collate function
            use_tokenizer: whether to use the tokenizer to tokenize the prompts
            default_prompt: default prompt to use if the annotation key is not found
            collate_max_length: maximum length of the tokenized prompts
        """
        super().__init__()
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        if use_tokenizer:
            self.collate_fn = build_tokenized_collate(
                max_length=collate_max_length,
                model_name=model_name,
                sampling_mode=sampling_mode,
                annotation_key=annotation_key,
                default_prompt=default_prompt,
            )
        else:
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

    def val_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.valid_datasets.items():
            dataset_params = self.valid_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config valid_dataloader_params."
                )
            iterables[dataset_name] = DataLoader(
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
        collate_max_length=128,
        model_name="google/paligemma-3b-mix-224",
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
            dataset=self.train_dataset1,
            collate_fn=self.collate_fn,
            **self.train_dataloader_params,
        )
        new_dataloader2 = DataLoader(
            dataset=self.train_dataset2,
            collate_fn=self.collate_fn,
            **self.train_dataloader_params,
        )
        return [new_dataloader1, new_dataloader2]

    ## to change embodiment sampling freq, just change the batch_size
    def val_dataloader(self):
        new_dataloader1 = DataLoader(
            dataset=self.valid_dataset1,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.valid_dataloader_params,
        )
        new_dataloader2 = DataLoader(
            dataset=self.valid_dataset2,
            collate_fn=self.collate_fn,
            shuffle=False,
            **self.valid_dataloader_params,
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
        collate_max_length=128,
        model_name="google/paligemma-3b-mix-224",
        sampling_mode: Literal["first", "random"] = "random",
        annotation_key=None,
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
            sampling_mode=sampling_mode,
            annotation_key=annotation_key,
        )

    def train_dataloader(self):
        new_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            **self.train_dataloader_params,
        )
        return new_dataloader

    def val_dataloader_1(self):
        new_dataloader = DataLoader(
            dataset=self.valid_dataset,
            collate_fn=self.collate_fn,
            **self.valid_dataloader_params,
        )
        return new_dataloader


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


def build_tokenized_collate(
    max_length=128,
    model_name="google/paligemma-3b-mix-224",
    sampling_mode: Literal["first", "random"] = "random",
    annotation_key="annotations",
    default_prompt="",
):
    """Return a collate_fn closure that tokenizes the annotations field."""

    if AutoTokenizer is None:
        raise ImportError("transformers is required when use_tokenizer=True")
    tok = AutoTokenizer.from_pretrained(model_name)

    def _collate(batch):
        if annotation_key is None:
            annotation = {}
            prompts = [default_prompt] * len(batch)
        else:
            if annotation_key not in batch[0]:
                raise KeyError(f"Annotation key {annotation_key} not found in batch")
            annotation = _extract_keys(batch, [annotation_key])
            prompts = []
            for sample in annotation[annotation_key]:
                if len(sample) == 0:
                    prompts.append(default_prompt)
                elif sampling_mode == "random":
                    prompts.append(sample[random.randint(0, len(sample) - 1)])
                elif sampling_mode == "first":
                    prompts.append(sample[0])

        list_keys = _extract_list_keys(batch)

        enc = tok(
            prompts,
            padding="max_length" if max_length is not None else "longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        collated = default_collate(batch)
        collated.update(annotation)
        collated.update(list_keys)
        attention_mask = enc["attention_mask"].bool()
        token_loss_mask = attention_mask.clone()
        token_loss_mask[:, -1] = False

        collated["tokenized_prompt"] = enc["input_ids"].requires_grad_(False)
        collated["tokenized_mask"] = attention_mask.requires_grad_(False)
        collated["token_loss_mask"] = token_loss_mask.requires_grad_(False)
        collated["token_ar_mask"] = attention_mask.clone().requires_grad_(False)
        return collated

    return _collate
