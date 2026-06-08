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
                policy on training data alongside the canonical validation
                pass. When set, ``val_dataloader()`` returns a list with the
                train-viz CombinedLoader at index 1 so Lightning populates
                ``dataloader_idx=1`` on validation_step.
            train_viz_dataloader_params: dict of per-dataset DataLoader kwargs
                for the train-viz loader (parallels valid_dataloader_params).

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
        self.train_viz_datasets = {
            k: v for k, v in (train_viz_datasets or {}).items() if v is not None
        }
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
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
    proprio_keys: list[str] | None = None,
    state_num_bins: int = 256,
    proprio: bool = False,
    embodiment_label: bool = False,
    control_mode: dict[str, str] | None = None,
):
    """Return a collate_fn closure that tokenizes the annotations field.

    Three orthogonal inclusion flags govern what gets spliced into the prompt:

      - ``proprio`` (bool): if True, append ``State: <bins>``. The per-sample
        proprio listed in ``proprio_keys`` is concatenated, clipped to
        ``[-1, 1]``, and discretized into ``state_num_bins`` bins (pi0.5 style;
        assumes upstream normalization).
      - ``embodiment_label`` (bool): if True, append ``Embodiment: <name>``.
      - ``control_mode`` (dict | None): if non-null, append ``Control mode:
        <descriptor>``. Keys are substrings matched against the (lowercased,
        ``_``→space) embodiment name; first match wins. Falls back to the
        built-in ``cam frame xyzypr [gripper] per arm`` defaults if no key
        matches.

    If any flag is active, the prompt is rendered as
    ``"Task: {prompt}, <blocks-in-order>;\\nAction: "`` (pi0.5 anchor).
    Otherwise the raw ``prompt`` is tokenized as-is.
    """
    from egomimic.rldb.embodiment.embodiment import get_embodiment

    tok = AutoTokenizer.from_pretrained(model_name)
    state_bin_edges = np.linspace(-1.0, 1.0, state_num_bins + 1)[:-1]
    # Default to the canonical concat key produced by the embodiment transform_list
    # (ConcatKeys with delete_old_keys=True removes the per-arm zarr keys).
    if proprio_keys is None:
        proprio_keys = ["observations.state.ee_pose"]
    else:
        proprio_keys = list(proprio_keys)

    def _embodiment_name(sample):
        eid = sample.get("embodiment")
        if eid is None:
            return None
        if isinstance(eid, torch.Tensor):
            eid = int(eid.item())
        elif isinstance(eid, np.ndarray):
            eid = int(eid.item())
        else:
            eid = int(eid)
        name = get_embodiment(eid)
        if name is None:
            return None
        return name.lower().replace("_", " ")

    def _control_mode_for(emb_name):
        if control_mode and emb_name is not None:
            for key, val in control_mode.items():
                if key.lower() in emb_name:
                    return val
        if emb_name is not None and "aria" in emb_name:
            return "cam frame xyzypr per arm"
        return "cam frame xyzypr gripper per arm"

    def _discretize_sample_state(sample):
        if not proprio_keys:
            return None
        parts = []
        for k in proprio_keys:
            if k not in sample:
                continue
            v = sample[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            else:
                v = np.asarray(v)
            v = np.asarray(v, dtype=np.float32)
            # Use the most recent timestep if proprio carries a time axis.
            while v.ndim > 1:
                v = v[-1]
            parts.append(v.reshape(-1))
        if not parts:
            return None
        state = np.concatenate(parts, axis=-1)
        state = np.clip(state, -1.0, 1.0)
        bins = np.digitize(state, bins=state_bin_edges) - 1
        return " ".join(map(str, bins.tolist()))

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
                    sampled_prompt = default_prompt
                elif sampling_mode == "random":
                    sampled_prompt = sample[random.randint(0, len(sample) - 1)]
                elif sampling_mode == "first":
                    sampled_prompt = sample[0]
                prompts.append(sampled_prompt)

        any_block_active = proprio or embodiment_label or bool(control_mode)
        if any_block_active:
            spliced = []
            for i, prompt in enumerate(prompts):
                emb_name = (
                    _embodiment_name(batch[i])
                    if (embodiment_label or control_mode)
                    else None
                )
                blocks = [f"Task: {prompt}"]
                if embodiment_label and emb_name:
                    blocks.append(f"Embodiment: {emb_name}")
                if control_mode:
                    blocks.append(f"Control mode: {_control_mode_for(emb_name)}")
                if proprio:
                    state_str = _discretize_sample_state(batch[i])
                    if state_str is not None:
                        blocks.append(f"State: {state_str}")
                spliced.append(", ".join(blocks) + ";\nAction: ")
            prompts = spliced

        list_keys = _extract_list_keys(batch)

        enc = tok(
            prompts,
            padding="max_length" if max_length is not None else "longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        collated = default_collate(batch)
        collated["sampled_prompt"] = prompts
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


class RiclDataModuleWrapper(MultiDataModuleWrapper):
    """``MultiDataModuleWrapper`` that attaches retrieved in-context demos (RICL).

    On top of the base behaviour it (1) wraps every query dataset in
    :class:`egomimic.ricl.data.RiclQueryDataset` so samples expose ``frame_idx``,
    and (2) swaps the collate for one that, per query sample, reads a precomputed
    retrieval cache and loads its top-k bank frames into ``ricl_*`` batch keys
    consumed by :class:`egomimic.algo.pi_ricl.PIRicl`.

    Build the cache first with ``python -m egomimic.ricl.retrieval`` (see
    ``egomimic/ricl/README.md``). Samples whose episode isn't in the cache fall
    back to the zero-context behaviour (all retrieved slots masked out).

    Args (beyond the base four):
        retrieval_cache_dir: dir written by ``RetrievalCache.save``.
        num_retrieved_observations: k.
        bank_zarr_root: ``{hash}`` template -> each bank episode's zarr store.
        bank_converter: action converter applied to retrieved actions (-> 32-D),
            e.g. ``HumanBimanualCartesianEuler`` for an aria bank.
    """

    def __init__(
        self,
        train_datasets: dict,
        valid_datasets: dict,
        train_dataloader_params: dict,
        valid_dataloader_params: dict,
        *,
        retrieval_cache_dir: str,
        num_retrieved_observations: int = 4,
        bank_zarr_root: str | None = None,
        bank_converter=None,
        image_hw=(224, 224),
        action_horizon: int = 1,
        state_dim: int = 32,
        action_dim: int = 32,
    ):
        super().__init__(
            train_datasets,
            valid_datasets,
            train_dataloader_params,
            valid_dataloader_params,
        )
        from egomimic.ricl.data import (
            RiclQueryDataset,
            ZarrBankFrameProvider,
            build_ricl_collate,
        )
        from egomimic.ricl.retrieval import RetrievalCache

        if bank_zarr_root is None:
            raise ValueError(
                "RiclDataModuleWrapper requires bank_zarr_root ('{hash}' template "
                "to per-episode bank zarr stores) to load retrieved frames."
            )

        cache = RetrievalCache.load(retrieval_cache_dir)
        provider = ZarrBankFrameProvider(
            resolve_store=lambda h: bank_zarr_root.format(hash=h),
            converter=bank_converter,
            action_horizon=action_horizon,
        )
        # Surface frame_idx on query samples (retrieval cache is per-frame).
        self.train_datasets = {
            k: RiclQueryDataset(v) for k, v in self.train_datasets.items()
        }
        self.valid_datasets = {
            k: RiclQueryDataset(v) for k, v in self.valid_datasets.items()
        }
        self.collate_fn = build_ricl_collate(
            cache,
            provider,
            k=num_retrieved_observations,
            base_collate=annotation_collate,
            image_hw=tuple(image_hw),
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
        )
        logger.info(
            "RiclDataModuleWrapper: k=%d, cache=%s, %d cached query episodes",
            num_retrieved_observations,
            retrieval_cache_dir,
            len(cache.query_hashes),
        )
