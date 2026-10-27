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
from concurrent.futures import ThreadPoolExecutor
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


class GpuResidentBatches:
    """Training batches built on the GPU (data.gpu_resident=true, 2026-10-16). Every column of the train rows is copied to
    the GPU once; a batch equals default_collate([dataset[i] for i in idx]) (verified 10-16, 0 mismatches), built by
    indexing, with the delta_timestamps windows (the 64-step action chunk) and their pad masks from episode-bound index
    arithmetic as in LeRobot's _get_query_indices. No loader workers, so the training process keeps the job's CPUs.
    Only for one point-cloud RLDBDataset without key maps, task strings, slow-down, annotations or sampling (leaf())."""

    def __init__(self, dataset, batch_size, drop_last=False, host=False, balanced=False, row_weights=None, obs_delay=0):
        ds = self.leaf(dataset)
        self.n, self.bs, self.drop_last = len(dataset), int(batch_size), bool(drop_last)   # len = rows - 1 (LeRobot num_frames)
        self.dev = torch.device("cuda", torch.cuda.current_device())
        # host=True (data.gpu_resident=host, 10-17: 60 Hz releases outgrow the GPU): the columns stay in host memory and
        # each batch is gathered there into pinned memory (one batch ahead, on a thread), then copied to the GPU
        self.store = torch.device("cpu") if host else self.dev
        hf = ds.hf_dataset
        if host:
            self.cols = {k: self._load(hf, k) for k in hf.column_names}
        else:
            self.cols = {
                k: torch.from_numpy(np.ascontiguousarray(hf.with_format("numpy", columns=[k])[:][k])).to(self.dev)
                for k in hf.column_names
            }
        ep = self.cols["episode_index"].cpu().numpy()
        if ds.episodes is not None:
            pos_of = {int(e): i for i, e in enumerate(ds.episodes)}
            ep = np.array([pos_of[int(e)] for e in ep])
        frm, to = np.asarray(ds.episode_data_index["from"]), np.asarray(ds.episode_data_index["to"])
        self.start, self.end = torch.from_numpy(frm[ep]).to(self.store), torch.from_numpy(to[ep]).to(self.store)
        # data.episode_balanced=true (10-20 add-on mixture): rows drawn with weight 1 / (episode length), with replacement,
        # so every train episode is equally likely (equal episode counts per source -> 50/50 by source); default = randperm
        self.w = (1.0 / (self.end - self.start).double())[: self.n].float() if balanced else None
        if balanced:
            cprint(f"gpu_resident: episode-balanced sampling over {self.n} rows (weights 1/episode length)", "green")
        # data.row_weights=<file.npy> (10-20, phase-weighted sampling): a float weight per release row, indexed by the release's
        # global `index` column (e.g. manipulation 1.0, navigation 0.5). Rows are drawn with probability ~ weight (x 1/episode
        # length when episode_balanced) and an epoch draws sum(weights) rows, so a weight-1 row is still drawn about once per
        # epoch and the epoch gets shorter by the down-weighted share. Default None = unchanged behaviour.
        self.n_draw = self.n
        if row_weights is not None:
            W = np.load(row_weights).astype(np.float64)
            gi = self.cols["index"].cpu().numpy().astype(np.int64)[: self.n]
            assert gi.max() < len(W), f"row_weights {row_weights}: {len(W)} weights, release index up to {gi.max()}"
            pw = torch.from_numpy(W[gi]).to(self.store)
            assert bool((pw >= 0).all()) and float(pw.sum()) > 0, "row_weights must be >= 0 with a positive sum"
            self.w = (pw if self.w is None else pw * self.w.to(pw.dtype)).float()
            self.n_draw = max(1, int(round(float(pw.sum()))))
            cprint(f"gpu_resident: row weights {row_weights}: {self.n_draw} draws per epoch over {self.n} rows "
                   f"({100.0 * self.n_draw / self.n:.1f} %), weight min {float(pw.min()):.2f} max {float(pw.max()):.2f}", "green")
        self.deltas = {k: torch.tensor(v, device=self.store) for k, v in (ds.delta_indices or {}).items()}
        self.emb = int(ds.embodiment)
        # multi-frame views (RLDBDataset.frame_views, 10-17): the per-row T_F_glass on the GPU, applied per batch
        self.fv = {k: torch.from_numpy(ds._fv_T[f]).to(self.dev) for k, f in getattr(ds, "frame_views", {}).items()}
        # data.obs_delay=N (10-23, latency augmentation): every obs.* column of a sample comes from d ~ U{0..N} rows before its
        # start row (clamped to the episode start); the action chunk and the row weight stay on the start row. Default 0 = off.
        self.obs_delay = int(obs_delay or 0)
        assert not (self.obs_delay and self.fv), "data.obs_delay does not support frame views"
        if self.obs_delay:
            cprint(f"gpu_resident: obs delay 0..{self.obs_delay} rows on every obs.* column", "green")
        cprint(f"gpu_resident: {self.n} rows on {self.store}, "
               f"{sum(t.numel() * t.element_size() for t in self.cols.values()) / 1e9:.1f} GB", "green")

    @staticmethod
    def _load(hf, k, chunk=16384):
        """one column as a host tensor, filled chunk by chunk (no full-size temporary copies)"""
        col = hf.with_format("numpy", columns=[k])
        first = np.ascontiguousarray(col[0:1][k])
        out = torch.empty((len(hf),) + first.shape[1:], dtype=torch.from_numpy(first).dtype)
        for s in range(0, len(hf), chunk):
            out[s:s + chunk] = torch.from_numpy(np.ascontiguousarray(col[s:s + chunk][k]))
        return out

    @staticmethod
    def leaf(dataset):
        """the single RLDBDataset under (nested) MultiRLDBDatasets, or None if its items cannot be reproduced here"""
        d = dataset
        while not isinstance(d, RLDBDataset):
            if getattr(d, "key_map", None) or len(getattr(d, "datasets", {})) != 1:
                return None
            d = next(iter(d.datasets.values()))
        plain = (len(d.meta.camera_keys) == 0 and d.sampled_indices is None and not d.use_task_string
                 and d.slow_down_factor == 1.0 and getattr(d, "annotation_df", None) is None and not d.cache_in_memory)
        return d if plain else None

    def __len__(self):
        return self.n_draw // self.bs if self.drop_last else -(-self.n_draw // self.bs)

    def __iter__(self):
        if self.w is not None:
            perm = torch.multinomial(self.w, self.n_draw, replacement=True)
        else:
            perm = torch.randperm(self.n, device=self.store)
        if self.store.type != "cpu":
            for s in range(0, len(self) * self.bs, self.bs):
                yield self.batch(perm[s:s + self.bs])
            return
        idxs = [perm[s:s + self.bs] for s in range(0, len(self) * self.bs, self.bs)]
        with ThreadPoolExecutor(1) as ex:
            nxt = ex.submit(self.gather, idxs[0]) if idxs else None
            for i, idx in enumerate(idxs):
                out = nxt.result()
                if i + 1 < len(idxs):
                    nxt = ex.submit(self.gather, idxs[i + 1])
                yield self.finish(out, idx)

    def batch(self, idx):
        return self.finish(self.gather(idx), idx)

    def _take(self, col, idx):
        """col[idx]; on the host straight into pinned memory, for a non-blocking copy to the GPU"""
        if self.store.type != "cpu":
            return col[idx]
        buf = torch.empty((idx.numel(),) + col.shape[1:], dtype=col.dtype, pin_memory=True)
        torch.index_select(col, 0, idx.reshape(-1), out=buf)
        return buf.reshape(idx.shape + col.shape[1:])

    def gather(self, idx):
        oidx = idx
        if self.obs_delay:
            lag = torch.randint(0, self.obs_delay + 1, idx.shape, device=idx.device)
            oidx = torch.maximum(idx - lag, self.start[idx])
        out = {k: self._take(v, oidx if k.startswith("obs.") else idx) for k, v in self.cols.items() if k not in self.deltas}
        for k, d in self.deltas.items():
            q = (oidx if k.startswith("obs.") else idx)[:, None] + d[None, :]
            s, e = self.start[idx][:, None], self.end[idx][:, None]
            out[k] = self._take(self.cols[k], torch.minimum(torch.maximum(q, s), e - 1))
            out[k + "_is_pad"] = (q < s) | (q >= e)
        return out

    def finish(self, out, idx):
        if self.store.type == "cpu":
            out = {k: v.to(self.dev, non_blocking=True) for k, v in out.items()}
            idx = idx.to(self.dev)
        out["annotations"] = [""] * len(idx)
        out["metadata.embodiment"] = torch.full((len(idx),), self.emb, dtype=torch.int64, device=self.dev)
        if self.fv:
            from egomimic.utils.rby1_frames import apply_frame

            for k, T in self.fv.items():
                out[k] = apply_frame(out["obs.aria_pcdc"], T[idx])
        return out


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
        gpu_resident = False,
        episode_balanced = False,
        row_weights = None,
        obs_delay = 0,
    ):
        super().__init__()
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        # training batches built on the GPU (GpuResidentBatches) when every train dataset allows it; plain collate only
        self.gpu_resident = bool(gpu_resident) and not use_tokenizer and not zero_obs
        self.gpu_host = gpu_resident == "host"   # data.gpu_resident=host: the same batches from pinned host memory (10-17)
        self.episode_balanced = bool(episode_balanced)   # data.episode_balanced=true: GpuResidentBatches only (10-20)
        self.row_weights = row_weights   # data.row_weights=<file.npy>: phase-weighted sampling, GpuResidentBatches only (10-20)
        self.obs_delay = int(obs_delay or 0)   # data.obs_delay=N: latency augmentation, GpuResidentBatches only (10-23)
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
            if self.gpu_resident and torch.cuda.is_available() and GpuResidentBatches.leaf(dataset) is not None:
                iterables[dataset.embodiment] = GpuResidentBatches(
                    dataset, dataset_params.get("batch_size", 1), dataset_params.get("drop_last", False), host=self.gpu_host,
                    balanced=self.episode_balanced, row_weights=self.row_weights, obs_delay=self.obs_delay,
                )
                continue
            if self.episode_balanced or self.row_weights or self.obs_delay:
                raise ValueError("data.episode_balanced / data.row_weights / data.obs_delay need the gpu_resident path (data.gpu_resident=true or host)")
            if self.gpu_resident:
                cprint(f"gpu_resident: {dataset_name} needs the DataLoader path (images, several datasets, key maps, ...)", "yellow")
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
