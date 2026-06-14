"""Replaces zarr per-frame DataLoader with pre-materialized tensor dataset.
Monkey-patches MultiDataset._from_resolver to return FastHPTDataset.
Implements all interfaces the training code expects."""
import torch
import numpy as np
from torch.utils.data import Dataset

# Auto-cast fp32→fp64 for modules with fp64 params (harmless for fp32 runs)
_orig_module_call = torch.nn.Module.__call__
def _auto_cast_call(self, *args, **kwargs):
    try:
        p = next(self.parameters())
        if p.dtype == torch.float64:
            args = tuple(a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args)
            kwargs = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in kwargs.items()}
    except StopIteration:
        pass
    return _orig_module_call(self, *args, **kwargs)
torch.nn.Module.__call__ = _auto_cast_call

import egomimic.rldb.zarr.zarr_dataset_multi as zdm
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes import get_keymap
import os
import time

DATASET_PT = os.environ.get("DATASET_PT", "/coc/flash7/paphiwetsa3/datasets/new_circle_3_materialized/dataset.pt")


class FastHPTDataset(Dataset):
    """Pre-materialized tensor dataset with full MultiDataset interface."""

    def __init__(self, data_dict):
        self.images = data_dict["images"]
        self.states = data_dict["states"]
        self.actions = data_dict["actions"]
        self.n = data_dict["total_frames"]

        # Interface expected by populate_from_datasets / _iter_leaves
        self.embodiment = "pushshapes_sim"
        self.key_map = get_keymap(action_horizon=32)
        self.datasets = {"pushshapes_sim": self}
        self.embodiments = {get_embodiment_id("pushshapes_sim")}
        self.norm_stats = {}
        self.key_types = {}
        self.zarr_keys = {}
        self.shapes = {}

        # MultiDataset interface stubs
        self.skip_bounds_check = True
        self.index_map = None
        self._global_indices_by_dataset = {}

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        emb_id = get_embodiment_id("pushshapes_sim")
        return {
            "front_img_1": self.images[idx],
            "state_agent_obj": self.states[idx],
            "actions": self.actions[idx],
            "metadata.robot_name": emb_id,
            "embodiment": emb_id,
        }

    # --- MultiDataset interface methods ---
    def set_norm_stats_from(self, norm_stats):
        if hasattr(norm_stats, 'norm_stats'):
            self.norm_stats = norm_stats.norm_stats

    def populate_from_datasets(self, datasets=None):
        pass

    def set_norm_stats(self, *args, **kwargs):
        pass

    def normalize(self, data, embodiment):
        return data

    @staticmethod
    def _iter_leaves(ds):
        if isinstance(ds, FastHPTDataset):
            yield ds
        elif hasattr(ds, 'datasets'):
            for child in ds.datasets.values():
                yield from FastHPTDataset._iter_leaves(child)
        else:
            yield ds

    def infer_shapes_from_batch(self, sample):
        pass

    def infer_norm_from_dataset(self, *args, **kwargs):
        pass

    def to_state(self):
        return {"stats": {}, "embodiments": list(self.embodiments)}

    def cache_stats(self, *args, **kwargs):
        pass


# Global singleton
_fast_dataset = None


@classmethod
def _fast_from_resolver(cls, resolver=None, **kwargs):
    global _fast_dataset
    if _fast_dataset is None:
        print(f"[fast_dataloader] Loading {DATASET_PT}...", flush=True)
        t0 = time.time()
        data = torch.load(DATASET_PT, weights_only=True)
        dt = time.time() - t0
        print(f"[fast_dataloader] Loaded {data['total_frames']} frames in {dt:.1f}s", flush=True)
        _fast_dataset = FastHPTDataset(data)
    return _fast_dataset


zdm.MultiDataset._from_resolver = _fast_from_resolver

print("[fast_dataloader] Patched for tensor-backed dataset", flush=True)

import runpy
runpy.run_module("egomimic.trainHydra", run_name="__main__")
