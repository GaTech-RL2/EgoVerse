import ast
import copy
import json
import logging
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42):
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


class DataSchematic(object):
    def __init__(self, schematic_dict, norm_mode="zscore"):
        """
        Initialize with a schematic dictionary and create a DataFrame.

        Args:
            schematic_dict:
                {embodiment_name}:
                    front_img_1_line:
                        key_type: camera_keys
                        zarr_key: front_img_1
                    right_wrist_img:
                        key_type: camera_keys
                        zarr_key: right_wrist_img
                    joint_positions:
                        key_type: proprio_keys
                        zarr_key: joint_positions
                    actions_joints_act:
                        key_type: action_keys
                        zarr_key: actions_joints
                    .
                    .
                    .
                    .
            viz_img_key (dict): Mapping from embodiment name to visualization image key.
            norm_mode (str): Normalization mode ("zscore", "minmax", or "quantile").

        Attributes:
            df (pd.DataFrame): Columns include "key_name", "key_type", "zarr_key",
                "shape", and "embodiment".
        """

        rows = []
        self.schematic_dict = copy.deepcopy(schematic_dict)
        self.embodiments = set()

        for embodiment, schematic in schematic_dict.items():
            embodiment_id = get_embodiment_id(embodiment)
            self.embodiments.add(embodiment_id)
            for key_name, key_info in schematic.items():
                rows.append(
                    {
                        "key_name": key_name,
                        "key_type": key_info["key_type"],
                        "zarr_key": key_info["zarr_key"],
                        "shape": None,
                        "embodiment": embodiment_id,
                    }
                )

        self.df = pd.DataFrame(rows)
        self.shapes_infered = False
        self.norm_mode = norm_mode
        self.norm_stats = {emb: {} for emb in self.embodiments}
        self._norm_run_metadata: dict[str, float | int | None] | None = None

    @staticmethod
    def _clone_norm_stats(norm_stats):
        out = {}
        for embodiment, embodiment_stats in (norm_stats or {}).items():
            out[embodiment] = {}
            for key, stats in embodiment_stats.items():
                out[embodiment][key] = {}
                for stat_name, value in stats.items():
                    if torch.is_tensor(value):
                        out[embodiment][key][stat_name] = value.detach().cpu().clone()
                    else:
                        out[embodiment][key][stat_name] = copy.deepcopy(value)
        return out

    def to_state(self):
        return {
            "schematic_dict": copy.deepcopy(self.schematic_dict),
            "norm_mode": self.norm_mode,
            "df_records": copy.deepcopy(self.df.to_dict("records")),
            "shapes_infered": self.shapes_infered,
            "norm_stats": self._clone_norm_stats(self.norm_stats),
        }

    @classmethod
    def from_state(cls, state):
        if state is None:
            raise ValueError("DataSchematic state must be provided for reconstruction.")

        schematic = cls(
            schematic_dict=copy.deepcopy(state["schematic_dict"]),
            norm_mode=state.get("norm_mode", "zscore"),
        )
        if "df_records" in state:
            schematic.df = pd.DataFrame(copy.deepcopy(state["df_records"]))
            schematic.embodiments = set(
                int(embodiment) for embodiment in schematic.df["embodiment"].unique()
            )
        schematic.shapes_infered = bool(state.get("shapes_infered", False))
        schematic.norm_stats = cls._clone_norm_stats(state.get("norm_stats", {}))
        for embodiment in schematic.embodiments:
            schematic.norm_stats.setdefault(embodiment, {})
        return schematic

    def zarr_key_to_keyname(self, zarr_key, embodiment):
        """
        Get the key name from the zarr key.
        Args:
            zarr_key (str): zarr key, e.g., "front_img_1".
            embodiment (int): Integer ID corresponding to an embodiment.

        Returns:
            str: Key name, e.g., "front_img_1".
        """
        df_filtered = self.df[
            (self.df["zarr_key"] == zarr_key) & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            return None

        return df_filtered["key_name"].item()

    def keyname_to_zarr_key(self, key_name, embodiment):
        """
        Get the zarr key from the key name.
        Args:
            key_name (str): Key name, e.g., "front_img_1_line".
            embodiment (int): Integer ID corresponding to an embodiment.

        Returns:
            str: zarr key, e.g., "front_img_1".
        """
        df_filtered = self.df[
            (self.df["key_name"] == key_name) & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            return None
        return df_filtered["zarr_key"].item()

    def infer_shapes_from_batch(self, batch):
        """
        Update shapes in the DataFrame based on a batch.

        Args:
            batch (dict): Maps key names (str) to tensors with shapes, e.g.,
                {"key": tensor of shape (3, 480, 640, 3)}.

        Updates:
            The 'shape' column in the DataFrame is updated to match the inferred shapes (stored as tuples).
        """
        for key, tensor in batch.items():
            if hasattr(tensor, "shape"):
                shape = tuple(tensor.shape)
            elif isinstance(tensor, int):
                shape = (1,)
            else:
                shape = None
            if key in self.df["zarr_key"].values:
                self.df.loc[self.df["zarr_key"] == key, "shape"] = str(shape)

        self.shapes_infered = True

    def infer_norm_from_dataset(
        self,
        dataset,
        dataset_name,
        sample_frac: float = 0.10,
        seed: int = 42,
        max_samples: int | None = None,
        batch_size: int = 512,
        num_workers: int = 4,
        precomputed_norm_path: str | None = None,
        checkpoint_fn=None,
    ):
        """
        Load or compute normalization statistics for an embodiment; does not write cache files.

        If ``precomputed_norm_path`` is set and contains ``norm_stats.json`` with stats for this
        embodiment, ``numpy`` float32 arrays are built from JSON numbers / nested lists under
        ``stats``. Otherwise stats are computed from ``dataset`` and stored the same way.

        Timing and sample count for the last run are stored in ``self._norm_run_metadata`` for
        :meth:`cache_stats`.

        Args:
            dataset: Dataset used to infer normalization stats (ignored when precomputed loads).
            dataset_name: Name or ID of the embodiment/dataset.
            sample_frac (float): Fraction of dataset elements to sample.
            seed (int): Random seed for sampling.
            max_samples (int | None): Optional upper bound on sampled elements.
            batch_size (int): Batch size used by the dataloader.
            num_workers (int): Number of dataloader workers.
            precomputed_norm_path (str | None): Directory containing ``norm_stats.json``.
        """
        embodiment = dataset_name
        if isinstance(embodiment, str):
            embodiment = get_embodiment_id(embodiment)

        norm_keys = []
        norm_keys.extend(self.keys_of_type("proprio_keys", embodiment))
        norm_keys.extend(self.keys_of_type("action_keys", embodiment))
        if len(norm_keys) == 0:
            logger.warning(
                f"[NormStats] No proprio/action keys for embodiment={embodiment}"
            )
            return

        if embodiment not in self.norm_stats:
            self.norm_stats[embodiment] = {}
        
        # Load precomputed norm stats if available
        if precomputed_norm_path is not None:
            if os.path.isdir(precomputed_norm_path):
                precomputed_file = os.path.join(precomputed_norm_path, "norm_stats.json")
            elif os.path.isfile(precomputed_norm_path):
                precomputed_file = precomputed_norm_path
            else:
                logger.warning(
                    f"[NormStats] precomputed_norm_path={precomputed_norm_path} is not a valid directory or file"
                )
                return
            if os.path.isfile(precomputed_file):
                with open(precomputed_file, "r") as f:
                    precomputed_norm_stats = json.load(f)
                    self.norm_stats[embodiment] = precomputed_norm_stats["stats"].get(str(embodiment), {})
                    self._norm_run_metadata = precomputed_norm_stats.get("norm_run_metadata", None)
                    logger.info(
                        f"[NormStats] Loaded precomputed stats for embodiment={embodiment} from {precomputed_file}"
                    )
                    return

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        N = len(dataset)
        if N <= 0:
            raise ValueError("Dataset is empty")

        n_samples = int(math.ceil(sample_frac * N))
        n_samples = max(1, min(n_samples, N))
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        logger.info(f"[NormStats] embodiment={embodiment} norm_keys={norm_keys}")
        logger.info(
            f"[NormStats] sampling {n_samples}/{N} (~{100 * sample_frac:.1f}%) indices"
        )

        loading_start_time = time.time()
        collected = self._collect_norm_samples(
            loader=loader,
            norm_keys=norm_keys,
            embodiment=embodiment,
            n_samples=n_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint_fn=checkpoint_fn,
        )
        del_keys = []
        for k in norm_keys:
            if len(collected[k]) == 0:
                del_keys.append(k)
        for k in del_keys:
            del collected[k]
            norm_keys.remove(k)

        loading_time = time.time() - loading_start_time

        computing_start_time = time.time()
        for k in norm_keys:
            if collected.get(k, None) is None:
                logger.warning(f"[NormStats] No data collected for key={k}")
                continue
            collected[k] = np.concatenate(collected[k], axis=0)

            X = collected[k]
            stats_np = self._compute_stats_for_array(X)
            self.norm_stats[embodiment][k] = {
                name: np.asarray(arr, dtype=np.float32)
                for name, arr in stats_np.items()
            }

            logger.info(
                f"[NormStats] key={k} samples={X.shape[0]} stat_shape={stats_np['mean'].shape}"
            )

        computing_time = time.time() - computing_start_time
        self._norm_run_metadata = {
            "loading_time": loading_time,
            "computing_time": computing_time,
            "frames": n_samples,
        }

        logger.info(
            f"[NormStats] Finished norm inference, loading_time={loading_time:.2f}s, computing_time={computing_time:.2f}s"
        )

    def _collect_norm_samples(
        self, loader, norm_keys, embodiment, n_samples: int, batch_size: int, num_workers: int,
        checkpoint_fn=None,
    ):
        collected = {k: [] for k in norm_keys}
        cur_num_samples = 0
        _next_ckpt_pct = 10
        logger.info(
            f"[NormStats] Starting to load data for norm inference with batch_size={batch_size} and num_workers={num_workers}"
        )
        with tqdm(total=n_samples, unit="sample") as pbar:
            for batch in loader:
                remaining = n_samples - cur_num_samples
                if remaining <= 0:
                    break

                batch_len = None
                for value in batch.values():
                    if hasattr(value, "shape") and len(value.shape) > 0:
                        batch_len = int(value.shape[0])
                        break
                if batch_len is None:
                    raise ValueError(
                        "[NormStats] Could not infer batch size from DataLoader batch"
                    )

                take = min(remaining, batch_len)
                for k in norm_keys:
                    batch_key = self.keyname_to_zarr_key(k, embodiment)
                    if batch_key is None or batch_key not in batch:
                        continue
                    x = batch[batch_key][:take]
                    if hasattr(x, "detach"):
                        x = x.detach().cpu().numpy()
                    collected[k].append(x)

                cur_num_samples += take
                pbar.update(take)

                if checkpoint_fn is not None and _next_ckpt_pct <= 90:
                    if 100 * cur_num_samples / n_samples >= _next_ckpt_pct:
                        checkpoint_fn(collected, _next_ckpt_pct)
                        _next_ckpt_pct += 10

        return collected

    @staticmethod
    def _compute_stats_for_array(X):
        return {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
            "median": np.median(X, axis=0),
            "quantile_1": np.percentile(X, 1, axis=0),
            "quantile_99": np.percentile(X, 99, axis=0),
            "quantile_0_01": np.percentile(X, 0.01, axis=0),
            "quantile_99_99": np.percentile(X, 99.99, axis=0),
        }

    def cache_stats(self, save_cache_dir: str):
        """Write ``norm_stats/norm_stats.json`` under ``save_cache_dir`` (stats as nested lists)."""
        cache_dir = os.path.join(save_cache_dir, "norm_stats")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, "norm_stats.json")

        stats_out: dict[str, dict[str, dict[str, list]]] = {}
        for emb, keys_dict in self.norm_stats.items():
            emb_key = str(emb)
            stats_out[emb_key] = {}
            for key_name, stat_dict in keys_dict.items():
                stats_out[emb_key][key_name] = {
                    stat_name: np.asarray(arr).tolist()
                    for stat_name, arr in stat_dict.items()
                }

        payload = {
            "stats": stats_out,
            "loading_time": None,
            "computing_time": None,
            "frames": None,
        }
        if self._norm_run_metadata is not None:
            for k in ("loading_time", "computing_time", "frames"):
                if k in self._norm_run_metadata:
                    payload[k] = self._norm_run_metadata[k]

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        logger.info(f"[NormStats] Cached stats to {out_path}")

    def viz_img_key(self):
        """
        Get visualization image keys by embodiment.
        """
        return self._viz_img_key

    def all_keys(self):
        """
        Get all key names.

        Returns:
            list: Key names (str).
        """
        return self.df["key_name"].tolist()

    def is_key_with_embodiment(self, key_name, embodiment):
        """
        Check whether a key exists for a given embodiment.

        Args:
            key_name (str): Name of key, e.g., "actions_joints".
            embodiment (int): Integer ID of embodiment.

        Returns:
            bool: if the key exists.
        """
        return (
            (self.df["key_name"] == key_name) & (self.df["embodiment"] == embodiment)
        ).any()

    def keys_of_type(self, key_type, embodiment):
        """
        Get keys of a specific type.

        Args:
            key_type (str): Type of keys, e.g., "camera_keys", "proprio_keys", "action_keys", "metadata_keys".
            embodiment (int): Integer ID of embodiment.

        Returns:
            list: Key names (str) of the given type.
        """
        return self.df[
            (self.df["key_type"] == key_type) & (self.df["embodiment"] == embodiment)
        ]["key_name"].tolist()

    def action_keys(self, embodiment):
        """Get action keys for a specific embodiment."""
        return self.keys_of_type("action_keys", embodiment)

    def key_shape(self, key, embodiment):
        """
        Get the shape of a specific key.

        Args:
            key (str): Name of the key.
            embodiment (int): Integer ID of embodiment.

        Returns:
            tuple or None: Shape as a tuple, or None if not found.
        """
        if key not in self.df["key_name"].values:
            raise ValueError(f"Keyname '{key}' is not in the schematic")

        df_filtered = self.df[
            (self.df["key_name"] == key) & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            raise ValueError(f"Keyname '{key}' with embodiment {embodiment} not found.")

        shape = df_filtered["shape"].item()
        return ast.literal_eval(shape)

    def normalize_data(self, data, embodiment):
        """
        Normalize data using the stored normalization statistics.

        Args:
            data (dict): Maps key names to tensors.
                joint_positions: tensor of shape (B, S, 7)
            embodiment (int): Id of the embodiment.

        Returns:
            dict: Maps key names to normalized tensors.
        """
        if self.norm_stats is None:
            raise ValueError(
                "Normalization statistics not set. Call infer_norm_from_dataset_zarr() first."
            )

        norm_data = {}
        for key, tensor in data.items():
            if key in self.keys_of_type(
                "proprio_keys", embodiment
            ) or key in self.keys_of_type("action_keys", embodiment):
                if (
                    embodiment not in self.norm_stats
                    or key not in self.norm_stats[embodiment]
                ):
                    raise ValueError(
                        f"Missing normalization stats for key {key} and embodiment {embodiment}."
                    )

                stats = self.norm_stats[embodiment][key]
                if self.norm_mode == "zscore":
                    mean = torch.as_tensor(
                        stats["mean"], device=tensor.device, dtype=torch.float32
                    )
                    std = torch.as_tensor(
                        stats["std"], device=tensor.device, dtype=torch.float32
                    )
                    norm_data[key] = (tensor - mean) / (std + 1e-6)
                elif self.norm_mode == "minmax":
                    min = torch.as_tensor(
                        stats["min"], device=tensor.device, dtype=torch.float32
                    )
                    max = torch.as_tensor(
                        stats["max"], device=tensor.device, dtype=torch.float32
                    )
                    ndata = (tensor - min) / (max - min + 1e-6)
                    norm_data[key] = 2.0 * ndata - 1.0
                elif self.norm_mode == "quantile":
                    quantile_1 = torch.as_tensor(
                        stats["quantile_1"], device=tensor.device, dtype=torch.float32
                    )
                    quantile_99 = torch.as_tensor(
                        stats["quantile_99"], device=tensor.device, dtype=torch.float32
                    )
                    ndata = (tensor - quantile_1) / (quantile_99 - quantile_1 + 1e-6)
                    norm_data[key] = 2.0 * ndata - 1.0
                else:
                    raise ValueError(f"Invalid normalization mode: {self.norm_mode}")
            else:
                norm_data[key] = tensor

        return norm_data

    def unnormalize_data(self, data, embodiment):
        """
        Unnormalize data using the stored normalization statistics.
        Args:
            data (dict): Maps key names to tensors.
                joint_positions: tensor of shape (B, S, 7)
            embodiment (int): Id of the embodiment.
        Returns:
            dict: Maps key names to denormalized tensors.
        """
        if self.norm_stats is None:
            raise ValueError(
                "Normalization statistics not set. Call infer_norm_from_dataset_zarr() first."
            )

        denorm_data = {}
        for key, tensor in data.items():
            if key in self.keys_of_type(
                "proprio_keys", embodiment
            ) or key in self.keys_of_type("action_keys", embodiment):
                if (
                    embodiment not in self.norm_stats
                    or key not in self.norm_stats[embodiment]
                ):
                    raise ValueError(
                        f"Missing normalization stats for key {key} and embodiment {embodiment}."
                    )

                stats = self.norm_stats[embodiment][key]
                if self.norm_mode == "zscore":
                    mean = torch.as_tensor(
                        stats["mean"], device=tensor.device, dtype=torch.float32
                    )
                    std = torch.as_tensor(
                        stats["std"], device=tensor.device, dtype=torch.float32
                    )
                    denorm_data[key] = tensor * (std + 1e-6) + mean

                elif self.norm_mode == "minmax":
                    min_val = torch.as_tensor(
                        stats["min"], device=tensor.device, dtype=torch.float32
                    )
                    max_val = torch.as_tensor(
                        stats["max"], device=tensor.device, dtype=torch.float32
                    )
                    denorm_data[key] = (tensor + 1) * 0.5 * (
                        max_val - min_val + 1e-6
                    ) + min_val

                elif self.norm_mode == "quantile":
                    quantile_1 = torch.as_tensor(
                        stats["quantile_1"], device=tensor.device, dtype=torch.float32
                    )
                    quantile_99 = torch.as_tensor(
                        stats["quantile_99"], device=tensor.device, dtype=torch.float32
                    )
                    denorm_data[key] = (tensor + 1) * 0.5 * (
                        quantile_99 - quantile_1 + 1e-6
                    ) + quantile_1

                else:
                    raise ValueError(f"Invalid normalization mode: {self.norm_mode}")
            else:
                denorm_data[key] = tensor

        return denorm_data

    @staticmethod
    def _iter_leaf_datasets(ds):
        """Yield leaf datasets from possibly nested dataset wrappers."""
        if isinstance(ds, ZarrDataset):
            yield ds
        elif isinstance(ds, MultiDataset):
            for child in ds.datasets.values():
                yield from DataSchematic._iter_leaf_datasets(child)
        else:
            yield ds

    @staticmethod
    def _key_map_for_any(ds) -> dict:
        """Return the key map for a dataset-like object."""
        km = getattr(ds, "key_map", None)
        return km

    @staticmethod
    def dataset_raw_norm_keys(
        ds,
        key_types=("proprio_keys", "action_keys"),
        extra_keys=(),
        include_all_key_map_keys=False,
    ) -> list[str]:
        """
        Collect raw key names to use for normalization.

        Args:
            ds: Dataset or nested dataset wrapper.
            key_types (tuple): Key types to include from each key map.
            extra_keys (tuple): Additional keys to include explicitly.
            include_all_key_map_keys (bool): If True, include all keys in key maps.

        Returns:
            list[str]: Sorted unique key names.
        """
        out = set(extra_keys)
        for leaf in DataSchematic._iter_leaf_datasets(ds):
            km = DataSchematic._key_map_for_any(leaf)
            if include_all_key_map_keys:
                out |= set(km.keys())
            else:
                for k, info in km.items():
                    if info.get("key_type") in set(key_types):
                        out.add(k)
        return sorted(out)
