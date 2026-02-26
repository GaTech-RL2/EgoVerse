import ast
import logging
import math
import os
import random

import numpy as np
import pandas as pd
import torch

from egomimic.rldb.embodiment import get_embodiment_id
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
    def __init__(self, schematic_dict, viz_img_key, norm_mode="zscore"):
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


        Attributes:
            df (pd.DataFrame): Columns include 'key_name', 'key_type', and 'shape', 'embodiment'.
        """

        rows = []
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
        self._viz_img_key = {get_embodiment_id(k): v for k, v in viz_img_key.items()}
        self.shapes_infered = False
        self.norm_mode = norm_mode
        self.norm_stats = {emb: {} for emb in self.embodiments}

    def zarr_key_to_keyname(self, zarr_key, embodiment):
        """
        Get the key name from the zarr key.


        Args:
            zarr_key (str): zarr key, e.g., "front_img_1".
            embodiment (int): int id corresponding to embodiment




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
            embodiment (int): int id corresponding to embodiment


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
        log_every: int = 200,
        include_all_key_map_keys: bool = False,
        extra_aux_keys=(),
    ):
        embodiment = dataset_name
        if isinstance(embodiment, str):
            embodiment = get_embodiment_id(embodiment)

        norm_keys = []
        norm_keys.extend(self.keys_of_type("proprio_keys"))
        norm_keys.extend(self.keys_of_type("action_keys"))
        norm_keys = [k for k in norm_keys if self.is_key_with_embodiment(k, embodiment)]

        if not norm_keys:
            logger.warning(
                f"[NormStats] No proprio/action keys for embodiment={embodiment}"
            )
            return

        aux_keys = self.dataset_raw_norm_keys(
            dataset,
            key_types=("proprio_keys", "action_keys"),
            extra_keys=extra_aux_keys,
            include_all_key_map_keys=include_all_key_map_keys,
        )

        if not aux_keys:
            logger.warning(f"[NormStats] No aux keys found for dataset={type(dataset)}")
            return

        def get_item_keys_from_any(ds, idx, keys):
            if hasattr(ds, "get_item_keys"):
                return ds.get_item_keys(idx, keys=keys)

            if hasattr(ds, "datasets") and hasattr(ds, "index_map"):
                dataset_name_, local_idx = ds.index_map[idx]
                child = ds.datasets[dataset_name_]
                if not hasattr(child, "get_item_keys"):
                    raise RuntimeError(
                        f"Child dataset {type(child)} has no get_item_keys"
                    )
                return child.get_item_keys(local_idx, keys=keys)

            raise RuntimeError(f"Unsupported dataset type: {type(ds)}")

        N = len(dataset)
        if N <= 0:
            raise ValueError("Dataset is empty")

        n_samples = int(math.ceil(sample_frac * N))
        n_samples = max(1, min(n_samples, N))
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        rng = random.Random(seed)
        sample_indices = rng.sample(range(N), k=n_samples)

        logger.info(f"[NormStats] embodiment={embodiment} norm_keys={norm_keys}")
        logger.info(f"[NormStats] aux_keys={aux_keys}")
        logger.info(
            f"[NormStats] sampling {n_samples}/{N} (~{100 * sample_frac:.1f}%) indices"
        )

        collected = {k: [] for k in norm_keys}
        expected_shapes = {}

        for i, idx in enumerate(sample_indices, 1):
            item = get_item_keys_from_any(dataset, idx, keys=aux_keys)
            for k in norm_keys:
                item_key = self.keyname_to_zarr_key(k, embodiment)
                if item_key not in item:
                    continue

                x = item[item_key]
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                x = np.asarray(x)

                if k not in expected_shapes:
                    expected_shapes[k] = x.shape
                else:
                    if x.shape != expected_shapes[k]:
                        raise ValueError(
                            f"[NormStats] Shape mismatch for key '{k}': "
                            f"expected {expected_shapes[k]}, got {x.shape}. "
                            "Ensure padding/horizon is consistent."
                        )

                collected[k].append(x)

            if log_every and (i % log_every == 0):
                logger.info(f"[NormStats] processed {i}/{n_samples} samples")

        for k in norm_keys:
            if not collected[k]:
                logger.warning(f"[NormStats] No data collected for key={k}")
                continue

            X = np.stack(collected[k], axis=0)

            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            minv = np.min(X, axis=0)
            maxv = np.max(X, axis=0)
            median = np.median(X, axis=0)
            q1 = np.percentile(X, 1, axis=0)
            q99 = np.percentile(X, 99, axis=0)

            self.norm_stats[embodiment][k] = {
                "mean": torch.from_numpy(mean).float(),
                "std": torch.from_numpy(std).float(),
                "min": torch.from_numpy(minv).float(),
                "max": torch.from_numpy(maxv).float(),
                "median": torch.from_numpy(median).float(),
                "quantile_1": torch.from_numpy(q1).float(),
                "quantile_99": torch.from_numpy(q99).float(),
            }

            logger.info(
                f"[NormStats] key={k} samples={X.shape[0]} stat_shape={mean.shape}"
            )

        logger.info("[NormStats] Finished norm inference")

    def viz_img_key(self):
        """
        Get the key that should be used for offline visualization
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
        Check if a key_name exists with a given embodiment


        Args:
            key_name (str): name of key, e.g. actions_joints
            embodiment (int): integer id of embodiment


        Returns:
            bool: if the key exists.
        """
        return (
            (self.df["key_name"] == key_name) & (self.df["embodiment"] == embodiment)
        ).any()

    def keys_of_type(self, key_type):
        """
        Get keys of a specific type.


        Args:
            key_type (str): Type of keys, e.g., "camera_keys", "proprio_keys", "action_keys", "metadata_keys".


        Returns:
            list: Key names (str) of the given type.
        """
        return self.df[self.df["key_type"] == key_type]["key_name"].tolist()

    def action_keys(self):
        return self.keys_of_type("action_keys")

    def key_shape(self, key, embodiment):
        """
        Get the shape of a specific key.


        Args:
            key (str): Name of the key.
            embodiment (int): integer id of embodiment


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
            if key in self.keys_of_type("proprio_keys") or key in self.keys_of_type(
                "action_keys"
            ):
                if (
                    embodiment not in self.norm_stats
                    or key not in self.norm_stats[embodiment]
                ):
                    raise ValueError(
                        f"Missing normalization stats for key {key} and embodiment {embodiment}."
                    )

                stats = self.norm_stats[embodiment][key]
                if self.norm_mode == "zscore":
                    mean = stats["mean"].to(tensor.device)
                    std = stats["std"].to(tensor.device)
                    norm_data[key] = (tensor - mean) / (std + 1e-6)
                elif self.norm_mode == "minmax":
                    min = stats["min"].to(tensor.device)
                    max = stats["max"].to(tensor.device)
                    ndata = (tensor - min) / (max - min + 1e-6)
                    norm_data[key] = 2.0 * ndata - 1.0
                elif self.norm_mode == "quantile":
                    quantile_1 = stats["quantile_1"].to(tensor.device)
                    quantile_99 = stats["quantile_99"].to(tensor.device)
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
            if key in self.keys_of_type("proprio_keys") or key in self.keys_of_type(
                "action_keys"
            ):
                if (
                    embodiment not in self.norm_stats
                    or key not in self.norm_stats[embodiment]
                ):
                    raise ValueError(
                        f"Missing normalization stats for key {key} and embodiment {embodiment}."
                    )

                stats = self.norm_stats[embodiment][key]
                if self.norm_mode == "zscore":
                    mean = stats["mean"].to(tensor.device)
                    std = stats["std"].to(tensor.device)
                    denorm_data[key] = tensor * (std + 1e-6) + mean

                elif self.norm_mode == "minmax":
                    min_val = stats["min"].to(tensor.device)
                    max_val = stats["max"].to(tensor.device)
                    denorm_data[key] = (tensor + 1) * 0.5 * (
                        max_val - min_val + 1e-6
                    ) + min_val

                elif self.norm_mode == "quantile":
                    quantile_1 = stats["quantile_1"].to(tensor.device)
                    quantile_99 = stats["quantile_99"].to(tensor.device)
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
        if isinstance(ds, ZarrDataset):
            yield ds
        elif isinstance(ds, MultiDataset):
            for child in ds.datasets.values():
                yield from DataSchematic._iter_leaf_datasets(child)
        else:
            yield ds

    @staticmethod
    def _key_map_for_any(ds) -> dict:
        km = getattr(ds, "key_map", None)
        return km

    @staticmethod
    def dataset_raw_norm_keys(
        ds,
        key_types=("proprio_keys", "action_keys"),
        extra_keys=(),
        include_all_key_map_keys=False,
    ) -> list[str]:
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
