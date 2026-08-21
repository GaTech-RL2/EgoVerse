"""Dataset-free normalization state used by training and deployment models."""

from __future__ import annotations

import copy

import numpy as np
import torch


class NormStats:
    """Checkpoint normalization metadata without importing a dataset backend."""

    NORMALIZE_KEY_TYPES = ("proprio_keys", "action_keys")

    def __init__(self, state: dict):
        if state is None:
            raise ValueError("normalization state is required")
        self.norm_mode = state.get("norm_mode", "zscore")
        self.embodiments = set(state.get("embodiments", []))
        self.key_types = copy.deepcopy(state.get("key_types", {}))
        self.zarr_keys = copy.deepcopy(state.get("zarr_keys", {}))
        self.shapes = copy.deepcopy(state.get("shapes", {}))
        self.norm_stats = self._clone_norm_stats(state.get("norm_stats", {}))
        for embodiment_id in self.embodiments:
            self.key_types.setdefault(embodiment_id, {})
            self.zarr_keys.setdefault(embodiment_id, {})
            self.shapes.setdefault(embodiment_id, {})
            self.norm_stats.setdefault(embodiment_id, {})

    @staticmethod
    def _clone_norm_stats(norm_stats):
        return {
            embodiment_id: {
                key: {
                    name: (
                        value.detach().cpu().clone()
                        if torch.is_tensor(value)
                        else copy.deepcopy(value)
                    )
                    for name, value in stats.items()
                }
                for key, stats in per_embodiment.items()
            }
            for embodiment_id, per_embodiment in (norm_stats or {}).items()
        }

    def keys_of_type(self, key_type: str, embodiment_id: int) -> list[str]:
        return [
            key
            for key, actual_type in self.key_types.get(embodiment_id, {}).items()
            if actual_type == key_type
        ]

    def is_key_with_embodiment(self, key_name: str, embodiment_id: int) -> bool:
        return key_name in self.key_types.get(embodiment_id, {})

    def keyname_to_zarr_key(self, key_name: str, embodiment_id: int) -> str | None:
        return self.zarr_keys.get(embodiment_id, {}).get(key_name)

    def zarr_key_to_keyname(self, zarr_key: str, embodiment_id: int) -> str | None:
        for key_name, candidate in self.zarr_keys.get(embodiment_id, {}).items():
            if candidate == zarr_key:
                return key_name
        return None

    def key_shape(self, key_name: str, embodiment_id: int) -> tuple:
        try:
            return self.shapes[embodiment_id][key_name]
        except KeyError as exc:
            raise ValueError(
                f"Shape for key {key_name!r} on embodiment {embodiment_id} "
                "is unavailable."
            ) from exc

    def _apply_norm_one(self, tensor, stats):
        if self.norm_mode == "zscore":
            mean = torch.as_tensor(stats["mean"], device=tensor.device).float()
            std = torch.as_tensor(stats["std"], device=tensor.device).float()
            return (tensor - mean) / (std + 1e-6)
        if self.norm_mode == "minmax":
            minimum = torch.as_tensor(stats["min"], device=tensor.device).float()
            maximum = torch.as_tensor(stats["max"], device=tensor.device).float()
            return 2.0 * ((tensor - minimum) / (maximum - minimum + 1e-6)) - 1.0
        if self.norm_mode == "quantile":
            q1 = torch.as_tensor(stats["quantile_1"], device=tensor.device).float()
            q99 = torch.as_tensor(stats["quantile_99"], device=tensor.device).float()
            return 2.0 * ((tensor - q1) / (q99 - q1 + 1e-6)) - 1.0
        raise ValueError(f"Invalid normalization mode: {self.norm_mode}")

    def _apply_unnorm_one(self, tensor, stats):
        if self.norm_mode == "zscore":
            mean = torch.as_tensor(stats["mean"], device=tensor.device).float()
            std = torch.as_tensor(stats["std"], device=tensor.device).float()
            return tensor * (std + 1e-6) + mean
        if self.norm_mode == "minmax":
            minimum = torch.as_tensor(stats["min"], device=tensor.device).float()
            maximum = torch.as_tensor(stats["max"], device=tensor.device).float()
            return (tensor + 1) * 0.5 * (maximum - minimum + 1e-6) + minimum
        if self.norm_mode == "quantile":
            q1 = torch.as_tensor(stats["quantile_1"], device=tensor.device).float()
            q99 = torch.as_tensor(stats["quantile_99"], device=tensor.device).float()
            return (tensor + 1) * 0.5 * (q99 - q1 + 1e-6) + q1
        raise ValueError(f"Invalid normalization mode: {self.norm_mode}")

    def normalize(self, data: dict, embodiment_id: int) -> dict:
        if not self.norm_stats.get(embodiment_id):
            return data
        out = dict(data)
        for key_name, key_type in self.key_types.get(embodiment_id, {}).items():
            if key_type not in self.NORMALIZE_KEY_TYPES:
                continue
            stats = self.norm_stats[embodiment_id].get(key_name)
            zarr_key = self.zarr_keys[embodiment_id].get(key_name)
            if stats is None or zarr_key not in out:
                continue
            value = out[zarr_key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            if torch.is_tensor(value):
                out[zarr_key] = self._apply_norm_one(value, stats)
        return out

    def unnormalize(self, data: dict, embodiment_id: int) -> dict:
        if not self.norm_stats.get(embodiment_id):
            return data
        out = dict(data)
        zarr_to_name = {
            value: key
            for key, value in self.zarr_keys.get(embodiment_id, {}).items()
        }
        for data_key, value in data.items():
            key_name = (
                data_key
                if data_key in self.norm_stats[embodiment_id]
                else zarr_to_name.get(data_key)
            )
            stats = self.norm_stats[embodiment_id].get(key_name)
            if stats is None:
                continue
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            if torch.is_tensor(value):
                out[data_key] = self._apply_unnorm_one(value, stats)
        return out
