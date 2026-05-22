"""
Standalone trainer for the OAT action tokenizer.

This algo wraps an `oat.tokenizer.oat.tokenizer.OATTok` instance so that it can
be trained on its own (encoder + FSQ + decoder, action-reconstruction MSE only)
through the standard egomimic / PyTorch Lightning training pipeline.

Action chunks come out of the egomimic data pipeline already normalized to
[-1, 1] by `data_schematic.normalize_data`. To avoid double-normalization we
seed the tokenizer's internal `LinearNormalizer` with the identity transform —
the same convention used by `AutoregressivePolicy` in joint-training mode.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from oat.model.common.normalizer import SingleFieldLinearNormalizer
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class OATTokenizerTrainer(Algo):
    """Train only the OAT action tokenizer on egomimic action chunks."""

    def __init__(
        self,
        data_schematic,
        domains: List[str],
        ac_keys: Dict[str, str],
        action_tokenizer,
        viz_func: Optional[dict] = None,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.viz_func = viz_func
        self.domains = list(domains)
        self.ac_keys = dict(ac_keys)
        self.action_tokenizer = action_tokenizer

        # Identity normalizer — egomimic data_schematic has already mapped
        # actions into [-1, 1] before they reach the tokenizer.
        action_dim = action_tokenizer.decoder.sample_dim
        scale = torch.ones(action_dim, dtype=torch.float32)
        offset = torch.zeros(action_dim, dtype=torch.float32)
        stats = {
            "min": -torch.ones(action_dim, dtype=torch.float32),
            "max": torch.ones(action_dim, dtype=torch.float32),
            "mean": torch.zeros(action_dim, dtype=torch.float32),
            "std": torch.ones(action_dim, dtype=torch.float32),
        }
        action_tokenizer.normalizer["action"] = (
            SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "OATTokenizerTrainer requires CUDA; no CUDA device is available."
            )
        device_arg = kwargs.get("device")
        if device_arg is not None:
            self.device = torch.device(device_arg)
            if self.device.type != "cuda":
                raise ValueError(
                    f"device must be CUDA (e.g. cuda or cuda:0), got {self.device!r}"
                )
        else:
            self.device = torch.device("cuda")

        # Resolve the action key for each embodiment id (mirrors AutoregressivePolicy).
        self.ac_keys_by_id: Dict[int, str] = {}
        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            for key in data_schematic.keys_of_type("action_keys", embodiment_id):
                if (
                    data_schematic.is_key_with_embodiment(key, embodiment_id)
                    and key == self.ac_keys[embodiment]
                ):
                    self.ac_keys_by_id[embodiment_id] = key
            if embodiment_id not in self.ac_keys_by_id:
                raise ValueError(
                    f"Could not resolve action key {self.ac_keys[embodiment]!r} "
                    f"for embodiment {embodiment!r} in data_schematic."
                )

        self.nets["tokenizer"] = action_tokenizer
        self.nets = self.nets.float().to(self.device)

        self.training_step = 0

    # ------------------------------------------------------------------
    # PyTorch Lightning hooks
    # ------------------------------------------------------------------

    @override
    def process_batch_for_training(self, batch):
        processed_batch = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.zarr_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    processed_batch[embodiment_id][key_name] = value

            ac_key = self.ac_keys_by_id[embodiment_id]
            if ac_key not in processed_batch[embodiment_id]:
                raise KeyError(
                    f"Action key {ac_key!r} missing from batch for embodiment "
                    f"{embodiment_name!r}; available keys: "
                    f"{list(processed_batch[embodiment_id].keys())}"
                )
            ac = processed_batch[embodiment_id][ac_key]
            if ac.ndim != 3:
                raise ValueError(
                    f"Expected actions of shape (B, S, D) for tokenizer training, "
                    f"got {tuple(ac.shape)}"
                )

            processed_batch[embodiment_id] = self.data_schematic.normalize_data(
                processed_batch[embodiment_id], embodiment_id
            )
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )
            for key, value in processed_batch[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed_batch[embodiment_id][key] = value

        return processed_batch

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        self.training_step += 1
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys_by_id[embodiment_id]
            actions = _batch[ac_key]
            loss = self.nets["tokenizer"]({"action": actions})
            predictions[f"{embodiment_name}_loss"] = loss
        return predictions

    @override
    def forward_eval(self, batch):
        recons_dict = {}
        with torch.inference_mode():
            for embodiment_id, _batch in batch.items():
                embodiment_name = get_embodiment(embodiment_id).lower()
                ac_key = self.ac_keys_by_id[embodiment_id]
                actions = _batch[ac_key]
                recons = self.nets["tokenizer"].autoencode(samples=actions)
                recons_dict[f"{embodiment_name}_{ac_key}"] = recons
        return recons_dict

    @override
    def forward_eval_logging(self, batch):
        metrics: Dict[str, torch.Tensor] = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys_by_id[embodiment_id]
            actions = _batch[ac_key]
            with torch.inference_mode():
                recons = self.nets["tokenizer"].autoencode(samples=actions)
                mse = F.mse_loss(recons, actions)
            metrics[f"{embodiment_name}_reconst_mse"] = mse
        return metrics, {}

    @override
    def visualize_preds(self, predictions, batch):
        return None

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for embodiment_id, _ in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            tok_loss = predictions[f"{embodiment_name}_loss"]
            loss_dict[f"{embodiment_name}_loss"] = tok_loss
            total = total + tok_loss
        loss_dict["action_loss"] = total / max(len(self.domains), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log
