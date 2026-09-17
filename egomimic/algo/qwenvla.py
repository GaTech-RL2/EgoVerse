"""QwenVLA: Qwen 3.5 VLM + layer-wise cross-DiT flow head (starVLA QwenPI port).

Design: docs/hpt-experiments/2026-09-16_qwenvla_design.md. Modeled on the PI
algo (prompt assembly with an embodiment block, one loop per embodiment,
norm stats, the unnormalised-prediction contract ``eval_hpt.py`` reads) and
fed HPT-style batches (native action layout, dotted dataset keys, proprio
``(B, D)`` or ``(B, K, D)`` from the ``proprio_history`` keymap).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Literal, Optional

import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.algo.eval_determinism import DeterministicEvalMixin
from egomimic.models.hpt_nets import verify_pretrained_weights
from egomimic.models.image_augs import PerSampleAugs
from egomimic.models.qwenvla_nets import QwenVLAModel, QwenVLBackbone
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.utils.action_utils import pad_to_width

logger = logging.getLogger(__name__)


class QwenVLA(DeterministicEvalMixin, Algo):
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        """Move the algo (and its nets) to ``value``; written by ``ModelWrapper``
        once Lightning knows the rank's device (see ``__init__``)."""
        self._device = torch.device(value) if value is not None else torch.device("cpu")
        nets = getattr(self, "nets", None)
        if nets is not None:
            nets.to(self._device)

    def __init__(
        self,
        norm_stats,
        backbone: QwenVLBackbone,
        head,  # functools.partial of LayerwiseFMHead (hydra ``_partial_: true``)
        domains: list,
        dims: dict,
        ac_keys: dict,
        action_width: Optional[int] = None,
        train_image_augs=None,
        eval_image_augs=None,
        annotation_key: Optional[str] = None,
        annotation_sampling_mode: Literal["random", "first"] = "random",
        default_prompt: str = "",
        embodiment_label: bool = True,
        control_mode: Optional[dict] = None,
        use_pad_mask: bool = False,
        history_len: int = 1,
        history_dropout: float = 0.0,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.norm_stats = norm_stats
        self.domains = list(domains)
        self.dims = {d: dict(dims[d]) for d in self.domains}
        widths = {d: int(self.dims[d]["action"]) for d in self.domains}
        self.action_width = (
            int(action_width) if action_width is not None else max(widths.values())
        )
        for d, w in widths.items():
            if w > self.action_width:
                raise ValueError(
                    f"{d} action is {w}-wide but action_width={self.action_width}"
                )
        self.annotation_key = annotation_key
        self.annotation_sampling_mode = annotation_sampling_mode
        self.default_prompt = default_prompt
        self.embodiment_label = bool(embodiment_label)
        self.control_mode = dict(control_mode) if control_mode else None
        self.use_pad_mask = bool(use_pad_mask)
        self.train_image_augs = (
            PerSampleAugs(train_image_augs) if train_image_augs is not None else None
        )
        self.eval_image_augs = eval_image_augs
        self.is_6dof = kwargs.get("6dof", False)
        # Read by eval_hpt.py; QwenVLA has one head, no auxiliary / shared keys.
        self.shared_ac_key = None
        self.auxiliary_ac_keys = {}
        # Construct on the CPU. Lightning learns each rank's GPU only after the
        # model exists and writes it to ``device`` (``ModelWrapper.on_fit_start``
        # / ``on_validation_start``), whose setter moves the nets. Picking
        # ``cuda`` here would put every DDP rank's 4.8 GB of fp32 VLM weights on
        # cuda:0 until then, which overflows it on a checkpoint resume.
        self._device = torch.device(kwargs.get("device", "cpu"))

        state_dims = {
            d: int(self.dims[d]["proprio"])
            for d in self.domains
            if self.dims[d].get("proprio")
        }
        head_module = head(
            action_width=self.action_width,
            vlm_hidden=backbone.hidden_size,
            num_layers=backbone.num_layers,
            state_dims=state_dims or None,
            history_len=history_len,
            history_dropout=history_dropout,
        )
        self.nets["policy"] = QwenVLAModel(backbone, head_module)
        self.nets = self.nets.float().to(self._device)
        verify_pretrained_weights(self.nets["policy"])

        self.ac_keys = dict(ac_keys)
        self.camera_keys, self.proprio_keys, self.lang_keys = {}, {}, {}
        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.camera_keys[embodiment_id] = []
            self.proprio_keys[embodiment_id] = []
            self.lang_keys[embodiment_id] = []
            for key in norm_stats.keys_of_type("action_keys", embodiment_id):
                if (
                    norm_stats.is_key_with_embodiment(key, embodiment_id)
                    and key == self.ac_keys[embodiment]
                ):
                    self.ac_keys[embodiment_id] = key
            for key in norm_stats.keys_of_type("camera_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.camera_keys[embodiment_id].append(key)
            for key in norm_stats.keys_of_type("proprio_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.proprio_keys[embodiment_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", embodiment_id):
                if norm_stats.is_key_with_embodiment(key, embodiment_id):
                    self.lang_keys[embodiment_id].append(key)
            if embodiment_id not in self.ac_keys:
                raise KeyError(
                    f"ac_keys[{embodiment}]={self.ac_keys[embodiment]!r} is not an action "
                    "key of the norm stats"
                )
        self.training_step = 0

    # ------------------------------------------------------------------
    # prompts
    # ------------------------------------------------------------------
    def _control_mode_for(self, embodiment_name: str) -> str:
        for key, value in self.control_mode.items():
            if key.lower() in embodiment_name.lower():
                return value
        raise ValueError(
            f"control_mode has no entry matching embodiment {embodiment_name!r} "
            f"(declared keys: {sorted(self.control_mode)})"
        )

    def compose_prompt(self, embodiment_name: str, annotation: str) -> str:
        """``Embodiment: human bimanual. [Control mode: X.] Task: <annotation>``;
        the task block is dropped when the annotation is empty."""
        blocks = []
        if self.embodiment_label:
            blocks.append(f"Embodiment: {embodiment_name.lower().replace('_', ' ')}.")
        if self.control_mode:
            blocks.append(f"Control mode: {self._control_mode_for(embodiment_name)}.")
        annotation = (annotation or "").strip()
        if annotation:
            blocks.append(f"Task: {annotation}")
        return " ".join(blocks)

    # ------------------------------------------------------------------
    # batches
    # ------------------------------------------------------------------
    @override
    def process_batch_for_training(self, batch):
        processed_batch = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            out = {}
            for key, value in _batch.items():
                key_name = (
                    self.norm_stats.zarr_key_to_keyname(key, embodiment_id) or key
                )
                out[key_name] = value
            ac_key = self.ac_keys[embodiment_id]
            if ac_key not in out or out[ac_key].ndim != 3:
                raise ValueError(f"expected a (B, T, D) action under {ac_key!r}")
            B, S, _ = out[ac_key].shape
            action_pad_mask = out.pop("action_pad_mask", None)
            out.pop("proprio_history_mask", None)  # front padding == history dropout
            if self.use_pad_mask and action_pad_mask is not None:
                if action_pad_mask.shape[-1] != S:
                    raise ValueError(
                        f"action_pad_mask length {action_pad_mask.shape[-1]} != horizon {S}"
                    )
                out["pad_mask"] = action_pad_mask[..., None]
            else:
                out["pad_mask"] = torch.ones(B, S, 1)
            out["sampled_prompt"] = self._build_prompts(_batch, B)
            out["embodiment"] = torch.tensor([embodiment_id], dtype=torch.int64)
            for key, value in out.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    out[key] = value
            processed_batch[embodiment_id] = out
        return processed_batch

    def _apply_image_augs(self, images: torch.Tensor) -> torch.Tensor:
        if self.nets.training and self.train_image_augs is not None:
            return self.train_image_augs(images)
        if not self.nets.training and self.eval_image_augs is not None:
            return self.eval_image_augs(images)
        return images

    def _to_model_data(self, _batch: dict, embodiment_id: int) -> dict:
        """Processed batch -> the dict ``QwenVLAModel`` consumes."""
        name = get_embodiment(embodiment_id).lower()
        frames = [
            self._apply_image_augs(_batch[key])
            for key in self.camera_keys[embodiment_id]
            if key in _batch
        ]
        if not frames:
            raise KeyError(
                f"no camera of {self.camera_keys[embodiment_id]} in the batch keys {sorted(_batch)}"
            )
        images = torch.stack(frames, dim=1)  # (B, F, 3, H, W)

        parts = []
        for key in self.proprio_keys[embodiment_id]:
            if key not in _batch:
                continue
            value = _batch[key]
            parts.append(value[:, None] if value.ndim == 2 else value)  # (B, K, D)
        state = None
        if parts:
            steps = {p.shape[1] for p in parts}
            if len(steps) != 1:
                raise ValueError(
                    f"proprio keys carry different history lengths: {steps}"
                )
            state = torch.cat(parts, dim=-1)
            expected = int(self.dims[name]["proprio"])
            if state.shape[-1] != expected:
                raise ValueError(
                    f"{name} proprio is {state.shape[-1]}-wide, dims says {expected}"
                )

        ac_key = self.ac_keys[embodiment_id]
        action = _batch[ac_key]
        native_width = action.shape[-1]
        action = pad_to_width(action, self.action_width)
        width_mask = torch.zeros(1, 1, self.action_width, device=action.device)
        width_mask[..., :native_width] = 1.0
        loss_mask = _batch["pad_mask"].to(action.device) * width_mask

        prompts = [self.compose_prompt(name, p) for p in _batch["sampled_prompt"]]
        return {
            "images": images,
            "prompts": prompts,
            "state": state,
            "action": action,
            "loss_mask": loss_mask,
            "embodiment_name": name,
        }

    # ------------------------------------------------------------------
    # train / eval
    # ------------------------------------------------------------------
    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        self.training_step += 1
        for embodiment_id, _batch in batch.items():
            name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id]
            data = self._to_model_data(_batch, embodiment_id)
            predictions[f"{name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{name}_loss"] = self.nets["policy"].compute_loss(data)
        return predictions

    @override
    def forward_eval(self, batch):
        unnorm_preds = {}
        generator = self._eval_generator()
        pass_seed = self._eval_pass_seed()
        fork_devices = self._eval_fork_devices()
        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                name = get_embodiment(embodiment_id).lower()
                ac_key = self.ac_keys[embodiment_id]
                data = self._to_model_data(_batch, embodiment_id)
                # Val loss draws noise / t from the global RNG inside the head:
                # fork + seed it so the reported loss is reproducible.
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(pass_seed + 1)
                    unnorm_preds[f"{name}_loss"] = self.nets["policy"].compute_loss(
                        data
                    )
                pred = self.nets["policy"].sample(data, generator=generator)
                ref = _batch[ac_key]
                _, T, D = ref.shape
                unnorm = self.norm_stats.unnormalize(
                    {ac_key: pred[:, :T, :D]}, embodiment_id
                )
                for key, value in unnorm.items():
                    unnorm_preds[f"{name}_{key}"] = value
        self._eval_pass_counter = int(getattr(self, "_eval_pass_counter", 0)) + 1
        return unnorm_preds

    @override
    def compute_losses(self, predictions, batch):
        loss_dict = OrderedDict()
        total = torch.tensor(0.0, device=self.device)
        for embodiment_id in batch:
            name = get_embodiment(embodiment_id).lower()
            bc_loss = predictions[f"{name}_loss"]
            total = total + bc_loss
            loss_dict[f"{name}_loss"] = bc_loss
        loss_dict["action_loss"] = total / len(self.domains)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for loss_key, loss in info["losses"].items():
            log[loss_key] = loss.item()
        return log
