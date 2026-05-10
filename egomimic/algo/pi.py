import logging
import os
import random
from collections import OrderedDict
from typing import Literal

import numpy as np
import openpi
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import safetensors
import torch
import torch.nn as nn
from openpi.shared.image_tools import resize_with_pad_torch
from overrides import override
from transformers import AutoTokenizer

from egomimic.algo.algo import Algo
from egomimic.models.preprocess_pi_obs import (
    _concat_proprio,
    _empty_lang_placeholders,
    _ensure_bchw,
    _fill_missing_images,
    _SimpleObservation,
    _to_minus1_1,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.utils.action_utils import ConverterRegistry

logger = logging.getLogger(__name__)
# Ensure logger propagates to root logger and has appropriate level
# Child loggers inherit from parent, but we explicitly set level to ensure INFO messages appear
logger.setLevel(logging.INFO)
logger.propagate = True  # Explicitly enable propagation (default, but ensures it works)


class PI(Algo):
    """ """

    def __init__(
        self,
        norm_stats,
        camera_transforms,
        domains,
        # ---------------------------
        # Image augmentations
        # ---------------------------
        train_image_augs,
        eval_image_augs,
        # ---------------------------
        # Model params
        # ---------------------------
        config,
        # ---------------------------
        ac_keys,
        action_converters,
        # ---------------------------
        # Prompt / tokenization (moved from data config). The PI algo owns
        # the tokenizer because the prompt template is model-specific
        # (paligemma + pi0.5 anchor). See ``process_batch_for_training`` for
        # how these knobs assemble the prompt and produce ``tokenized_*``.
        # ---------------------------
        tokenizer_model_name: str = "google/paligemma-3b-mix-224",
        tokenizer_max_length: int = 128,
        sampling_mode: Literal["first", "random"] = "random",
        annotation_key: str | None = None,
        default_prompt: str = "",
        proprio_in_prompt: bool = False,
        embodiment_label: bool = False,
        state_num_bins: int = 256,
        control_mode: dict[str, str] | None = None,
        proprio_keys_for_prompt: list[str] | None = None,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.norm_stats = norm_stats

        # ---- Prompt assembly + tokenization (was in collate_fn) ----
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.tokenizer_max_length = tokenizer_max_length
        self.sampling_mode = sampling_mode
        self.annotation_key = annotation_key
        self.default_prompt = default_prompt
        self.proprio_in_prompt = proprio_in_prompt
        self.embodiment_label = embodiment_label
        self.state_num_bins = state_num_bins
        self.control_mode = control_mode
        # Default to the canonical concat key produced by each embodiment's
        # transform_list (ConcatKeys with delete_old_keys removes per-arm zarr keys).
        self.proprio_keys_for_prompt = (
            list(proprio_keys_for_prompt)
            if proprio_keys_for_prompt is not None
            else ["observations.state.ee_pose"]
        )
        self._state_bin_edges = np.linspace(-1.0, 1.0, state_num_bins + 1)[:-1]

        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        if "image_resolution" in kwargs:
            self.image_resolution = kwargs["image_resolution"]
        self.pi_cam_keys = kwargs.get(
            "pi_cam_keys", ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        )
        self.config = config

        self.ac_keys = ac_keys

        self.domains = domains

        self.device = None

        self.camera_keys = {}
        self.proprio_keys = {}
        self.lang_keys = {}

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

        self.num_steps = getattr(self.config, "num_sampling_steps", 10)
        self.is_6dof = kwargs.get("is_6dof", True)

        self.action_converters = action_converters

        self.action_registry = ConverterRegistry()

        arcfg = self.action_converters
        default_ac_key = getattr(arcfg, "ac_key", "actions_cartesian")

        for emb_name, conv_obj in arcfg.rules.items():
            emb_id = get_embodiment_id(emb_name)
            self.action_registry.register(emb_id, self.ac_keys[emb_id], conv_obj)

        fb_obj = arcfg.fallback
        self.action_registry.register("*", default_ac_key, fb_obj)
        self.action_registry.register("*", "*", fb_obj)

        # Create the model
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=self.config.pytorch_training_precision,
            action_dim=self.config.model.action_dim,
            action_horizon=self.config.model.action_horizon,
            max_token_len=self.config.model.max_token_len,
            paligemma_variant=getattr(
                self.config.model, "paligemma_variant", "gemma_2b"
            ),
            action_expert_variant=getattr(
                self.config.model, "action_expert_variant", "gemma_300m"
            ),
            pi05=getattr(config.model, "pi05", False),
        )

        self.model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)

        if self.config.pytorch_weight_path is not None:
            model_path = os.path.join(
                self.config.pytorch_weight_path, "model.safetensors"
            )
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Pretrained weight file not found: {model_path}"
                )
            target = (
                self.model.module
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                else self.model
            )
            safetensors.torch.load_model(target, model_path)
            logger.info(
                "Loaded pretrained weights from %s (%d parameters)",
                model_path,
                sum(p.numel() for p in target.parameters()),
            )
        else:
            logger.warning("No pytorch_weight_path specified — training from scratch")
        self.nets = nn.ModuleDict()
        self.nets["policy"] = self.model

    def _control_mode_for(self, emb_name: str | None) -> str:
        if self.control_mode and emb_name is not None:
            for key, val in self.control_mode.items():
                if key.lower() in emb_name:
                    return val
        if emb_name is not None and "aria" in emb_name:
            return "cam frame xyzypr per arm"
        return "cam frame xyzypr gripper per arm"

    def _discretize_state_for_sample(self, _batch, sample_idx: int) -> str | None:
        """Pick the latest proprio timestep for sample i, clip to [-1, 1],
        digitize into ``state_num_bins`` bins, and return as a space-joined
        string. Returns None if no proprio key is present.
        """
        parts = []
        for k in self.proprio_keys_for_prompt:
            if k not in _batch:
                continue
            v = _batch[k]
            if isinstance(v, torch.Tensor):
                v = v[sample_idx].detach().cpu().numpy()
            else:
                v = np.asarray(v)[sample_idx]
            v = np.asarray(v, dtype=np.float32)
            while v.ndim > 1:
                v = v[-1]
            parts.append(v.reshape(-1))
        if not parts:
            return None
        state = np.concatenate(parts, axis=-1)
        state = np.clip(state, -1.0, 1.0)
        bins = np.digitize(state, bins=self._state_bin_edges) - 1
        return " ".join(map(str, bins.tolist()))

    def _build_prompts(
        self, _batch, embodiment_name: str, batch_size: int
    ) -> list[str]:
        """Sample one prompt per item from the raw annotation lists and
        splice in any of the active blocks. Returns ``batch_size`` strings.

        Mirrors the prompt assembly previously done in
        ``build_tokenized_collate``. Embodiment is known per-batch (one
        DataLoader per embodiment), so we don't re-derive it per sample.
        """
        if self.annotation_key is None or self.annotation_key not in _batch:
            prompts = [self.default_prompt] * batch_size
        else:
            prompts = []
            for sample in _batch[self.annotation_key]:
                if not sample:
                    prompts.append(self.default_prompt)
                elif self.sampling_mode == "random":
                    prompts.append(sample[random.randint(0, len(sample) - 1)])
                else:  # "first"
                    prompts.append(sample[0])

        any_block_active = (
            self.proprio_in_prompt or self.embodiment_label or bool(self.control_mode)
        )
        if not any_block_active:
            return prompts

        emb_name = embodiment_name.lower().replace("_", " ")
        spliced = []
        for i, prompt in enumerate(prompts):
            blocks = [f"Task: {prompt}"]
            if self.embodiment_label:
                blocks.append(f"Embodiment: {emb_name}")
            if self.control_mode:
                blocks.append(f"Control mode: {self._control_mode_for(emb_name)}")
            if self.proprio_in_prompt:
                state_str = self._discretize_state_for_sample(_batch, i)
                if state_str is not None:
                    blocks.append(f"State: {state_str}")
            spliced.append(", ".join(blocks) + ";\nAction: ")
        return spliced

    def _tokenize_prompts(self, prompts: list[str]) -> dict:
        enc = self.tokenizer(
            prompts,
            padding="max_length"
            if self.tokenizer_max_length is not None
            else "longest",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        attention_mask = enc["attention_mask"].bool()
        token_loss_mask = attention_mask.clone()
        token_loss_mask[:, -1] = False
        return {
            "tokenized_prompt": enc["input_ids"].requires_grad_(False),
            "tokenized_mask": attention_mask.requires_grad_(False),
            "token_loss_mask": token_loss_mask.requires_grad_(False),
            "token_ar_mask": attention_mask.clone().requires_grad_(False),
        }

    @override
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            batch (dict): processed dict of batchs that works with pi0.
        """
        processed_batch = {}

        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.norm_stats.zarr_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    processed_batch[embodiment_id][key_name] = value

            ac_key = self.ac_keys[embodiment_id]
            if ac_key not in processed_batch[embodiment_id]:
                raise KeyError(
                    f"Missing action key '{ac_key}' for embodiment {embodiment_id}. "
                    f"Incoming keys were: {list(_batch.keys())}"
                )
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError("Action shape in batch is not 2")

            B, S, _ = processed_batch[embodiment_id][ac_key].shape

            # Build prompts + tokenize. Reads raw `annotations` (list[list[str]])
            # left in `_batch` by `annotation_collate`, plus per-sample proprio
            # tensors from `_batch` for the optional State block.
            prompts = self._build_prompts(_batch, embodiment_name, B)
            processed_batch[embodiment_id]["sampled_prompt"] = prompts
            processed_batch[embodiment_id].update(self._tokenize_prompts(prompts))
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=self.device
            )
            # Samples are already normalized by NormalizeTransform in the leaf's transform_list.
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )

            for key, value in processed_batch[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed_batch[embodiment_id][key] = value

        if not processed_batch:
            raise ValueError(
                f"No valid embodiments found in batch. Batch contained: {list(batch.keys())}, "
                f"but ac_keys only has: {list(self.ac_keys.keys())}"
            )

        return processed_batch

    @override
    def forward_training(self, batch):
        """
        One iteration of training. Sequentially, forward pass loss, Compute forward pass and compute losses.  Return predictions dictionary.  HPT also calculates loss here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """
        # self.nets["policy"].train()
        predictions = OrderedDict()
        for embodiment_id, _batch in batch.items():
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            camera_keys = self.camera_keys.get(embodiment_id, self.pi_cam_keys)
            embodiment_name = get_embodiment(embodiment_id).lower()
            processed_obs, action = self._robomimic_to_pi_data(
                _batch,
                camera_keys,
                proprio_keys,
                lang_keys,
                ac_key,
                embodiment_name,
            )

            losses = self.nets["policy"].forward(processed_obs, action)

            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=action.device, dtype=torch.float32)

            loss = losses.mean()

            predictions[f"{embodiment_name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss

        return predictions

    @override
    def forward_eval(self, batch):
        """
        Compute forward pass and return network outputs in @predictions dict.
        Unnormalize data here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            unnorm_preds (dict): {<embodiment_name>_<ac_key>: torch.Tensor (B, Seq, D)}
        """
        unnorm_preds = {}
        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                proprio_keys = self.proprio_keys[embodiment_id]
                lang_keys = self.lang_keys[embodiment_id]
                ac_key = self.ac_keys[embodiment_id]
                camera_keys = self.camera_keys.get(embodiment_id, self.pi_cam_keys)
                embodiment_name = get_embodiment(embodiment_id).lower()
                processed_obs, action = self._robomimic_to_pi_data(
                    _batch,
                    camera_keys,
                    proprio_keys,
                    lang_keys,
                    ac_key,
                    embodiment_name,
                )

                pred_actions = self.nets["policy"].sample_actions(
                    device=self.device,
                    observation=processed_obs,
                    noise=None,
                    num_steps=self.num_steps,
                )

                predictions = OrderedDict()
                ref = _batch[ac_key]
                B, T, D = ref.shape

                converter = self.action_registry.get(embodiment_id, ac_key)
                pred_actions_orig = converter.from32(pred_actions)

                pred = pred_actions_orig[:, :T, :D]
                predictions[ac_key] = pred

                unnorm_actions = self.norm_stats.unnormalize(predictions, embodiment_id)
                for key in unnorm_actions:
                    unnorm_preds[f"{embodiment_name}_{key}"] = unnorm_actions[key]

        return unnorm_preds

    @override
    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            losses (dict): dictionary of losses computed over the batch
                loss_key_name: torch.Tensor (1)
        """
        loss_dict = OrderedDict()
        total_action_loss = None

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            bc_loss = predictions[f"{embodiment_name}_loss"]
            if total_action_loss is None:
                total_action_loss = torch.tensor(0.0, device=bc_loss.device)
            total_action_loss += bc_loss
            loss_dict[f"{embodiment_name}_loss"] = bc_loss  # for logging

        # in the case we put all embodiments in one batch, get rid of this norm.
        loss_dict["action_loss"] = total_action_loss / len(self.domains)

        return loss_dict

    @override
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
                losses:
                    loss_key_name: torch.Tensor (1)
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for loss_key, loss in info["losses"].items():
            log[loss_key] = loss.item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def _robomimic_to_pi_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment
    ):
        """ """
        if ac_key not in batch:
            raise KeyError(f"Missing action key '{ac_key}' in batch")

        device = self.device
        action = batch[ac_key].to(device)
        image_resolution = getattr(self, "image_resolution", (224, 224))
        required_cam_keys = getattr(self, "pi_cam_keys", cam_keys)

        present_flags = {
            k: (
                k in batch and isinstance(batch[k], torch.Tensor) and batch[k].ndim == 4
            )
            for k in required_cam_keys
        }

        emb_id = get_embodiment_id(embodiment)  # embodiment is a name string
        converter = self.action_registry.get(emb_id, ac_key)
        action32 = converter.to32(action)

        # OpenPI expects a fixed camera tuple. Human datasets only provide
        # `base_0_rgb`, so duplicate that view into the missing wrist slots and
        # mark those synthesized views as masked out below.
        raw_images = _fill_missing_images(batch, required_cam_keys, device)

        # ---- Images (dict[str, Tensor]) ----
        images = {}
        for k in required_cam_keys:
            img = _ensure_bchw(raw_images[k])
            img = _to_minus1_1(img)
            if img.shape[2:] != tuple(image_resolution):
                img = resize_with_pad_torch(img, *image_resolution)
            if img.ndim != 4:
                raise ValueError(
                    f"Expected 4D BCHW image for key '{k}', got shape {tuple(img.shape)}"
                )
            images[k] = img

        if not images:
            raise ValueError("No camera tensors found for the provided cam_keys.")

        # ---- Proprio -> state [B, D] ----
        state = _concat_proprio(batch, proprio_keys, device)
        if state.numel() == 0:
            B = next(iter(images.values())).shape[0]
            state = torch.zeros(B, 0, device=device)
        else:
            B = state.shape[0]

        # ---- Masks for duplicated images + empty language fields ----
        image_masks = {
            k: (
                torch.ones(B, dtype=torch.bool, device=device)
                if present_flags[k]
                else torch.zeros(B, dtype=torch.bool, device=device)
            )
            for k in images.keys()
        }

        has_lang = "tokenized_prompt" in batch and batch["tokenized_prompt"].numel() > 0
        if has_lang:
            tokenized_prompt = batch["tokenized_prompt"].to(device)
            tokenized_prompt_mask = batch["tokenized_mask"].to(device)
            token_ar_mask = batch["token_ar_mask"].to(device)
            token_loss_mask = batch["token_loss_mask"].to(device)
        else:
            tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = (
                _empty_lang_placeholders(B, device)
            )

        # ---- Wrap into simple observation (helpers) ----
        observation = _SimpleObservation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        )

        # Do NOT call _preprocessing here; the PI model does it internally.
        return observation, action32

    def _clone_batch(self, batch):
        """Recursively clones all tensors inside a nested dictionary."""
        if isinstance(batch, dict):
            return {key: self._clone_batch(val) for key, val in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.clone()
        else:
            return batch  # Return as is for non-tensor types

    def _extract_xyz(self, x):
        """
        Extract xyz (3D position) and rotation from 6DoF or 6DoF+gripper actions.

        Supports:
        - 6: 6DoF (single arm)
        - 7: 6DoF + gripper (single arm)
        - 12: 2 arms × 6DoF
        - 14: 2 arms × (6DoF + gripper)

        Returns:
            xyz: Tensor with only xyz per arm (shape: ..., 3) or (..., 6) for dual-arm.
            rot: Tensor with only rotation per arm (shape: ..., 3) or (..., 6) for dual-arm.
        """
        if x.shape[-1] == 6:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 7:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 12:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 6:9]
            rot_left = x[..., 9:12]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        elif x.shape[-1] == 14:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 7:10]
            rot_left = x[..., 10:13]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        else:
            raise ValueError(f"Unexpected shape for 6DoF input: {x.shape}")
