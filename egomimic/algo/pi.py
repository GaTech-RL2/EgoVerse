import logging
import os
from collections import OrderedDict

import numpy as np
import openpi
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import safetensors
import torch
import torch.nn as nn
from overrides import override
from torchmetrics import MeanSquaredError

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
from egomimic.utils.egomimicUtils import (
    draw_actions,
    draw_annotation_text,
    draw_rotation_text,
)

logger = logging.getLogger(__name__)
# Ensure logger propagates to root logger and has appropriate level
# Child loggers inherit from parent, but we explicitly set level to ensure INFO messages appear
logger.setLevel(logging.INFO)
logger.propagate = True  # Explicitly enable propagation (default, but ensures it works)


class PI(Algo):
    """ """

    def __init__(
        self,
        data_schematic,
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
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic

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
            for key in data_schematic.keys_of_type("action_keys"):
                if (
                    data_schematic.is_key_with_embodiment(key, embodiment_id)
                    and key == self.ac_keys[embodiment]
                ):
                    self.ac_keys[embodiment_id] = key
            for key in data_schematic.keys_of_type("camera_keys"):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.camera_keys[embodiment_id].append(key)
            for key in data_schematic.keys_of_type("proprio_keys"):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.proprio_keys[embodiment_id].append(key)
            for key in data_schematic.keys_of_type("lang_keys"):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
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
            safetensors.torch.load_model(
                (
                    self.model.module
                    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                    else self.model
                ),
                model_path,
            )
        self.nets = nn.ModuleDict()
        self.nets["policy"] = self.model

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

        for embodiment_id, _batch in batch.items():
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.lerobot_key_to_keyname(
                    key, embodiment_id
                )
                if key_name is not None:
                    processed_batch[embodiment_id][key_name] = value

            # Carry through language tokenization tensors produced by collate_fn
            for tk in (
                "tokenized_prompt",
                "tokenized_mask",
                "token_loss_mask",
                "token_ar_mask",
            ):
                if tk in _batch:
                    processed_batch[embodiment_id][tk] = _batch[tk]

            ac_key = self.ac_keys[embodiment_id]
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError("Action shape in batch is not 2")

            B, S, _ = processed_batch[embodiment_id][ac_key].shape
            device = processed_batch[embodiment_id][ac_key].device
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=device
            )
            processed_batch[embodiment_id] = self.data_schematic.normalize_data(
                processed_batch[embodiment_id], embodiment_id
            )

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
            embodiment_name = get_embodiment(embodiment_id).lower()
            processed_obs, action = self._robomimic_to_pi_data(
                _batch,
                self.pi_cam_keys,
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
                embodiment_name = get_embodiment(embodiment_id).lower()
                processed_obs, action = self._robomimic_to_pi_data(
                    _batch,
                    self.pi_cam_keys,
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

                unnorm_actions = self.data_schematic.unnormalize_data(
                    predictions, embodiment_id
                )
                for key in unnorm_actions:
                    unnorm_preds[f"{embodiment_name}_{key}"] = unnorm_actions[key]

        return unnorm_preds

    @override
    def forward_eval_logging(self, batch):
        """
        Called by pl_model to generate a dictionary of metrics and an image visualization
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            metrics (dict):
                metricname: value (float)
            image: (B, 3, H, W)
        """
        preds = self.forward_eval(batch)
        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        for embodiment_id, _batch in batch.items():
            _batch = self.data_schematic.unnormalize_data(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id]
            pred_key = f"{embodiment_name}_{ac_key}"
            if pred_key in preds:
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(), _batch[ac_key][:, -1].cpu()
                )

            ims = self.visualize_preds(preds, _batch)
            images_dict[embodiment_id] = ims
        return metrics, images_dict

    @override
    def visualize_preds(self, predictions, batch):
        """
        Helper function to visualize predictions on top of images
        Args:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
            batch (dict): {ac_key: torch.Tensor (B, Seq, D), front_img_1: torch.Tensor (B, 3, H, W), embodiment: torch.Tensor (1)}
        Returns:
            ims (np.ndarray): (B, H, W, 3) - images with actions drawn on top
        """
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        ac_key = self.ac_keys[embodiment_id]

        viz_img_key = self.data_schematic.viz_img_key()[embodiment_id]
        ims = (batch[viz_img_key].cpu().numpy().transpose((0, 2, 3, 1)) * 255).astype(
            np.uint8
        )

        for key in batch:
            if f"{embodiment_name}_{key}" in predictions:
                preds = predictions[f"{embodiment_name}_{key}"]
                gt = batch[key]

                if self.is_6dof and ac_key == "actions_cartesian":
                    gt, gt_rot = self._extract_xyz(gt)
                    preds, preds_rot = self._extract_xyz(preds)

                for b in range(ims.shape[0]):
                    if preds.shape[-1] == 7 or preds.shape[-1] == 14:
                        ac_type = "joints"
                    elif preds.shape[-1] == 3 or preds.shape[-1] == 6:
                        ac_type = "xyz"
                    else:
                        raise ValueError(
                            f"Unknown action type with shape {preds.shape}"
                        )

                    arm = (
                        "right"
                        if preds.shape[-1] == 7 or preds.shape[-1] == 3
                        else "both"
                    )
                    ims[b] = draw_actions(
                        ims[b],
                        ac_type,
                        "Purples",
                        preds[b].cpu().numpy(),
                        self.camera_transforms.extrinsics,
                        self.camera_transforms.intrinsics,
                        arm=arm,
                    )
                    ims[b] = draw_actions(
                        ims[b],
                        ac_type,
                        "Greens",
                        gt[b].cpu().numpy(),
                        self.camera_transforms.extrinsics,
                        self.camera_transforms.intrinsics,
                        arm=arm,
                    )

                    if self.is_6dof and ac_key == "actions_cartesian":
                        ims[b] = draw_rotation_text(
                            ims[b], gt_rot[b][0], preds_rot[b][0], position=(340, 20)
                        )

                    if "annotations" in batch:
                        annotation = batch["annotations"][b]
                        ims[b] = draw_annotation_text(ims[b], annotation)

        return ims

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

        action = batch[ac_key]

        device = action.device

        present_flags = {
            k: (
                k in batch and isinstance(batch[k], torch.Tensor) and batch[k].ndim == 4
            )
            for k in cam_keys
        }

        emb_id = get_embodiment_id(embodiment)  # embodiment is a name string
        converter = self.action_registry.get(emb_id, ac_key)
        action32 = converter.to32(action)

        raw_images = _fill_missing_images(batch, cam_keys, device)

        # ---- Images (dict[str, Tensor]) ----
        images = {}
        for k in cam_keys:
            # keep your loop style, but pull from raw_images (always present)
            img = _ensure_bchw(raw_images[k])  # ensure BCHW
            img = _to_minus1_1(img)  # normalize to [-1, 1]
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

        if not lang_keys:
            tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = (
                _empty_lang_placeholders(B, device)
            )

        else:
            tokenized_prompt = batch["tokenized_prompt"].to(device)
            tokenized_prompt_mask = batch["tokenized_mask"].to(device)
            token_ar_mask = batch["token_ar_mask"].to(device)
            token_loss_mask = batch["token_loss_mask"].to(device)

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
