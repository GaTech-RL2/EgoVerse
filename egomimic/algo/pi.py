from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from functools import partial
from typing import List, Optional
import numpy as np
import einops
from torchmetrics import MeanSquaredError

from egomimic.models.hpt_nets import *
from egomimic.algo.algo import Algo
from egomimic.utils.egomimicUtils import draw_actions

from egomimic.utils.egomimicUtils import draw_actions, draw_rotation_text

import numpy as np

from overrides import override

from egomimic.algo.algo import Algo

from rldb.utils import get_embodiment_id, get_embodiment

from termcolor import cprint

import os

import openpi
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

import safetensors

class PI(Algo):
    """
    """
    def __init__(
        self,
        data_schematic,
        camera_transforms,
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
        **kwargs
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic

        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        self.config = config
        
        self.ac_keys = ac_keys
        
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=self.config.pytorch_training_precision,
            action_dim=self.config.model.action_dim,
            action_horizon=self.config.model.action_horizon,
            max_token_len=self.config.model.max_token_len,
            paligemma_variant=getattr(self.config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(self.config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
        self.model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(self.device)
        
        if self.config.pytorch_weight_path is not None:
            model_path = os.path.join(self.config.pytorch_weight_path, "model.safetensors")
            safetensors.torch.load_model(
                (self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model), model_path
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
            breakpoint()
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.lerobot_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    processed_batch[embodiment_id][key_name] = value
            
            ac_key = self.ac_keys[embodiment_id]
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError("Action shape in batch is not 2")
            
            B, S, _ = processed_batch[embodiment_id][ac_key].shape
            device = processed_batch[embodiment_id][ac_key].device
            processed_batch[embodiment_id]["pad_mask"]  = torch.ones(B, S, 1, device=device)
            processed_batch[embodiment_id] = self.data_schematic.normalize_data(processed_batch[embodiment_id], embodiment_id)
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

        predictions = OrderedDict()

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
        ims = {}
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
        total_action_loss = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        loss_dict["action_loss"] = total_action_loss
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

    def _robomimic_to_pi_data(self, batch, cam_keys, proprio_keys, lang_keys, ac_key):
        data = {}
        data["action"] = batch[ac_key]
        for key in cam_keys:
            if key in batch:
                _data = batch[key]
                if not torch.all(_data == 0):
                    if self.nets.training and key in self.encoders:
                        _data = self.train_image_augs(_data)
                    elif self.eval_image_augs and key in self.encoders:
                        _data = self.eval_image_augs(_data)

        
    def _clone_batch(self, batch):
        """ Recursively clones all tensors inside a nested dictionary. """
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
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat([rot_right, rot_left], dim=-1)
        elif x.shape[-1] == 14:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 7:10]
            rot_left = x[..., 10:13]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat([rot_right, rot_left], dim=-1)
        else:
            raise ValueError(f"Unexpected shape for 6DoF input: {x.shape}")
