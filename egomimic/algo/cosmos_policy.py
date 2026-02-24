from collections import OrderedDict
import copy
import logging

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from overrides import override
from omegaconf import DictConfig, OmegaConf
import importlib
import pickle

from egomimic.algo.algo import Algo
from egomimic.rldb.utils import get_embodiment_id, get_embodiment

from cosmos_policy.models.policy_video2world_model import (
    CosmosPolicyVideo2WorldModel,
    CosmosPolicyVideo2WorldConfig,
)
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from cosmos_policy.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from cosmos_policy.config.conditioner.video2world_conditioner import Video2WorldConditioner
from cosmos_policy._src.predict2.conditioner import BooleanFlag, ReMapkey, TextAttr
from cosmos_policy._src.predict2.models.text2world_model import EMAConfig
from cosmos_policy._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_policy._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_policy.datasets.dataset_utils import preprocess_image, resize_images
from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence

logger = logging.getLogger(__name__)

# Default net spec for Cosmos-Predict2-2B checkpoint (cosmos_v1_2B / MinimalV1LVGDiT for policy).
# Aligned with cosmos-policy libero / Stage-102. Used when checkpoint_path is set and net is omitted.
DEFAULT_NET_FROM_CHECKPOINT = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=dict(mode="predict2_2b_720"),
)

# Small net for debug runs (2 blocks, 1024 dim). Use with model_config.use_mini_net: true to avoid OOM.
# Weights from a 2B checkpoint will not load into this net; use for pipeline/debug only.
MINI_NET_FOR_DEBUG = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=1024,
    num_blocks=2,
    num_heads=8,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=dict(mode="block_wise"),
)


class CosmosPolicy(Algo):
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
        # Cosmos Policy specific params
        # ---------------------------
        model_config,
        action_chunk: int = 25,
        use_proprio: bool = True,
        use_values: bool = False,
        final_image_size: int = 224,
        num_duplicates_per_image: int = 4,
        normalize_images=False,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        demonstration_sampling_prob: float = 0.5,
        p_world_model: float = 0.5,
        return_value_function_returns: bool = False,
        gamma: float = 0.998,
        val_num_inference_steps: int = 35,
        debug: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        self.domains = domains
        self.device = None
        
        # Cosmos Policy specific parameters
        self.action_chunk = action_chunk
        self.use_proprio = use_proprio
        self.use_values = use_values
        self.final_image_size = final_image_size
        self.num_duplicates_per_image = num_duplicates_per_image
        self.model_config = model_config
        self.normalize_images = normalize_images
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.p_world_model = p_world_model
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.val_num_inference_steps = val_num_inference_steps
        
        self.debug = debug
        
        # Initialize camera, proprio, and language keys per embodiment
        self.ac_keys = {}
        self.camera_keys = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        
        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            
            action_keys = data_schematic.df[
                (data_schematic.df["key_type"] == "action_keys") & 
                (data_schematic.df["embodiment"] == embodiment_id)
            ]["key_name"].tolist()
            
            camera_keys = data_schematic.df[
                (data_schematic.df["key_type"] == "camera_keys") & 
                (data_schematic.df["embodiment"] == embodiment_id)
            ]["key_name"].tolist()
            
            proprio_keys = data_schematic.df[
                (data_schematic.df["key_type"] == "proprio_keys") & 
                (data_schematic.df["embodiment"] == embodiment_id)
            ]["key_name"].tolist()
            
            lang_keys = data_schematic.df[
                (data_schematic.df["key_type"] == "lang_keys") & 
                (data_schematic.df["embodiment"] == embodiment_id)
            ]["key_name"].tolist()
            
            # Set the keys for this embodiment
            self.ac_keys[embodiment_id] = action_keys
            self.camera_keys[embodiment_id] = camera_keys
            self.proprio_keys[embodiment_id] = proprio_keys
            self.lang_keys[embodiment_id] = lang_keys
        
        # Cosmos Policy model initialization
        cosmos_config = self._get_cosmos_policy_config()
        self.model = CosmosPolicyVideo2WorldModel(config=cosmos_config)
        self.nets["policy"] = self.model
        self.model_config = self.model.config if hasattr(self.model, 'config') else None

        # Load checkpoint weights when checkpoint_path was set (same as cosmos-policy).
        if getattr(self, "_checkpoint_path", None):
            self._load_checkpoint_weights()
    
    
    def _initialize_model(self, device):
        """Move the cosmos_policy model to the given device."""
        # Model is already initialized in __init__, just move to device
        if self.model is not None:
            self.model = self.model.to(device)
            self.device = device
    
    def _get_cosmos_policy_config(self):
        """
        Create a CosmosPolicyVideo2WorldConfig object from the model_config.
        
        This method handles the conversion of nested config dictionaries to their proper types:
        - sde: dict -> L(HybridEDMSDE)(...) LazyCall
        - tokenizer: dict -> L(Wan2pt1VAEInterface)(...) LazyCall
        - conditioner: dict -> L(Video2WorldConditioner)(...) LazyCall with fps, padding_mask, use_video_condition
            (same as video_prediction_conditioner) and nested text -> L(TextAttr)(...) from config
        - ema: dict -> EMAConfig object
        - net: dict -> L(MinimalV1LVGDiT)(...) LazyCall (with nested sac_config -> L(SACConfig)(...))
        
        Returns:
            CosmosPolicyVideo2WorldConfig: Properly configured config object with all nested
                configs converted to the correct types
        """
        model_config_dict = OmegaConf.to_container(self.model_config, resolve=True)

        # Pop checkpoint_path so it is not passed to CosmosPolicyVideo2WorldConfig; store for later loading.
        self._checkpoint_path = model_config_dict.pop("checkpoint_path", None)
        self._use_mini_net = model_config_dict.pop("use_mini_net", False)
        if model_config_dict.get("net") is None or model_config_dict.get("net") == {}:
            if self._use_mini_net:
                model_config_dict["net"] = copy.deepcopy(MINI_NET_FOR_DEBUG)
            elif self._checkpoint_path is not None:
                model_config_dict["net"] = copy.deepcopy(DEFAULT_NET_FROM_CHECKPOINT)
        
        # Convert SDE config dict to LazyCall (required by config class)
        if "sde" in model_config_dict:
            sde_config = model_config_dict["sde"]
            model_config_dict["sde"] = L(HybridEDMSDE)(**sde_config)
        
        # Convert tokenizer config dict to LazyCall (required by config class - expects LazyDict)
        if "tokenizer" in model_config_dict:
            tokenizer_config = model_config_dict["tokenizer"]
            model_config_dict["tokenizer"] = L(Wan2pt1VAEInterface)(**tokenizer_config)
        
        # Convert conditioner config dict to LazyCall. Align with cosmos-policy video_prediction_conditioner
        # (fps, padding_mask, use_video_condition) so condition gets them from batch and DiT receives valid padding_mask.
        if "conditioner" in model_config_dict:
            conditioner_config = model_config_dict["conditioner"]
            if isinstance(conditioner_config, dict):
                # Start with shared defaults (same as VideoPredictionConditioner) so padding_mask/fps are in condition
                processed_conditioner = {
                    "fps": L(ReMapkey)(
                        input_key="fps",
                        output_key="fps",
                        dropout_rate=0.0,
                        dtype=None,
                    ),
                    "padding_mask": L(ReMapkey)(
                        input_key="padding_mask",
                        output_key="padding_mask",
                        dropout_rate=0.0,
                        dtype=None,
                    ),
                    "use_video_condition": L(BooleanFlag)(
                        input_key="fps",
                        output_key="use_video_condition",
                        dropout_rate=0.2,
                    ),
                }
                # Apply user overrides (e.g. text with dropout_rate=0.0)
                for key, value in conditioner_config.items():
                    if key == "text":
                        text_config = {"input_key": ["t5_text_embeddings"], "use_empty_string": False}
                        text_config.update(value)  # Override with provided values
                        processed_conditioner[key] = L(TextAttr)(**text_config)
                    else:
                        processed_conditioner[key] = value
                model_config_dict["conditioner"] = L(Video2WorldConditioner)(**processed_conditioner)
        
        # Convert ema config dict to EMAConfig object (required by config class)
        if "ema" in model_config_dict:
            ema_config = model_config_dict["ema"]
            model_config_dict["ema"] = EMAConfig(**ema_config)
        
        # Convert net config dict to LazyCall (required by config class - expects LazyDict)
        if "net" in model_config_dict:
            net_config = model_config_dict["net"]
            if isinstance(net_config, dict):
                processed_net = {}
                for key, value in net_config.items():
                    if key == "sac_config":
                        processed_net[key] = L(SACConfig)(**value)
                    else:
                        processed_net[key] = value
                model_config_dict["net"] = L(MinimalV1LVGDiT)(**processed_net)
        
        # Filter out None values to use config class defaults
        filtered_config = {k: v for k, v in model_config_dict.items() if v is not None}
        cosmos_config = CosmosPolicyVideo2WorldConfig(**filtered_config)
        
        return cosmos_config

    def _load_checkpoint_weights(self):
        """Load model weights from checkpoint_path (same resolution as cosmos-policy)."""
        if getattr(self, "_use_mini_net", False):
            logger.info("Skipping checkpoint load (use_mini_net=True); net is randomly initialized for debug.")
            return
        local_path = get_checkpoint_path(self._checkpoint_path)
        logger.info("Loading Cosmos Policy checkpoint from %s", local_path)
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            model_state = ckpt["model"]
        elif "state_dict" in ckpt:
            model_state = ckpt["state_dict"]
        else:
            model_state = ckpt
        incompatible = self.model.load_state_dict(model_state, strict=False)
        if incompatible is not None:
            if incompatible.missing_keys:
                logger.warning("Checkpoint missing keys (first 10): %s", incompatible.missing_keys[:10])
            if incompatible.unexpected_keys:
                logger.warning("Checkpoint unexpected keys (first 10): %s", incompatible.unexpected_keys[:10])
        logger.info("Loaded checkpoint from %s", local_path)
    
    @override
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            batch (dict): processed dict of batches organized by embodiment_id
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
            
            ac_key = self.ac_keys[embodiment_id][0]
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError("Action shape in batch is not 3 (B, S, D)")
            
            B, S, _ = processed_batch[embodiment_id][ac_key].shape
            device = processed_batch[embodiment_id][ac_key].device
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=device
            )
            processed_batch[embodiment_id] = self.data_schematic.normalize_data(
                processed_batch[embodiment_id], embodiment_id
            )
        return processed_batch
    
    
    def _robomimic_to_cosmos_policy_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_keys, embodiment
    ):
        """
        Transform robomimic single-frame batch to cosmos policy video sequence format.
        
        Args:
            batch: dict with keys like front_img_1, left_wrist_img, right_wrist_img,
                   joint_positions, actions_cartesian, etc.
            cam_keys: list of camera key names
            proprio_keys: list of proprioception key names
            lang_keys: list of language key names
            ac_keys: list of action key names
            embodiment: embodiment name string
            
        Returns:
            dict: cosmos policy format batch with video, actions, embeddings, etc.
        """
        device = batch[ac_keys[0]].device
        B = batch[ac_keys[0]].shape[0]
        
        # ---------------------------------------
        # Demo / rollout masks
        # ---------------------------------------
        # Priority 1 – explicit labels from S3RLDBDataset (separate demo/rollout datasets).
        #   Each sample carries rollout_data_mask=0/1, rollout_data_success_mask, and
        #   world_model_sample_mask / value_function_sample_mask set at __getitem__ time,
        #   mirroring the ALOHADataset approach.
        # Priority 2 – deterministic pattern across accumulated micro-batches (single mixed
        #   dataset, but accumulate_grad_batches > 1).
        # Priority 3 – fully random (legacy / single-sample fallback).
        if "rollout_data_mask" in batch:
            rollout_data_mask = batch["rollout_data_mask"].to(torch.int64).to(device)
        else:
            micro_pos = getattr(self, '_micro_batch_position', None)
            acc = getattr(self, '_accumulate_grad_batches', 1)
            if micro_pos is not None and acc > 1:
                # Guarantee both demo AND rollout appear within each optimizer step.
                n_demo = max(1, min(round(acc * self.demonstration_sampling_prob), acc - 1))
                is_rollout_val = int(micro_pos >= n_demo)
                rollout_data_mask = torch.full((B,), is_rollout_val, dtype=torch.int64, device=device)
            else:
                demo_rollout_rand = torch.rand(B, device=device)
                rollout_data_mask = (demo_rollout_rand >= self.demonstration_sampling_prob).to(torch.int64)

        if "rollout_data_success_mask" in batch:
            rollout_data_success_mask = batch["rollout_data_success_mask"].to(torch.int64).to(device)
        else:
            success_rand = torch.rand(B, device=device)
            rollout_data_success_mask = (
                (rollout_data_mask == 1) & (success_rand >= 0.5)
            ).to(torch.int64)

        if "world_model_sample_mask" in batch and "value_function_sample_mask" in batch:
            world_model_sample_mask = batch["world_model_sample_mask"].to(torch.int64).to(device)
            value_function_sample_mask = batch["value_function_sample_mask"].to(torch.int64).to(device)
        else:
            world_model_sample_mask = torch.zeros(B, dtype=torch.int64, device=device)
            value_function_sample_mask = torch.zeros(B, dtype=torch.int64, device=device)
            if self.return_value_function_returns:
                rollout_indices = (rollout_data_mask == 1)
                if rollout_indices.any():
                    world_model_rand = torch.rand(B, device=device)
                    world_model_mask = (world_model_rand < self.p_world_model)
                    world_model_sample_mask[rollout_indices] = world_model_mask[rollout_indices].to(torch.int64)
                    value_function_sample_mask[rollout_indices] = (~world_model_mask)[rollout_indices].to(torch.int64)
            else:
                world_model_sample_mask = rollout_data_mask.clone()
                value_function_sample_mask = torch.zeros(B, dtype=torch.int64, device=device)
        
        # Generate global_rollout_idx: -1 for demos, random indices for rollouts
        global_rollout_idx = torch.full((B,), -1, dtype=torch.int64, device=device)
        rollout_indices = (rollout_data_mask == 1)
        if rollout_indices.any():
            num_rollouts = rollout_indices.sum().item()
            global_rollout_idx[rollout_indices] = torch.randint(
                0, 10000, (num_rollouts,), device=device, dtype=torch.int64
            )
        
        # ---------------------------------------
        # Concatenation priprio and actions
        # ---------------------------------------
        # Concatenate current proprio keys
        curr_proprio_list = []
        for proprio_key in proprio_keys:
            if proprio_key in batch and "future" not in proprio_key.lower():
                curr_proprio_list.append(batch[proprio_key])
        if curr_proprio_list:
            curr_priprios = torch.cat(curr_proprio_list, dim=-1)  # Concatenate along feature dimension
        
        # Concatenate future proprio keys
        future_proprio_list = []
        for proprio_key in proprio_keys:
            if proprio_key in batch and "future" in proprio_key.lower():
                future_proprio_list.append(batch[proprio_key])
        if future_proprio_list:
            future_priprios = torch.cat(future_proprio_list, dim=-1)  # Concatenate along feature dimension
        
        # Concatenate current action keys
        curr_action_list = []
        for ac_key in ac_keys:
            if ac_key in batch and "future" not in ac_key.lower():
                curr_action_list.append(batch[ac_key])
        if curr_action_list:
            curr_actions = torch.cat(curr_action_list, dim=-1)  # Concatenate along feature dimension
        
        # Concatenate future action keys
        future_action_list = []
        for ac_key in ac_keys:
            if ac_key in batch and "future" in ac_key.lower():
                future_action_list.append(batch[ac_key])
        if future_action_list:
            future_actions = torch.cat(future_action_list, dim=-1)  # Concatenate along feature dimension
        
        
        # Helper function to convert torch tensor (C, H, W) to numpy (H, W, C) uint8
        def tensor_to_numpy_uint8(img_tensor):
            """Convert torch tensor (C, H, W) to numpy (H, W, C) uint8."""
            if img_tensor.dtype == torch.uint8:
                img_np = img_tensor.cpu().numpy()
            elif img_tensor.dtype in (torch.float32, torch.float64):
                # Assume range [0, 1] or [0, 255]
                img_np = img_tensor.cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            else:
                img_np = img_tensor.cpu().numpy().astype(np.uint8)
            # Permute from (C, H, W) to (H, W, C)
            if img_np.shape[0] == 3 or img_np.shape[0] == 1:
                img_np = np.transpose(img_np, (1, 2, 0))
                # Ensure 3 channels
                if img_np.shape[-1] == 1:
                    img_np = np.repeat(img_np, 3, axis=-1)
            return img_np
        
        # Ensure frames are the expected size
        def _ensure_size(img: np.ndarray):
            if img.shape[0] != self.final_image_size or img.shape[1] != self.final_image_size:
                return resize_images(np.expand_dims(img, axis=0), self.final_image_size).squeeze(0)
            return img
        
        # ---------------------------------------
        # Process all batch samples
        # ---------------------------------------
        # Lists to collect results for each batch sample
        batch_videos = []
        batch_actions = []
        batch_future_actions = []
        batch_proprios = []
        batch_future_proprios = []
        batch_value_returns = []
        batch_next_value_returns = []
        
        # Latent indices (same for all batch samples, will be set in first iteration)
        action_latent_idx = -1
        value_latent_idx = -1
        current_proprio_latent_idx = -1
        current_wrist_image_latent_idx = -1
        current_wrist_image2_latent_idx = -1
        current_image_latent_idx = -1
        future_proprio_latent_idx = -1
        future_wrist_image_latent_idx = -1
        future_wrist_image2_latent_idx = -1
        future_image_latent_idx = -1
        
        # Process each batch sample
        for batch_idx in range(B):
            # Extract images for this batch sample
            primary_img = None
            left_wrist_img = None
            right_wrist_img = None
            future_primary_img = None
            future_left_wrist_img = None
            future_right_wrist_img = None
            
            for cam_key in cam_keys:
                if cam_key in batch:
                    img = batch[cam_key][batch_idx]  # (C, H, W)
                    if "front" in cam_key.lower():
                        if "future" in cam_key.lower():
                            future_primary_img = tensor_to_numpy_uint8(img)
                        else:
                            primary_img = tensor_to_numpy_uint8(img)
                    elif "left_wrist" in cam_key.lower():
                        if "future" in cam_key.lower():
                            future_left_wrist_img = tensor_to_numpy_uint8(img)
                        else:
                            left_wrist_img = tensor_to_numpy_uint8(img)
                    elif "right_wrist" in cam_key.lower():
                        if "future" in cam_key.lower():
                            future_right_wrist_img = tensor_to_numpy_uint8(img)
                        else:
                            right_wrist_img = tensor_to_numpy_uint8(img)
            
            # Ensure frames are the expected size
            primary_img = _ensure_size(primary_img)
            left_wrist_img = _ensure_size(left_wrist_img) if left_wrist_img is not None else None
            right_wrist_img = _ensure_size(right_wrist_img) if right_wrist_img is not None else None
            future_primary_img = _ensure_size(future_primary_img) if future_primary_img is not None else None
            future_left_wrist_img = _ensure_size(future_left_wrist_img) if future_left_wrist_img is not None else None
            future_right_wrist_img = _ensure_size(future_right_wrist_img) if future_right_wrist_img is not None else None
            
            # ---------------------------------------
            # Injection video sequences
            # ---------------------------------------
            # Build a list of unique frames (no per-frame duplication) and per-frame repeat counts
            # We'll preprocess the unique frames once (same aug params across the whole sequence),
            # then expand by repeat counts to produce the final sequence.
            frames = []  # list of np.ndarray frames with shape (H, W, C)
            repeats = []  # list of ints with how many times to repeat each frame in time dimension
            segment_idx = 0  # logical segment index used for *_latent_idx
            # Reset indices for this batch sample (will be set to same values for all samples)
            action_latent_idx_b = -1
            value_latent_idx_b = -1
            current_proprio_latent_idx_b = -1
            current_wrist_image_latent_idx_b = -1
            current_wrist_image2_latent_idx_b = -1
            current_image_latent_idx_b = -1
            future_proprio_latent_idx_b = -1
            future_wrist_image_latent_idx_b = -1
            future_wrist_image2_latent_idx_b = -1
            future_image_latent_idx_b = -1
            
            # (1) Add blank first input image (needed for the tokenizer)
            ref_image_for_shape = copy.deepcopy(primary_img)
            blank_first_input_frame = np.zeros_like(ref_image_for_shape)
            frames.append(blank_first_input_frame)
            repeats.append(1)
            segment_idx += 1
            
            # (2) Add current proprio
            if self.use_proprio:
                blank_proprio_image = np.zeros_like(ref_image_for_shape)
                current_proprio_latent_idx_b = segment_idx
                frames.append(blank_proprio_image)
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
            
            # (3) Add current left wrist image
            if left_wrist_img is not None:
                current_wrist_image_latent_idx_b = segment_idx
                frames.append(left_wrist_img)
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
            
            # (4) Add current right wrist image
            if right_wrist_img is not None:
                current_wrist_image2_latent_idx_b = segment_idx
                frames.append(right_wrist_img)
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
            
            # (5) Add current primary image
            current_image_latent_idx_b = segment_idx
            frames.append(primary_img)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
            
            # (6) Add blank image for action chunk
            blank_action_image = np.zeros_like(ref_image_for_shape)
            action_latent_idx_b = segment_idx
            frames.append(blank_action_image)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
            
            # (7) Add future proprio
            if self.use_proprio:
                blank_proprio_image = np.zeros_like(ref_image_for_shape)
                future_proprio_latent_idx_b = segment_idx
                frames.append(blank_proprio_image)
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
            
            # (8) Add future left wrist image
            if future_left_wrist_img is not None:
                future_wrist_image_latent_idx_b = segment_idx
                frames.append(future_left_wrist_img)
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
            
            # (9) Add future right wrist image
            if future_right_wrist_img is not None:
                future_wrist_image2_latent_idx_b = segment_idx
                frames.append(future_right_wrist_img)
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
            
            # (10) Add future primary image
            future_image_latent_idx_b = segment_idx
            frames.append(future_primary_img)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
            
            # (11) Add blank value image
            if self.use_values:
                value_image = np.zeros_like(ref_image_for_shape)
                value_latent_idx_b = segment_idx
                frames.append(value_image)
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
            else:
                value_latent_idx_b = -1
            
            # Store indices from first batch sample (they're the same for all)
            if batch_idx == 0:
                action_latent_idx = action_latent_idx_b
                value_latent_idx = value_latent_idx_b
                current_proprio_latent_idx = current_proprio_latent_idx_b
                current_wrist_image_latent_idx = current_wrist_image_latent_idx_b
                current_wrist_image2_latent_idx = current_wrist_image2_latent_idx_b
                current_image_latent_idx = current_image_latent_idx_b
                future_proprio_latent_idx = future_proprio_latent_idx_b
                future_wrist_image_latent_idx = future_wrist_image_latent_idx_b
                future_wrist_image2_latent_idx = future_wrist_image2_latent_idx_b
                future_image_latent_idx = future_image_latent_idx_b
            
            # Sanity: segment indices must be within [0, len(frames)-1]
            num_segments = len(frames)
            for name, val in (
                ("action_latent_idx", action_latent_idx_b),
                ("value_latent_idx", value_latent_idx_b),
                ("current_proprio_latent_idx", current_proprio_latent_idx_b if self.use_proprio else -1),
                ("current_wrist_image_latent_idx", current_wrist_image_latent_idx_b),
                ("current_wrist_image2_latent_idx", current_wrist_image2_latent_idx_b),
                ("current_image_latent_idx", current_image_latent_idx_b),
                ("future_proprio_latent_idx", future_proprio_latent_idx_b if self.use_proprio else -1),
                ("future_wrist_image_latent_idx", future_wrist_image_latent_idx_b),
                ("future_wrist_image2_latent_idx", future_wrist_image2_latent_idx_b),
                ("future_image_latent_idx", future_image_latent_idx_b)
            ):
                if val != -1:
                    assert 0 <= val < num_segments, f"{name}={val} out of range for num_segments={num_segments}"
            
            # Concatenate unique frames and preprocess once
            all_unique_images = np.stack(frames, axis=0) # (num_segments, H, W, C)
            all_unique_images = preprocess_image(
                all_unique_images,
                final_image_size=self.final_image_size,
                normalize_images=self.normalize_images,
                use_image_aug=self.use_image_aug,
                stronger_image_aug=self.use_stronger_image_aug,
            )
            
            # Expand unique preprocessed images by repeat counts along time dimension
            # all_unique_images after preprocess_image: (C, num_segments, H, W)
            lengths = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
            all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)  # (C, T_total, H, W) where T_total = sum(repeats)
            # Sanity: expanded length matches repeats sum
            assert all_images.shape[1] == int(lengths.sum().item()), "Expanded T does not match repeats sum"
            
            # all_images is now (C, T_total, H, W), add batch dimension: (1, C, T_total, H, W)
            all_images = all_images.unsqueeze(0)  # (1, C, T_total, H, W)
            batch_videos.append(all_images)
            
            # Collect actions and proprio for this batch sample
            curr_action_chunk = curr_actions[batch_idx][:self.action_chunk, :]  # (action_chunk, action_dim)
            future_action_chunk = future_actions[batch_idx][:self.action_chunk, :]  # (action_chunk, action_dim)
            batch_actions.append(curr_action_chunk)
            batch_future_actions.append(future_action_chunk)
            
            if self.use_proprio:
                proprio = curr_priprios[batch_idx].cpu().numpy()
                future_priprio = future_priprios[batch_idx].cpu().numpy()
                batch_proprios.append(torch.from_numpy(proprio).to(device))
                batch_future_proprios.append(torch.from_numpy(future_priprio).to(device))
        
        # Stack all batch samples
        # Video: stack (1, C, T, H, W) tensors -> (B, C, T, H, W)
        all_videos = torch.cat(batch_videos, dim=0)  # (B, C, T, H, W)
        
        # Actions: stack (action_chunk, action_dim) tensors -> (B, action_chunk, action_dim)
        all_actions = torch.stack(batch_actions, dim=0)  # (B, action_chunk, action_dim)
        all_future_actions = torch.stack(batch_future_actions, dim=0)  # (B, action_chunk, action_dim)
        
        # Proprio: stack (proprio_dim,) tensors -> (B, proprio_dim)
        if self.use_proprio:
            all_proprios = torch.stack(batch_proprios, dim=0)  # (B, proprio_dim)
            all_future_proprios = torch.stack(batch_future_proprios, dim=0)  # (B, proprio_dim)
        else:
            all_proprios = None
            all_future_proprios = None
        
        # Value function returns – Monte-Carlo discounted returns.
        #
        # When return_value_function_returns=True AND the batch contains episode_length
        # and frame_index (added by S3RLDBDataset.__getitem__), we compute the scalar
        # return on-the-fly for each sample, mirroring ALOHADataset's logic:
        #   returns[t] = gamma^(T-1-t) * terminal_reward  (rescaled to [-1, 1])
        #
        # terminal_reward:
        #   - demo (rollout_data_mask=0)          → 1.0  (all demos are successes)
        #   - successful rollout (success_mask=1)  → 1.0
        #   - failed rollout    (success_mask=0)   → 0.0  → return = -1 everywhere
        #
        # We compute the return at TWO timesteps:
        #   value_function_return      → future_frame_idx = frame_index + action_chunk
        #   next_value_function_return → next_future_frame_idx = frame_index + 2 * action_chunk
        # Both are clamped to [0, episode_length - 1].
        #
        # Gamma uses ALOHA's default (0.998) which is also our config default.
        _gamma = getattr(self, "gamma", 0.998)

        def _mc_return(ep_len: int, future_t: int, terminal_reward: float) -> float:
            """Return the rescaled Monte-Carlo return at timestep future_t."""
            if ep_len <= 0 or future_t < 0:
                return float("-100")
            t = min(future_t, ep_len - 1)
            raw = (_gamma ** (ep_len - 1 - t)) * terminal_reward
            if terminal_reward > 0:
                return float(2.0 * raw / terminal_reward - 1.0)  # rescale to [-1, 1]
            return -1.0  # failure episode → -1 everywhere

        if (
            self.return_value_function_returns
            and "frame_index" in batch
            and "episode_length" in batch
        ):
            vf_returns, next_vf_returns = [], []
            for b_i in range(B):
                frame_idx = int(batch["frame_index"][b_i].item())
                ep_len    = int(batch["episode_length"][b_i].item())
                is_demo   = rollout_data_mask[b_i].item() == 0
                is_succ   = rollout_data_success_mask[b_i].item() == 1
                t_reward  = 1.0 if (is_demo or is_succ) else 0.0

                future_t      = frame_idx + self.action_chunk
                next_future_t = frame_idx + 2 * self.action_chunk

                vf_returns.append(_mc_return(ep_len, future_t, t_reward))
                next_vf_returns.append(_mc_return(ep_len, next_future_t, t_reward))

            all_value_returns      = torch.tensor(vf_returns,      dtype=torch.float32, device=device)
            all_next_value_returns = torch.tensor(next_vf_returns, dtype=torch.float32, device=device)
        else:
            all_value_returns      = torch.full((B,), float("-100"), dtype=torch.float32, device=device)
            all_next_value_returns = torch.full((B,), float("-100"), dtype=torch.float32, device=device)
        
        # ---------------------------------------
        # Construct text embeddings
        # ---------------------------------------
        #TODO: hard code for now
        text_embeddings_path = '/coc/flash7/scratch/egowm/wmprocessedDataset/t5_embeddings.pkl'
        with open(text_embeddings_path, 'rb') as f:
            text_embeddings_dict = pickle.load(f)
        command = next(iter(text_embeddings_dict))
        t5_text_embeddings = text_embeddings_dict[command]  # (1, 512, 1024)
        # Broadcast to batch size: (1, 512, 1024) -> (B, 512, 1024)
        t5_text_embeddings_batched = t5_text_embeddings.repeat(B, 1, 1).to(device)
        
        # Latent indices are the same for all batch samples (they're determined by the sequence structure)
        # Convert to batched tensors
        action_latent_idx_tensor = torch.full((B,), action_latent_idx, dtype=torch.int64, device=device)
        value_latent_idx_tensor = torch.full((B,), value_latent_idx, dtype=torch.int64, device=device)
        current_proprio_latent_idx_tensor = torch.full((B,), current_proprio_latent_idx if self.use_proprio else -1, dtype=torch.int64, device=device)
        current_wrist_image_latent_idx_tensor = torch.full((B,), current_wrist_image_latent_idx, dtype=torch.int64, device=device)
        current_wrist_image2_latent_idx_tensor = torch.full((B,), current_wrist_image2_latent_idx, dtype=torch.int64, device=device)
        current_image_latent_idx_tensor = torch.full((B,), current_image_latent_idx, dtype=torch.int64, device=device)
        future_proprio_latent_idx_tensor = torch.full((B,), future_proprio_latent_idx if self.use_proprio else -1, dtype=torch.int64, device=device)
        future_wrist_image_latent_idx_tensor = torch.full((B,), future_wrist_image_latent_idx, dtype=torch.int64, device=device)
        future_wrist_image2_latent_idx_tensor = torch.full((B,), future_wrist_image2_latent_idx, dtype=torch.int64, device=device)
        future_image_latent_idx_tensor = torch.full((B,), future_image_latent_idx, dtype=torch.int64, device=device)
        
        # ---------------------------------------
        # Make final data
        # ---------------------------------------
        cosmos_batch = {
            "video": all_videos,  # (B, C, T, H, W)
            "actions": all_actions,  # (B, action_chunk, action_dim)
            "t5_text_embeddings": t5_text_embeddings_batched,  # (B, 512, 1024)
            "t5_text_mask": torch.ones(B, 512, dtype=torch.int64, device=device),  # (B, 512)
            "fps": torch.full((B,), 30, dtype=torch.float32, device=device),  # (B,)
            "padding_mask": torch.zeros(B, 1, self.final_image_size, self.final_image_size, device=device),  # (B, 1, H, W)
            "image_size": self.final_image_size * torch.ones(B, 4, device=device),  # (B, 4)
            "proprio": all_proprios,  # (B, proprio_dim) or None
            "future_proprio": all_future_proprios,  # (B, proprio_dim) or None
            "value_function_return": all_value_returns,  # (B,)
            "next_action_chunk": all_future_actions,  # (B, action_chunk, action_dim)
            "next_value_function_return": all_next_value_returns,  # (B,)
            "action_latent_idx": action_latent_idx_tensor,  # (B,)
            "value_latent_idx": value_latent_idx_tensor,  # (B,)
            "current_proprio_latent_idx": current_proprio_latent_idx_tensor,  # (B,)
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx_tensor,  # (B,)
            "current_wrist_image2_latent_idx": current_wrist_image2_latent_idx_tensor,  # (B,)
            "current_image_latent_idx": current_image_latent_idx_tensor,  # (B,)
            "future_proprio_latent_idx": future_proprio_latent_idx_tensor,  # (B,)
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx_tensor,  # (B,)
            "future_wrist_image2_latent_idx": future_wrist_image2_latent_idx_tensor,  # (B,)
            "future_image_latent_idx": future_image_latent_idx_tensor,  # (B,)
            "rollout_data_mask": rollout_data_mask,  # (B,)
            "rollout_data_success_mask": rollout_data_success_mask,  # (B,)
            "world_model_sample_mask": world_model_sample_mask,  # (B,)
            "value_function_sample_mask": value_function_sample_mask,  # (B,)
            "global_rollout_idx": global_rollout_idx,  # (B,)
        }
        
        return cosmos_batch
    
    @override
    def forward_training(self, batch):
        """
        One iteration of training. Compute forward pass and compute losses.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): {ac_key: Tensor, loss_key: Tensor, cosmos sub-losses…}
        """
        if self.device is None:
            first_tensor = next(iter(batch.values())) if isinstance(batch, dict) else None
            if first_tensor is not None:
                self.device = first_tensor.device
                self._initialize_model(self.device)

        predictions = OrderedDict()

        for embodiment_id, _batch in batch.items():
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_keys = self.ac_keys[embodiment_id]
            embodiment_name = get_embodiment(embodiment_id).lower()

            cosmos_batch = self._robomimic_to_cosmos_policy_data(
                _batch, cam_keys, proprio_keys, lang_keys, ac_keys, embodiment_name
            )

            iteration = getattr(self, "_current_iteration", 0)

            # Pre-call NaN diagnostics (for debug)
            if self.debug:
                if isinstance(cosmos_batch, dict):
                    nan_input_keys = [
                        k for k, v in cosmos_batch.items()
                        if torch.is_tensor(v) and torch.isnan(v).any()
                    ]
                    if nan_input_keys:
                        logger.warning(
                            "[iter %d] NaN in cosmos_batch inputs for embodiment '%s': %s",
                            iteration, embodiment_name, nan_input_keys,
                        )

            output_batch, loss = self.model.training_step(cosmos_batch, iteration)

            # Post-call NaN diagnostics (for debug)
            if self.debug:
                if torch.is_tensor(loss) and torch.isnan(loss).any():
                    logger.warning(
                        "[iter %d] NaN loss for embodiment '%s': %s",
                        iteration, embodiment_name,
                        loss.item() if loss.numel() == 1 else loss,
                    )
                if isinstance(output_batch, dict):
                    nan_keys = [
                        k for k, v in output_batch.items()
                        if torch.is_tensor(v) and torch.isnan(v).any()
                    ]
                    if nan_keys:
                        logger.warning(
                            "[iter %d] NaN in output_batch keys for embodiment '%s': %s",
                            iteration, embodiment_name, nan_keys,
                        )

            for ac_key in ac_keys:
                if ac_key in output_batch:
                    predictions[f"{embodiment_name}_{ac_key}"] = output_batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss

            # Propagate rich cosmos sub-losses (demo / WM / VF breakdowns) into
            # predictions so compute_losses can log them to TensorBoard.
            # Keys that are NaN (e.g. no WM samples in this batch) are still passed
            # through; compute_losses will filter them out before logging.
            _sub_loss_keys = (
                "edm_loss",
                "mse_loss",
                "demo_sample_action_mse_loss",
                "demo_sample_action_l1_loss",
                "demo_sample_future_image_mse_loss",
                "demo_sample_future_image_l1_loss",
                "demo_sample_future_wrist_image_mse_loss",
                "demo_sample_future_wrist_image_l1_loss",
                "demo_sample_future_proprio_mse_loss",
                "demo_sample_future_proprio_l1_loss",
                "demo_sample_value_mse_loss",
                "demo_sample_value_l1_loss",
                "world_model_sample_future_image_mse_loss",
                "world_model_sample_future_image_l1_loss",
                "world_model_sample_future_wrist_image_mse_loss",
                "world_model_sample_future_wrist_image_l1_loss",
                "world_model_sample_future_proprio_mse_loss",
                "world_model_sample_future_proprio_l1_loss",
                "world_model_sample_value_mse_loss",
                "world_model_sample_value_l1_loss",
                "value_function_sample_value_mse_loss",
                "value_function_sample_value_l1_loss",
            )
            for k in _sub_loss_keys:
                if k in output_batch:
                    predictions[f"{embodiment_name}_{k}"] = output_batch[k]

        return predictions

    @override
    def forward_eval(self, batch, return_wm_data=False):
        """
        Compute forward pass and return network outputs in @predictions dict.
        Args:
            batch (dict): processed batch from @process_batch_for_training
            return_wm_data (bool): when True also return wm_data_dict keyed by
                embodiment_id with "cosmos_batch", "generated_latent",
                "orig_clean_latent_frames" for WM visualization.
        Returns:
            predictions (dict): {ac_key: Tensor}
            wm_data_dict (dict): only returned when return_wm_data=True
        """
        if self.device is None:
            first_tensor = next(iter(batch.values())) if isinstance(batch, dict) else None
            if first_tensor is not None:
                self.device = first_tensor.device
                self._initialize_model(self.device)

        predictions = OrderedDict()
        wm_data_dict = {}

        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                cam_keys = self.camera_keys[embodiment_id]
                proprio_keys = self.proprio_keys[embodiment_id]
                lang_keys = self.lang_keys[embodiment_id]
                ac_keys = self.ac_keys[embodiment_id]
                ac_key = ac_keys[0]
                embodiment_name = get_embodiment(embodiment_id).lower()

                cosmos_batch = self._robomimic_to_cosmos_policy_data(
                    _batch, cam_keys, proprio_keys, lang_keys, ac_keys, embodiment_name
                )

                # val_num_inference_steps lets us trade quality for speed; fewer steps
                # (e.g., 5) are ~7x faster and allow full-episode validation videos.
                # return_orig_clean_latent_frames is needed for WM visualization.
                if return_wm_data:
                    generated_latent, orig_clean_latent_frames = (
                        self.model.generate_samples_from_batch(
                            cosmos_batch,
                            num_steps=self.val_num_inference_steps,
                            return_orig_clean_latent_frames=True,
                        )
                    )
                    wm_data_dict[embodiment_id] = {
                        "cosmos_batch": cosmos_batch,
                        "generated_latent": generated_latent,
                        "orig_clean_latent_frames": orig_clean_latent_frames,
                    }
                else:
                    generated_latent = self.model.generate_samples_from_batch(
                        cosmos_batch, num_steps=self.val_num_inference_steps
                    )

                action_shape = (
                    cosmos_batch["actions"].shape[1],
                    cosmos_batch["actions"].shape[2],
                )
                extracted_actions = extract_action_chunk_from_latent_sequence(
                    generated_latent,
                    action_shape=action_shape,
                    action_indices=cosmos_batch["action_latent_idx"],
                )

                pred_dict = {}
                offset = 0
                for _ac_key in ac_keys:
                    if _ac_key in _batch and "future" not in _ac_key.lower():
                        d = _batch[_ac_key].shape[-1]
                        pred_dict[_ac_key] = extracted_actions[:, :, offset:offset + d]
                        offset += d

                unnorm_preds = self.data_schematic.unnormalize_data(pred_dict, embodiment_id)
                predictions[f"{embodiment_name}_{ac_key}"] = unnorm_preds[ac_key]

        if return_wm_data:
            return predictions, wm_data_dict
        return predictions

    def forward_eval_logging(self, batch):
        """
        Called by pl_model during validation to produce metrics and visualizations.
        Args:
            batch (dict): processed batch from @process_batch_for_training
        Returns:
            metrics (dict): name -> float
            images_dict (dict): embodiment_id -> np.ndarray (B, H, W, 3)
        """
        from torchmetrics import MeanSquaredError

        preds, wm_data_dict = self.forward_eval(batch, return_wm_data=True)
        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()

        for embodiment_id, _batch in batch.items():
            _batch = self.data_schematic.unnormalize_data(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id][0]
            pred_key = f"{embodiment_name}_{ac_key}"

            if pred_key in preds:
                pred_action = preds[pred_key].cpu().contiguous()
                gt_action_full = _batch[ac_key].cpu()
                gt_action = gt_action_full[:, : pred_action.shape[1], :].contiguous()
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(pred_action, gt_action)
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    pred_action[:, -1].contiguous(), gt_action[:, -1].contiguous()
                )

            # Build WM visualization panels (predicted future vs GT future images).
            wm_preds = None
            if embodiment_id in wm_data_dict:
                try:
                    wm_preds = self._decode_wm_future_images(
                        wm_data_dict[embodiment_id], _batch,
                        self.camera_keys[embodiment_id],
                    )
                except Exception as _wm_err:
                    logger.warning("WM visualization failed: %s", _wm_err)

            ims = self.visualize_preds(preds, _batch, wm_preds=wm_preds)
            images_dict[embodiment_id] = ims

        return metrics, images_dict

    def _decode_wm_future_images(self, wm_data, batch, cam_keys):
        """
        Decode world-model predicted future images from the generated diffusion latent.

        The cosmos model outputs a full latent sequence (B, C, T_lat, H_lat, W_lat)
        where each temporal slot maps to one video segment (blank, proprio, wrist, front,
        action, future proprio, future wrist, future front, value).  We undo latent
        injection in non-image slots (proprio, action, value) to avoid VAE decode
        artifacts, then call model.decode() and extract the future-image frames.

        Args:
            wm_data (dict): {"cosmos_batch", "generated_latent", "orig_clean_latent_frames"}
            batch (dict): unnormalized robomimic batch (GT future images from here)
            cam_keys (list[str]): camera key names for this embodiment

        Returns:
            dict:
              "future_image_pred"       (B, H, W, 3) uint8 -- WM predicted future primary
              "future_image_gt"         (B, H, W, 3) uint8 -- GT future primary from batch
              "future_wrist_image_pred" (B, H, W, 3) uint8 -- WM predicted wrist (optional)
              "future_wrist_image_gt"   (B, H, W, 3) uint8 -- GT future wrist (optional)
        """
        generated_latent         = wm_data["generated_latent"]
        orig_clean_latent_frames = wm_data["orig_clean_latent_frames"]
        cosmos_batch             = wm_data["cosmos_batch"]

        def _idx(key):
            v = cosmos_batch.get(key, -1)
            if isinstance(v, torch.Tensor):
                return int(v.flatten()[0].item())
            return int(v)

        future_image_latent_idx       = _idx("future_image_latent_idx")
        future_wrist_image_latent_idx = _idx("future_wrist_image_latent_idx")
        if future_image_latent_idx == -1:
            return None  # no future image slot configured

        # Undo latent injection on non-image slots to avoid VAE decode artifacts
        # from injected proprio / action / value data.
        cleaned = generated_latent.clone()
        for key in ("current_proprio_latent_idx", "action_latent_idx",
                    "future_proprio_latent_idx", "value_latent_idx"):
            idx = _idx(key)
            if idx != -1:
                cleaned[:, :, idx] = orig_clean_latent_frames[:, :, idx]

        # Decode full latent sequence: (B, 3, T_raw, H, W) in [-1, 1]
        with torch.no_grad():
            decoded = self.model.decode(cleaned.float())
        decoded_u8 = (
            ((decoded + 1.0) * 127.5).clamp(0, 255)
            .permute(0, 2, 3, 4, 1).contiguous().to(torch.uint8).cpu().numpy()
        )  # (B, T_raw, H, W, 3) -- contiguous so PIL fromarray works

        # raw frame index: raw_idx = (latent_idx - 1) * TCF + 1
        # Cosmos temporal_compression_factor = 4.
        TCF = 4
        T_raw = decoded_u8.shape[1]
        result = {}

        pred_idx = (future_image_latent_idx - 1) * TCF + 1
        pred_idx = max(0, min(pred_idx, T_raw - 1))
        # decoded_u8[:, idx] is (B, H, W, 3) -- ensure C-contiguous for PIL
        result["future_image_pred"] = np.ascontiguousarray(decoded_u8[:, pred_idx])

        if future_wrist_image_latent_idx != -1:
            wrist_idx = (future_wrist_image_latent_idx - 1) * TCF + 1
            wrist_idx = max(0, min(wrist_idx, T_raw - 1))
            result["future_wrist_image_pred"] = np.ascontiguousarray(decoded_u8[:, wrist_idx])

        # GT future images from the unnormalized batch tensors.
        # Tensors from the robomimic batch can be (B, C, H, W) or (B, H, W, C).
        # We normalise to (B, H, W, 3) uint8 before storing.
        def _to_hwc(t):
            """Return (B, H, W, 3) uint8 numpy from a batch image tensor."""
            arr = t.cpu().float().numpy()
            if arr.ndim == 4:
                if arr.shape[1] in (1, 3, 4):  # (B, C, H, W) channels-first
                    arr = arr.transpose(0, 2, 3, 1)
                # else assume already (B, H, W, C)
            elif arr.ndim == 3:  # (B, H, W) grayscale
                arr = arr[:, :, :, np.newaxis]
            # Normalise to [0, 255] – already unnormalised but may be in [0,1] or [0,255]
            if arr.max() <= 1.01:
                arr = arr * 255.0
            arr = np.ascontiguousarray(arr.clip(0, 255).astype(np.uint8))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            return arr

        for cam_key in cam_keys:
            if cam_key not in batch:
                continue
            gt_img = _to_hwc(batch[cam_key])
            if "front" in cam_key.lower() and "future" in cam_key.lower():
                result["future_image_gt"] = gt_img
            elif "wrist" in cam_key.lower() and "future" in cam_key.lower():
                if "future_wrist_image_gt" not in result:
                    result["future_wrist_image_gt"] = gt_img

        return result

    @override
    def visualize_preds(self, predictions, batch, wm_preds=None):
        """
        Visualize action predictions overlaid on the current observation.

        When wm_preds is provided (from _decode_wm_future_images), each output frame
        is a 3-panel horizontal strip:
            [ current obs + action trajectory | GT future image | WM predicted future ]
        If wrist images are available a second row is stacked beneath:
            [ current wrist | GT future wrist | WM predicted future wrist ]

        Args:
            predictions (dict): {ac_key: Tensor (B, Seq, D)}
            batch (dict): unnormalized batch with image and action tensors
            wm_preds (dict | None): output from _decode_wm_future_images (optional)
        Returns:
            ims (np.ndarray): (B, H, W*3, 3) or (B, H*2, W*3, 3)
        """
        from egomimic.utils.egomimicUtils import draw_actions

        embodiment_id = batch.get("embodiment", [0])[0].item() if "embodiment" in batch else 0
        embodiment_name = get_embodiment(embodiment_id).lower()
        ac_key = self.ac_keys[embodiment_id][0]

        viz_img_key = self.data_schematic.viz_img_key()[embodiment_id]
        ims = (batch[viz_img_key].cpu().numpy().transpose((0, 2, 3, 1)) * 255).astype(np.uint8)

        pred_key = f"{embodiment_name}_{ac_key}"
        if pred_key in predictions:
            preds = predictions[pred_key]
            gt_full = batch[ac_key]
            gt = gt_full[:, : preds.shape[1], :]

            gt, gt_rot = self._extract_xyz(gt)
            preds, preds_rot = self._extract_xyz(preds)

            for b in range(ims.shape[0]):
                if preds.shape[-1] in (7, 14):
                    ac_type = "joints"
                elif preds.shape[-1] in (3, 6):
                    ac_type = "xyz"
                else:
                    ac_type = "joints"

                arm = "right" if preds.shape[-1] in (7, 3) else "both"
                ims[b] = draw_actions(
                    ims[b], ac_type, "Purples", preds[b].cpu().numpy(),
                    self.camera_transforms.extrinsics, self.camera_transforms.intrinsics, arm=arm,
                )
                ims[b] = draw_actions(
                    ims[b], ac_type, "Greens", gt[b].cpu().numpy(),
                    self.camera_transforms.extrinsics, self.camera_transforms.intrinsics, arm=arm,
                )

        # ---- World-model comparison panel ------------------------------------
        # Append GT-future and WM-predicted-future columns so each output frame
        # shows [current+actions | GT future | WM pred] as a 3-panel strip.
        if wm_preds is not None:
            from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw

            B_vis = ims.shape[0]
            H_vis, W_vis = ims.shape[1], ims.shape[2]

            def _labeled(img_np, text, color=(255, 255, 255)):
                arr = np.ascontiguousarray(img_np).astype(np.uint8)
                pil = _PIL_Image.fromarray(arr)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                draw = _PIL_Draw.Draw(pil)
                draw.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
                draw.text((3, 2), text, fill=color)
                return np.array(pil)

            def _resize_to(img_np, h, w):
                from PIL import Image as _I
                arr = np.ascontiguousarray(img_np).astype(np.uint8)
                pil = _I.fromarray(arr)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                return np.array(pil.resize((w, h)))

            gt_imgs = wm_preds.get("future_image_gt",
                                   np.zeros((B_vis, H_vis, W_vis, 3), dtype=np.uint8))
            wm_imgs = wm_preds.get("future_image_pred",
                                   np.zeros((B_vis, H_vis, W_vis, 3), dtype=np.uint8))

            panels_gt   = np.stack([_resize_to(gt_imgs[b],  H_vis, W_vis) for b in range(B_vis)])
            panels_wm   = np.stack([_resize_to(wm_imgs[b],  H_vis, W_vis) for b in range(B_vis)])
            ims_labeled = np.stack([_labeled(ims[b], "Current + actions", (180, 180, 255))
                                    for b in range(B_vis)])
            panels_gt   = np.stack([_labeled(panels_gt[b],  "GT future",  (180, 255, 180))
                                    for b in range(B_vis)])
            panels_wm   = np.stack([_labeled(panels_wm[b],  "WM pred",    (255, 180, 180))
                                    for b in range(B_vis)])

            primary_row = np.concatenate([ims_labeled, panels_gt, panels_wm], axis=2)

            if "future_wrist_image_gt" in wm_preds and "future_wrist_image_pred" in wm_preds:
                wrist_cur = None
                for _wk in self.camera_keys.get(embodiment_id, []):
                    if "wrist" in _wk.lower() and "future" not in _wk.lower() and _wk in batch:
                        wrist_cur = (
                            batch[_wk].cpu().numpy().transpose(0, 2, 3, 1) * 255
                        ).astype(np.uint8)
                        break
                if wrist_cur is None:
                    wrist_cur = np.zeros((B_vis, H_vis, W_vis, 3), dtype=np.uint8)
                gt_wrist = wm_preds["future_wrist_image_gt"]
                wm_wrist = wm_preds["future_wrist_image_pred"]
                wrist_cur = np.stack([_resize_to(wrist_cur[b], H_vis, W_vis) for b in range(B_vis)])
                gt_wrist  = np.stack([_resize_to(gt_wrist[b],  H_vis, W_vis) for b in range(B_vis)])
                wm_wrist  = np.stack([_resize_to(wm_wrist[b],  H_vis, W_vis) for b in range(B_vis)])
                wrist_cur = np.stack([_labeled(wrist_cur[b], "Wrist current",   (180, 180, 255)) for b in range(B_vis)])
                gt_wrist  = np.stack([_labeled(gt_wrist[b],  "Wrist GT future", (180, 255, 180)) for b in range(B_vis)])
                wm_wrist  = np.stack([_labeled(wm_wrist[b],  "Wrist WM pred",   (255, 180, 180)) for b in range(B_vis)])
                wrist_row = np.concatenate([wrist_cur, gt_wrist, wm_wrist], axis=2)
                ims = np.concatenate([primary_row, wrist_row], axis=1)  # (B, 2H, 3W, 3)
            else:
                ims = primary_row  # (B, H, 3W, 3)

        return ims

    @override
    def compute_losses(self, predictions, batch):
        """
        Compute losses from predictions dict.

        In addition to the main EDM action_loss, also extracts and logs the rich
        per-category breakdown losses propagated by forward_training:
          - demo_sample_*   (policy BC losses)
          - world_model_sample_*  (WM future-state losses)
          - value_function_sample_*  (VF value losses)
        NaN values (e.g. when no WM/VF samples appeared in the batch) are silently
        skipped to keep TensorBoard clean.

        Args:
            predictions (dict): output of forward_training
            batch (dict): processed batch (used for embodiment iteration)
        Returns:
            losses (dict): losses to log
        """
        loss_dict = OrderedDict()
        total_action_loss = None

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            loss_key = f"{embodiment_name}_loss"

            if loss_key in predictions:
                bc_loss = predictions[loss_key]
                if total_action_loss is None:
                    total_action_loss = torch.tensor(0.0, device=bc_loss.device)
                total_action_loss += bc_loss
                loss_dict[loss_key] = bc_loss

            # Forward cosmos sub-losses (NaN-filtered) so they appear in TensorBoard.
            # Keys follow the pattern "{embodiment_name}_{cosmos_loss_key}".
            _cosmos_prefix = f"{embodiment_name}_"
            _sub_loss_keys = (
                "edm_loss",
                "mse_loss",
                "demo_sample_action_mse_loss",
                "demo_sample_action_l1_loss",
                "demo_sample_future_image_mse_loss",
                "demo_sample_future_image_l1_loss",
                "demo_sample_future_wrist_image_mse_loss",
                "demo_sample_future_wrist_image_l1_loss",
                "demo_sample_future_proprio_mse_loss",
                "demo_sample_future_proprio_l1_loss",
                "demo_sample_value_mse_loss",
                "demo_sample_value_l1_loss",
                "world_model_sample_future_image_mse_loss",
                "world_model_sample_future_image_l1_loss",
                "world_model_sample_future_wrist_image_mse_loss",
                "world_model_sample_future_wrist_image_l1_loss",
                "world_model_sample_future_proprio_mse_loss",
                "world_model_sample_future_proprio_l1_loss",
                "world_model_sample_value_mse_loss",
                "world_model_sample_value_l1_loss",
                "value_function_sample_value_mse_loss",
                "value_function_sample_value_l1_loss",
            )
            for sub_key in _sub_loss_keys:
                pred_key = _cosmos_prefix + sub_key
                if pred_key in predictions:
                    v = predictions[pred_key]
                    if torch.is_tensor(v) and not torch.isnan(v).any():
                        loss_dict[pred_key] = v

        if total_action_loss is not None:
            loss_dict["action_loss"] = total_action_loss / len(self.domains)
        else:
            loss_dict["action_loss"] = torch.tensor(0.0)

        return loss_dict

    @override
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize for TensorBoard.
        Organizes losses into three groups:
          Train/Loss            -- main EDM action loss
          Train/Policy/*        -- demo (policy BC) sub-losses
          Train/WorldModel/*    -- world-model sub-losses
          Train/ValueFunction/* -- value-function sub-losses
        Args:
            info (dict): dictionary containing "losses"
        Returns:
            loss_log (dict): name -> float
        """
        log = OrderedDict()

        if "losses" not in info:
            return log

        losses = info["losses"]

        if "action_loss" in losses:
            log["Loss"] = losses["action_loss"].item()

        for loss_key, loss_val in losses.items():
            if not torch.is_tensor(loss_val):
                continue
            v = loss_val.item()
            # Route to a TensorBoard sub-group for readability.
            if "demo_sample_" in loss_key:
                # strip embodiment prefix (everything before first "_demo_")
                short = loss_key[loss_key.index("demo_sample_"):]
                log[f"Policy/{short}"] = v
            elif "world_model_sample_" in loss_key:
                short = loss_key[loss_key.index("world_model_sample_"):]
                log[f"WorldModel/{short}"] = v
            elif "value_function_sample_" in loss_key:
                short = loss_key[loss_key.index("value_function_sample_"):]
                log[f"ValueFunction/{short}"] = v
            elif loss_key.endswith("_edm_loss"):
                log["edm_loss"] = v
            elif loss_key.endswith("_mse_loss") and "_sample_" not in loss_key:
                log["mse_loss"] = v
            else:
                log[loss_key] = v

        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]

        return log

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

    
