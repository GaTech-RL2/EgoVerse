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
from cosmos_policy._src.predict2.conditioner import TextAttr
from cosmos_policy._src.predict2.models.text2world_model import EMAConfig
from cosmos_policy._src.predict2.networks.wan2pt1 import WanModel
from cosmos_policy._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy.datasets.dataset_utils import preprocess_image, resize_images

logger = logging.getLogger(__name__)


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
        - conditioner: dict -> L(Video2WorldConditioner)(...) LazyCall (with nested text -> L(TextAttr)(...))
        - ema: dict -> EMAConfig object
        - net: dict -> L(WanModel)(...) LazyCall (with nested sac_config -> L(SACConfig)(...))
        
        Returns:
            CosmosPolicyVideo2WorldConfig: Properly configured config object with all nested
                configs converted to the correct types
        """
        model_config_dict = OmegaConf.to_container(self.model_config, resolve=True)
        
        # Convert SDE config dict to LazyCall (required by config class)
        if "sde" in model_config_dict:
            sde_config = model_config_dict["sde"]
            model_config_dict["sde"] = L(HybridEDMSDE)(**sde_config)
        
        # Convert tokenizer config dict to LazyCall (required by config class - expects LazyDict)
        if "tokenizer" in model_config_dict:
            tokenizer_config = model_config_dict["tokenizer"]
            model_config_dict["tokenizer"] = L(Wan2pt1VAEInterface)(**tokenizer_config)
        
        # Convert conditioner config dict to LazyCall (handle nested text config)
        if "conditioner" in model_config_dict:
            conditioner_config = model_config_dict["conditioner"]
            if isinstance(conditioner_config, dict):
                processed_conditioner = {}
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
                model_config_dict["net"] = L(WanModel)(**processed_net)
        
        # Filter out None values to use config class defaults
        filtered_config = {k: v for k, v in model_config_dict.items() if v is not None}
        cosmos_config = CosmosPolicyVideo2WorldConfig(**filtered_config)
        
        return cosmos_config
    
    
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
        
        # Concatenation priprio and actions
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
        
        # Value function returns (placeholders)
        value_function_return = float("-100")  # Just a placeholder
        next_value_function_return = float("-100")
        all_value_returns = torch.full((B,), value_function_return, dtype=torch.float32, device=device)
        all_next_value_returns = torch.full((B,), next_value_function_return, dtype=torch.float32, device=device)
        
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
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """
        if self.device is None:
            # Set device from batch
            first_tensor = next(iter(batch.values())) if isinstance(batch, dict) else None
            if first_tensor is not None:
                self.device = first_tensor.device
                self._initialize_model(self.device)
        
        predictions = OrderedDict()
        
        # batch: dict_keys(['front_img_1', 'left_wrist_img', 'right_wrist_img', 
        # 'joint_positions', 'ee_pose', 'actions_joints', 'actions_cartesian', 'embodiment', 'pad_mask'])
        for embodiment_id, _batch in batch.items():
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_keys = self.ac_keys[embodiment_id]
            embodiment_name = get_embodiment(embodiment_id).lower()
            
            # Transform to cosmos_policy format
            cosmos_batch = self._robomimic_to_cosmos_policy_data(
                _batch, cam_keys, proprio_keys, lang_keys, ac_keys, embodiment_name
            )
            
            # Call cosmos_policy model training_step
            # Get current iteration (would need to track this)
            iteration = getattr(self, "_current_iteration", 0)
            output_batch, loss = self.model.training_step(cosmos_batch, iteration)
            
            for ac_key in ac_keys:
                if ac_key in output_batch:
                    predictions[f"{embodiment_name}_{ac_key}"] = output_batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss
        
        return predictions
    
    @override
    def forward_eval(self, batch):
        """
        Compute forward pass and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
        """
        if self.device is None:
            first_tensor = next(iter(batch.values())) if isinstance(batch, dict) else None
            if first_tensor is not None:
                self.device = first_tensor.device
                self._initialize_model(self.device)
        
        predictions = OrderedDict()
        
        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                cam_keys = self.camera_keys[embodiment_id]
                proprio_keys = self.proprio_keys[embodiment_id]
                lang_keys = self.lang_keys[embodiment_id]
                ac_key = self.ac_keys[embodiment_id]
                embodiment_name = get_embodiment(embodiment_id).lower()
                
                # Transform to cosmos_policy format
                cosmos_batch = self._robomimic_to_cosmos_policy_data(
                    _batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment_name
                )
                
                # Call cosmos_policy model inference
                if self.model is None:
                    logger.warning("CosmosPolicy model not initialized - using placeholder prediction")
                    # Return original actions as placeholder
                    pred_actions = _batch[ac_key]
                else:
                    # Use model's generate_samples_from_batch or similar method
                    # This is a placeholder - actual implementation needs proper inference
                    pred_actions = _batch[ac_key]  # Placeholder
                
                # Unnormalize predictions
                pred_dict = {ac_key: pred_actions}
                unnorm_preds = self.data_schematic.unnormalize_data(pred_dict, embodiment_id)
                
                for key in unnorm_preds:
                    predictions[f"{embodiment_name}_{key}"] = unnorm_preds[key]
        
        return predictions
    
    @override
    def forward_eval_logging(self, batch):
        """
        Called by pl_model to generate a dictionary of metrics and an image visualization
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            metrics (dict): metricname: value (float)
            images_dict (dict): embodiment_id: images (np.ndarray)
        """
        from torchmetrics import MeanSquaredError
        
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
                    preds[pred_key].cpu(),
                    _batch[ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(),
                    _batch[ac_key][:, -1].cpu()
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
            batch (dict): {ac_key: torch.Tensor (B, Seq, D), front_img_1: torch.Tensor (B, 3, H, W)}
        Returns:
            ims (np.ndarray): (B, H, W, 3) - images with actions drawn on top
        """
        from egomimic.utils.egomimicUtils import draw_actions
        
        # Get embodiment from batch
        embodiment_id = batch.get("embodiment", [0])[0].item() if "embodiment" in batch else 0
        embodiment_name = get_embodiment(embodiment_id).lower()
        ac_key = self.ac_keys[embodiment_id]
        
        # Get visualization image key
        viz_img_key = self.data_schematic.viz_img_key()[embodiment_id]
        ims = (batch[viz_img_key].cpu().numpy().transpose((0, 2, 3, 1)) * 255).astype(np.uint8)
        
        # Draw predictions and ground truth
        pred_key = f"{embodiment_name}_{ac_key}"
        if pred_key in predictions:
            preds = predictions[pred_key]
            gt = batch[ac_key]
            
            for b in range(ims.shape[0]):
                if preds.shape[-1] == 7 or preds.shape[-1] == 14:
                    ac_type = "joints"
                elif preds.shape[-1] == 3 or preds.shape[-1] == 6:
                    ac_type = "xyz"
                else:
                    ac_type = "joints"  # Default
                
                arm = "right" if preds.shape[-1] == 7 or preds.shape[-1] == 3 else "both"
                ims[b] = draw_actions(
                    ims[b], ac_type, "Purples", preds[b].cpu().numpy(),
                    self.camera_transforms.extrinsics, self.camera_transforms.intrinsics, arm=arm
                )
                ims[b] = draw_actions(
                    ims[b], ac_type, "Greens", gt[b].cpu().numpy(),
                    self.camera_transforms.extrinsics, self.camera_transforms.intrinsics, arm=arm
                )
        
        return ims
    
    @override
    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
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
        
        if total_action_loss is not None:
            loss_dict["action_loss"] = total_action_loss / len(self.domains)
        else:
            loss_dict["action_loss"] = torch.tensor(0.0)
        
        return loss_dict
    
    @override
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()
        
        if "losses" in info:
            if "action_loss" in info["losses"]:
                log["Loss"] = info["losses"]["action_loss"].item()
            
            for loss_key, loss in info["losses"].items():
                log[loss_key] = loss.item()
        
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        
        return log

