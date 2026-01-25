"""
DWM Policy Rollout for EgoVerse.

This module provides a rollout class for policies trained with the 
dualistic-world-model (DWM) framework, making them compatible with
EgoVerse's robot interface.

Supports:
- eepose_6drot (20D): xyz(3) + rot6d(6) + grip(1) per arm
- eepose_ypr (14D): xyz(3) + ypr(3) + grip(1) per arm

Streaming Architecture:
- DWMInferenceServer: Runs on desktop GPU, receives observations, sends actions
- DWMStreamingClient: Runs on Jetson, sends observations, receives actions
"""

import os
import sys
import time
import numpy as np
import torch
import threading
import queue
from typing import Dict, Optional, Tuple, Any

# Add dualistic-world-model to path for model loading
DWM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../dualistic-world-model"))
if DWM_PATH not in sys.path:
    sys.path.insert(0, DWM_PATH)

from egomimic.utils.egomimicUtils import CameraTransforms, cam_frame_to_base_frame

# Optional streaming dependencies
try:
    import zmq
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()  # Enable numpy array serialization
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


# =============================================================================
# S3 Utilities
# =============================================================================

def download_s3_checkpoint(s3_path: str) -> str:
    """Download checkpoint from S3 if needed.
    
    Args:
        s3_path: Path to checkpoint (local or s3://)
        
    Returns:
        Local path to checkpoint
    """
    if not s3_path.startswith('s3://'):
        return s3_path  # Already local
    
    # Import from DWM utils
    from utils.data_utils import download_s3_file
    print(f"Downloading checkpoint from S3: {s3_path}")
    local_path = download_s3_file(s3_path)
    print(f"Downloaded to: {local_path}")
    return local_path


# =============================================================================
# Action Conversion Utilities
# =============================================================================

def reconstruct_rot_from_6d(rot6d: torch.Tensor) -> torch.Tensor:
    """Reconstruct rotation matrix from 6D representation.
    
    The 6D vector is stored as a row-major flatten of the first two
    columns of the rotation matrix (as saved by collect_demo.py):
    [R00, R01, R10, R11, R20, R21].
    
    Args:
        rot6d: (B, T, 6) tensor containing flattened first two columns of rotation matrix
        
    Returns:
        R: (B, T, 3, 3) rotation matrix
    """
    B, T, _ = rot6d.shape
    rot_cols = rot6d.reshape(B, T, 3, 2)
    c1 = rot_cols[..., :, 0]  # (B, T, 3)
    c2 = rot_cols[..., :, 1]  # (B, T, 3)
    
    eps = 1e-8
    # Normalize c1
    c1n = c1 / (c1.norm(dim=-1, keepdim=True).clamp_min(eps))
    # Gram-Schmidt orthogonalization for c2
    proj = (c2 * c1n).sum(dim=-1, keepdim=True) * c1n
    c2o = c2 - proj
    c2n = c2o / (c2o.norm(dim=-1, keepdim=True).clamp_min(eps))
    # Third column via cross product
    c3n = torch.cross(c1n, c2n, dim=-1)
    # Stack as columns
    R = torch.stack([c1n, c2n, c3n], dim=-1)  # (B, T, 3, 3)
    return R


def rot_matrix_to_ypr(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to yaw-pitch-roll (ZYX convention).
    
    Args:
        R: (B, T, 3, 3) rotation matrix
        
    Returns:
        ypr: (B, T, 3) yaw, pitch, roll in radians
    """
    # Extract sin(pitch) from R[2,0]
    sy = -R[..., 2, 0]
    sy = sy.clamp(-1.0, 1.0)
    pitch = torch.asin(sy)
    
    # Compute yaw and roll
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    
    return torch.stack([yaw, pitch, roll], dim=-1)


def convert_rot6d_to_ypr(actions_rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 20D rot6d actions to 14D ypr actions.
    
    Args:
        actions_rot6d: (B, T, 20) tensor with format:
            Left arm:  xyz[0:3], rot6d[3:9], grip[9]
            Right arm: xyz[10:13], rot6d[13:19], grip[19]
            
    Returns:
        actions_ypr: (B, T, 14) tensor with format:
            Left arm:  xyz[0:3], ypr[3:6], grip[6]
            Right arm: xyz[7:10], ypr[10:13], grip[13]
    """
    B, T, D = actions_rot6d.shape
    if D != 20:
        raise ValueError(f"Expected 20D rot6d actions, got {D}D")
    
    # Extract left arm components
    left_xyz = actions_rot6d[..., 0:3]      # (B, T, 3)
    left_rot6d = actions_rot6d[..., 3:9]    # (B, T, 6)
    left_grip = actions_rot6d[..., 9:10]    # (B, T, 1)
    
    # Extract right arm components
    right_xyz = actions_rot6d[..., 10:13]   # (B, T, 3)
    right_rot6d = actions_rot6d[..., 13:19] # (B, T, 6)
    right_grip = actions_rot6d[..., 19:20]  # (B, T, 1)
    
    # Convert rot6d to ypr
    left_R = reconstruct_rot_from_6d(left_rot6d)    # (B, T, 3, 3)
    right_R = reconstruct_rot_from_6d(right_rot6d)  # (B, T, 3, 3)
    
    left_ypr = rot_matrix_to_ypr(left_R)   # (B, T, 3)
    right_ypr = rot_matrix_to_ypr(right_R) # (B, T, 3)
    
    # Concatenate into 14D format
    actions_ypr = torch.cat([
        left_xyz, left_ypr, left_grip,
        right_xyz, right_ypr, right_grip
    ], dim=-1)
    
    return actions_ypr


# =============================================================================
# Noise Scheduler Utilities
# =============================================================================

def extract_scheduler_config_from_checkpoint(checkpoint_path: str) -> Dict:
    """Extract noise scheduler config from DWM checkpoint.
    
    Args:
        checkpoint_path: Path to DWM checkpoint file (local or s3://)
        
    Returns:
        scheduler_config: Dictionary with scheduler parameters
    """
    # Handle S3 paths
    local_path = download_s3_checkpoint(checkpoint_path)
    
    checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint missing 'config' key: {checkpoint_path}")
    
    model_cfg = checkpoint['config']
    
    # Import losses module from DWM
    from losses import ModalityLossManager, DiffusionLoss
    
    objectives_cfg = model_cfg["training"]["objectives"]
    loss_mgr = ModalityLossManager(objectives_cfg)
    diff_loss = next(
        (loss for loss in loss_mgr.losses.values() if isinstance(loss, DiffusionLoss)),
        None
    )
    
    if diff_loss is None:
        raise ValueError("No DiffusionLoss found in checkpoint's objectives config")
    
    return dict(diff_loss.noise_scheduler.config)


def create_noise_scheduler(scheduler_config: Dict, decoder_name: str):
    """Create noise scheduler from config.
    
    Args:
        scheduler_config: Dictionary with scheduler parameters
        decoder_name: Name of the decoder (used to determine scheduler type)
        
    Returns:
        noise_scheduler: Diffusion scheduler instance
    """
    from diffusers import DDIMScheduler, CogVideoXDDIMScheduler
    
    if 'cogvideox' in decoder_name.lower():
        scheduler = CogVideoXDDIMScheduler(
            num_train_timesteps=scheduler_config['num_train_timesteps'],
            beta_schedule=scheduler_config['beta_schedule'],
            beta_start=scheduler_config['beta_start'],
            beta_end=scheduler_config['beta_end'],
            clip_sample=scheduler_config['clip_sample'],
            set_alpha_to_one=scheduler_config['set_alpha_to_one'],
            steps_offset=scheduler_config['steps_offset'],
            prediction_type=scheduler_config['prediction_type'],
            clip_sample_range=scheduler_config['clip_sample_range'],
            sample_max_value=scheduler_config['sample_max_value'],
            timestep_spacing=scheduler_config['timestep_spacing'],
            rescale_betas_zero_snr=scheduler_config['rescale_betas_zero_snr'],
            snr_shift_scale=scheduler_config['snr_shift_scale'],
        )
    else:
        scheduler = DDIMScheduler(
            num_train_timesteps=scheduler_config['num_train_timesteps'],
            beta_start=scheduler_config['beta_start'],
            beta_end=scheduler_config['beta_end'],
            beta_schedule=scheduler_config['beta_schedule'],
            clip_sample=scheduler_config['clip_sample'],
            set_alpha_to_one=scheduler_config['set_alpha_to_one'],
            steps_offset=scheduler_config['steps_offset'],
            prediction_type=scheduler_config['prediction_type'],
        )
    
    return scheduler


# =============================================================================
# DWM Policy Rollout
# =============================================================================

class DWMPolicyRollout:
    """Rollout class for DWM-trained policies compatible with EgoVerse robots.
    
    This class loads a DWM checkpoint, runs inference, and converts actions
    to the 7D-per-arm format expected by the robot interface.
    """
    
    def __init__(
        self,
        arm: str,
        checkpoint_path: str,
        query_frequency: int,
        extrinsics_key: str,
        num_inference_steps: int,
        task_description: str,
        img_size: int = 128,
    ):
        """Initialize DWM policy rollout.
        
        Args:
            arm: Which arm(s) to control ("left", "right", or "both")
            checkpoint_path: Path to DWM checkpoint
            query_frequency: How often to run inference (in timesteps)
            extrinsics_key: Key for camera extrinsics transforms
            num_inference_steps: Number of diffusion denoising steps
            task_description: Text description of the task
            img_size: Image size expected by model
        """
        self.arm = arm
        self.checkpoint_path = checkpoint_path
        self.query_frequency = query_frequency
        self.num_inference_steps = num_inference_steps
        self.task_description = task_description
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Camera transforms
        self.extrinsics = CameraTransforms(
            intrinsics_key="base", 
            extrinsics_key=extrinsics_key
        ).extrinsics
        
        # Load checkpoint and extract config
        print(f"Loading DWM checkpoint: {checkpoint_path}")
        self._load_model(checkpoint_path)
        
        # Action buffer for temporal execution
        self.actions = None
        self.debug_actions = None
        
    def _load_model(self, checkpoint_path: str):
        """Load model and extract configuration."""
        from models.model_loading import load_unified_model_from_checkpoint
        from models import register_components
        
        # Register DWM components
        register_components()
        
        # Handle S3 paths
        local_path = download_s3_checkpoint(checkpoint_path)
        
        # Load checkpoint to extract config
        checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)
        self.model_cfg = checkpoint['config']
        self._local_checkpoint_path = local_path  # Store for model loading
        
        # Extract action space from config
        dataset_cfg = self.model_cfg.get('training', {}).get('dataset', {})
        self.action_space = dataset_cfg.get('action_space', 'eepose_ypr')
        self.action_dim = 20 if self.action_space == 'eepose_6drot' else 14
        
        print(f"  Action space: {self.action_space} ({self.action_dim}D)")
        
        # Extract camera views from config
        self.cond_cameraviews = dataset_cfg.get('cond_cameraviews', ['front_img_1'])
        print(f"  Camera views: {self.cond_cameraviews}")
        
        # Load the model wrapper (use local path for S3 checkpoints)
        self.model = load_unified_model_from_checkpoint(local_path, self.device)
        self.model.eval()
        
        # Extract scheduler config and create scheduler (already handles S3)
        self.scheduler_config = extract_scheduler_config_from_checkpoint(local_path)
        self.noise_scheduler = create_noise_scheduler(
            self.scheduler_config, 
            self.model.action_decoder_name
        )
        print(f"  Scheduler: {type(self.noise_scheduler).__name__}")
        print(f"  Num inference steps: {self.num_inference_steps}")
        
    def _process_obs(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """Process observations for DWM model input.
        
        Args:
            obs: Raw observations dict with keys like 'front_img_1', 'joint_positions'
            
        Returns:
            Processed batch dict for model
        """
        # Process images - convert from BGR (uint8) to RGB (float, normalized)
        images = []
        for view in self.cond_cameraviews:
            if view not in obs:
                raise KeyError(f"Missing camera view '{view}' in observations. Available: {list(obs.keys())}")
            
            img = obs[view]
            if img is None:
                raise ValueError(f"Camera view '{view}' is None")
            
            # Convert BGR to RGB, normalize to [0, 1]
            img_rgb = img[..., ::-1].copy()  # BGR to RGB
            img_tensor = torch.from_numpy(img_rgb).float() / 255.0  # (H, W, C)
            img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
            
            # Resize if needed
            if img_tensor.shape[1] != self.img_size or img_tensor.shape[2] != self.img_size:
                import torch.nn.functional as F
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0), 
                    size=(self.img_size, self.img_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            images.append(img_tensor)
        
        # Stack views: (V, C, H, W) -> (1, V, 1, C, H, W) for video format
        images = torch.stack(images, dim=0)  # (V, C, H, W)
        images = images.unsqueeze(0).unsqueeze(2)  # (1, V, 1, C, H, W)
        images = images.to(self.device)
        
        # Process proprioception (joint positions)
        if 'joint_positions' in obs:
            proprio = torch.from_numpy(obs['joint_positions']).float()
            proprio = proprio.unsqueeze(0).to(self.device)  # (1, 14)
        else:
            # Create zero proprioception if not available
            proprio = torch.zeros(1, 14, device=self.device)
        
        return {
            'video': images,
            'text': [self.task_description],
            'proprioception': proprio,
        }
    
    def _run_inference(self, processed_obs: Dict) -> torch.Tensor:
        """Run DWM inference.
        
        Args:
            processed_obs: Processed observations dict
            
        Returns:
            actions: (1, T, action_dim) tensor of predicted actions
        """
        with torch.no_grad():
            # Get conditioning from trunk
            cond_dict = self.model.get_conditioning_dict(
                video=processed_obs['video'],
                text=processed_obs['text'],
                proprioception=processed_obs['proprioception'],
            )
            
            # Run diffusion decoding
            actions = self.model.forward(
                cond_dict,
                self.noise_scheduler,
                num_inference_steps=self.num_inference_steps,
            )
        
        return actions
    
    def _convert_actions(self, actions: torch.Tensor) -> np.ndarray:
        """Convert model output actions to robot format.
        
        Args:
            actions: (1, T, action_dim) tensor from model
            
        Returns:
            actions_np: (T, 14) numpy array with ypr format
        """
        # Convert rot6d to ypr if needed
        if self.action_space == 'eepose_6drot':
            actions = convert_rot6d_to_ypr(actions)  # (1, T, 14)
        
        # Remove batch dimension
        actions_np = actions.squeeze(0).cpu().numpy()  # (T, 14)
        
        return actions_np
    
    def _transform_to_base_frame(self, actions: np.ndarray) -> np.ndarray:
        """Transform actions from camera frame to robot base frame.
        
        Args:
            actions: (T, 14) array with left and right arm actions
            
        Returns:
            transformed: (T, 14) array in base frame
        """
        if self.arm == "both":
            left_actions = actions[:, :7]
            right_actions = actions[:, 7:14]
            
            # Transform position + orientation, preserve gripper
            left_6dof = cam_frame_to_base_frame(left_actions[:, :6].copy(), self.extrinsics["left"])
            right_6dof = cam_frame_to_base_frame(right_actions[:, :6].copy(), self.extrinsics["right"])
            
            left_out = np.hstack([left_6dof, left_actions[:, 6:7]])
            right_out = np.hstack([right_6dof, right_actions[:, 6:7]])
            
            return np.hstack([left_out, right_out])
        else:
            arm_offset = 7 if self.arm == "right" else 0
            arm_actions = actions[:, arm_offset:arm_offset + 7]
            
            transformed_6dof = cam_frame_to_base_frame(
                arm_actions[:, :6].copy(), 
                self.extrinsics[self.arm]
            )
            
            return np.hstack([transformed_6dof, arm_actions[:, 6:7]])
    
    def rollout_step(self, i: int, obs: Dict) -> Optional[np.ndarray]:
        """Execute one rollout step.
        
        Args:
            i: Current timestep
            obs: Current observations
            
        Returns:
            action: (7,) or (14,) array for single or bimanual control
        """
        # Run inference at query frequency intervals
        if i % self.query_frequency == 0:
            start_t = time.time()
            
            # Process observations
            processed_obs = self._process_obs(obs)
            
            # Run model inference
            actions_tensor = self._run_inference(processed_obs)
            
            # Convert and transform actions
            actions_ypr = self._convert_actions(actions_tensor)
            self.actions = self._transform_to_base_frame(actions_ypr)
            self.debug_actions = self.actions.copy()
            
            print(f"Inference time: {time.time() - start_t:.3f}s, actions shape: {self.actions.shape}")
        
        if self.actions is None:
            return None
        
        # Get action at current index within query window
        act_idx = i % self.query_frequency
        if act_idx >= self.actions.shape[0]:
            act_idx = self.actions.shape[0] - 1
        
        action = self.actions[act_idx]
        
        # Return appropriate action based on arm configuration
        if self.arm == "both":
            return action  # (14,)
        elif self.arm == "right":
            return action if action.shape[0] == 7 else action[7:14]  # (7,)
        else:  # left
            return action if action.shape[0] == 7 else action[:7]  # (7,)
    
    def reset(self):
        """Reset rollout state."""
        self.actions = None
        self.debug_actions = None
        print("DWM policy reset")


# =============================================================================
# Streaming Architecture for Desktop-Jetson Communication
# =============================================================================

class DWMInferenceServer:
    """Inference server that runs on desktop GPU.
    
    Receives observations from Jetson, runs DWM inference, sends back actions.
    Uses ZeroMQ for low-latency communication.
    
    Protocol:
    - REQ-REP pattern for synchronous request-response
    - Jetson sends: {"cmd": "infer", "obs": {...}, "step": int}
    - Server replies: {"actions": ndarray, "step": int}
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        task_description: str,
        num_inference_steps: int,
        bind_address: str = "tcp://*:5555",
        img_size: int = 128,
    ):
        """Initialize inference server.
        
        Args:
            checkpoint_path: Path to DWM checkpoint (local or s3://)
            task_description: Text description of the task
            num_inference_steps: Number of diffusion denoising steps
            bind_address: ZMQ bind address (e.g., "tcp://*:5555")
            img_size: Image size expected by model
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("zmq and msgpack-numpy required: pip install pyzmq msgpack msgpack-numpy")
        
        self.task_description = task_description
        self.num_inference_steps = num_inference_steps
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bind_address = bind_address
        self.running = False
        
        # Load model
        print(f"Loading DWM model for inference server...")
        self._load_model(checkpoint_path)
        
        # Setup ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(bind_address)
        print(f"DWM Inference Server bound to {bind_address}")
        
    def _load_model(self, checkpoint_path: str):
        """Load model for inference."""
        from models.model_loading import load_unified_model_from_checkpoint
        from models import register_components
        
        register_components()
        
        # Handle S3 paths
        local_path = download_s3_checkpoint(checkpoint_path)
        
        # Load config
        checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)
        self.model_cfg = checkpoint['config']
        
        # Extract action space
        dataset_cfg = self.model_cfg.get('training', {}).get('dataset', {})
        self.action_space = dataset_cfg.get('action_space', 'eepose_ypr')
        self.cond_cameraviews = dataset_cfg.get('cond_cameraviews', ['front_img_1'])
        
        print(f"  Action space: {self.action_space}")
        print(f"  Camera views: {self.cond_cameraviews}")
        
        # Load model
        self.model = load_unified_model_from_checkpoint(local_path, self.device)
        self.model.eval()
        
        # Create scheduler
        self.scheduler_config = extract_scheduler_config_from_checkpoint(local_path)
        self.noise_scheduler = create_noise_scheduler(
            self.scheduler_config,
            self.model.action_decoder_name
        )
        
    def _process_obs(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """Process observations for model input."""
        images = []
        for view in self.cond_cameraviews:
            if view not in obs:
                raise KeyError(f"Missing camera view '{view}'")
            
            img = obs[view]
            if isinstance(img, np.ndarray):
                # Assume BGR uint8 from camera
                img_rgb = img[..., ::-1].copy()
                img_tensor = torch.from_numpy(img_rgb).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)
            else:
                img_tensor = img
            
            # Resize if needed
            if img_tensor.shape[1] != self.img_size or img_tensor.shape[2] != self.img_size:
                import torch.nn.functional as F
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            images.append(img_tensor)
        
        images = torch.stack(images, dim=0).unsqueeze(0).unsqueeze(2).to(self.device)
        
        # Process proprioception
        if 'joint_positions' in obs:
            proprio = torch.from_numpy(np.asarray(obs['joint_positions'])).float()
            proprio = proprio.unsqueeze(0).to(self.device)
        else:
            proprio = torch.zeros(1, 14, device=self.device)
        
        return {
            'video': images,
            'text': [self.task_description],
            'proprioception': proprio,
        }
    
    def _run_inference(self, processed_obs: Dict) -> np.ndarray:
        """Run model inference and return actions."""
        with torch.no_grad():
            cond_dict = self.model.get_conditioning_dict(
                video=processed_obs['video'],
                text=processed_obs['text'],
                proprioception=processed_obs['proprioception'],
            )
            
            actions = self.model.forward(
                cond_dict,
                self.noise_scheduler,
                num_inference_steps=self.num_inference_steps,
            )
        
        # Convert rot6d to ypr if needed
        if self.action_space == 'eepose_6drot':
            actions = convert_rot6d_to_ypr(actions)
        
        return actions.squeeze(0).cpu().numpy()
    
    def run(self):
        """Run the inference server loop."""
        print("DWM Inference Server running. Waiting for requests...")
        self.running = True
        
        while self.running:
            try:
                # Receive request (with timeout for graceful shutdown)
                if self.socket.poll(timeout=1000):  # 1 second timeout
                    msg_bytes = self.socket.recv()
                    msg = msgpack.unpackb(msg_bytes, raw=False)
                    
                    cmd = msg.get('cmd')
                    
                    if cmd == 'infer':
                        start_t = time.time()
                        obs = msg['obs']
                        step = msg.get('step', 0)
                        
                        # Process and run inference
                        processed_obs = self._process_obs(obs)
                        actions = self._run_inference(processed_obs)
                        
                        infer_time = time.time() - start_t
                        
                        # Send response
                        reply = {
                            'actions': actions,
                            'step': step,
                            'infer_time': infer_time,
                        }
                        self.socket.send(msgpack.packb(reply, use_bin_type=True))
                        print(f"Step {step}: inference {infer_time*1000:.1f}ms, actions shape {actions.shape}")
                        
                    elif cmd == 'ping':
                        self.socket.send(msgpack.packb({'status': 'ok'}, use_bin_type=True))
                        
                    elif cmd == 'shutdown':
                        self.socket.send(msgpack.packb({'status': 'shutting_down'}, use_bin_type=True))
                        self.running = False
                        
                    else:
                        self.socket.send(msgpack.packb({'error': f'Unknown cmd: {cmd}'}, use_bin_type=True))
                        
            except KeyboardInterrupt:
                print("\nShutting down server...")
                self.running = False
            except Exception as e:
                print(f"Server error: {e}")
                try:
                    self.socket.send(msgpack.packb({'error': str(e)}, use_bin_type=True))
                except:
                    pass
        
        self.socket.close()
        self.context.term()
        print("DWM Inference Server stopped.")
    
    def stop(self):
        """Stop the server."""
        self.running = False


class DWMStreamingClient:
    """Streaming client that runs on Jetson.
    
    Sends observations to desktop server, receives actions.
    Compatible with EgoVerse rollout interface.
    """
    
    def __init__(
        self,
        arm: str,
        server_address: str,
        query_frequency: int,
        extrinsics_key: str,
        timeout_ms: int = 5000,
    ):
        """Initialize streaming client.
        
        Args:
            arm: Which arm(s) to control ("left", "right", or "both")
            server_address: ZMQ server address (e.g., "tcp://192.168.1.100:5555")
            query_frequency: How often to request new actions
            extrinsics_key: Key for camera extrinsics transforms
            timeout_ms: Request timeout in milliseconds
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("zmq and msgpack-numpy required: pip install pyzmq msgpack msgpack-numpy")
        
        self.arm = arm
        self.server_address = server_address
        self.query_frequency = query_frequency
        self.timeout_ms = timeout_ms
        
        # Camera transforms
        self.extrinsics = CameraTransforms(
            intrinsics_key="base",
            extrinsics_key=extrinsics_key
        ).extrinsics
        
        # Setup ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(server_address)
        print(f"DWM Streaming Client connected to {server_address}")
        
        # Test connection
        self._ping()
        
        # Action buffer
        self.actions = None
        self.debug_actions = None
        
    def _ping(self):
        """Test server connection."""
        try:
            self.socket.send(msgpack.packb({'cmd': 'ping'}, use_bin_type=True))
            reply = msgpack.unpackb(self.socket.recv(), raw=False)
            if reply.get('status') == 'ok':
                print("Server connection verified.")
            else:
                raise RuntimeError(f"Server ping failed: {reply}")
        except zmq.error.Again:
            raise RuntimeError(f"Server not responding at {self.server_address}")
    
    def _request_inference(self, obs: Dict, step: int) -> np.ndarray:
        """Request inference from server."""
        # Prepare observation data for transmission
        obs_data = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs_data[key] = val
            elif val is not None:
                obs_data[key] = np.asarray(val)
        
        msg = {
            'cmd': 'infer',
            'obs': obs_data,
            'step': step,
        }
        
        start_t = time.time()
        self.socket.send(msgpack.packb(msg, use_bin_type=True))
        reply_bytes = self.socket.recv()
        reply = msgpack.unpackb(reply_bytes, raw=False)
        roundtrip_t = time.time() - start_t
        
        if 'error' in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        
        actions = reply['actions']
        infer_time = reply.get('infer_time', 0)
        print(f"Step {step}: roundtrip {roundtrip_t*1000:.1f}ms (inference {infer_time*1000:.1f}ms)")
        
        return actions
    
    def _transform_to_base_frame(self, actions: np.ndarray) -> np.ndarray:
        """Transform actions from camera frame to robot base frame."""
        if self.arm == "both":
            left_actions = actions[:, :7]
            right_actions = actions[:, 7:14]
            
            left_6dof = cam_frame_to_base_frame(left_actions[:, :6].copy(), self.extrinsics["left"])
            right_6dof = cam_frame_to_base_frame(right_actions[:, :6].copy(), self.extrinsics["right"])
            
            left_out = np.hstack([left_6dof, left_actions[:, 6:7]])
            right_out = np.hstack([right_6dof, right_actions[:, 6:7]])
            
            return np.hstack([left_out, right_out])
        else:
            arm_offset = 7 if self.arm == "right" else 0
            arm_actions = actions[:, arm_offset:arm_offset + 7]
            
            transformed_6dof = cam_frame_to_base_frame(
                arm_actions[:, :6].copy(),
                self.extrinsics[self.arm]
            )
            
            return np.hstack([transformed_6dof, arm_actions[:, 6:7]])
    
    def rollout_step(self, i: int, obs: Dict) -> Optional[np.ndarray]:
        """Execute one rollout step.
        
        Args:
            i: Current timestep
            obs: Current observations
            
        Returns:
            action: (7,) or (14,) array for single or bimanual control
        """
        if i % self.query_frequency == 0:
            # Request new actions from server
            actions_ypr = self._request_inference(obs, i)
            self.actions = self._transform_to_base_frame(actions_ypr)
            self.debug_actions = self.actions.copy()
        
        if self.actions is None:
            return None
        
        act_idx = i % self.query_frequency
        if act_idx >= self.actions.shape[0]:
            act_idx = self.actions.shape[0] - 1
        
        action = self.actions[act_idx]
        
        if self.arm == "both":
            return action
        elif self.arm == "right":
            return action if action.shape[0] == 7 else action[7:14]
        else:
            return action if action.shape[0] == 7 else action[:7]
    
    def reset(self):
        """Reset client state."""
        self.actions = None
        self.debug_actions = None
        print("DWM streaming client reset")
    
    def close(self):
        """Close connection."""
        self.socket.close()
        self.context.term()
        print("DWM streaming client closed")


# =============================================================================
# Entry point for running inference server standalone
# =============================================================================

def run_inference_server():
    """Run DWM inference server from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DWM Inference Server")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DWM checkpoint (local or s3://)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task description for inference")
    parser.add_argument("--inference-steps", type=int, default=10,
                        help="Number of diffusion denoising steps")
    parser.add_argument("--bind", type=str, default="tcp://*:5555",
                        help="ZMQ bind address")
    parser.add_argument("--img-size", type=int, default=128,
                        help="Image size for model")
    
    args = parser.parse_args()
    
    server = DWMInferenceServer(
        checkpoint_path=args.checkpoint,
        task_description=args.task,
        num_inference_steps=args.inference_steps,
        bind_address=args.bind,
        img_size=args.img_size,
    )
    
    server.run()


if __name__ == "__main__":
    run_inference_server()
