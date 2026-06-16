# ruff: noqa: E402
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore", message="Can't initialize NVML")

import cv2
import h5py
import numpy as np
import torch
from robot_utils import RateLoop
from scipy.spatial.transform import Rotation as R
from torch.utils.data import default_collate

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_data_utils import build_tokenized_collate
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.utils.egomimicUtils import (
    cam_frame_to_base_frame,
    draw_actions,
    interpolate_arr,
    interpolate_arr_euler,
)
from egomimic.utils.pose_utils import xyzw_to_wxyz

sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))

import select
import sys
import termios
import tty


def visualize_actions(ims, actions, extrinsics, intrinsics, arm="both"):
    if actions.shape[-1] == 7 or actions.shape[-1] == 14:
        ac_type = "joints"
    elif actions.shape[-1] == 3 or actions.shape[-1] == 6:
        ac_type = "xyz"
    else:
        raise ValueError(f"Unknown action type with shape {actions.shape}")

    ims = draw_actions(
        ims, ac_type, "Purples", actions, extrinsics, intrinsics, arm=arm
    )

    return ims


R_t_e = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ],
    dtype=float,
)

inv_R_t_e = np.linalg.inv(R_t_e)


def ee_pose_to_rot_ee_frame_batch(pose):
    pose = np.asarray(pose)
    xyz = pose[..., :3]
    ypr = pose[..., 3:6]
    R_ee = R.from_euler("ZYX", ypr).as_matrix()
    R_rot = R_t_e @ R_ee
    ypr_rot = R.from_matrix(R_rot).as_euler("ZYX")
    return np.concatenate([xyz, ypr_rot], axis=-1)


def rot_ee_frame_to_ee_pose_batch(pose_rot):
    pose_rot = np.asarray(pose_rot)
    xyz = pose_rot[..., :3]
    ypr = pose_rot[..., 3:6]
    R_rot = R.from_euler("ZYX", ypr).as_matrix()
    R_ee = inv_R_t_e @ R_rot
    ypr_ee = R.from_matrix(R_ee).as_euler("ZYX")
    return np.concatenate([xyz, ypr_ee], axis=-1)


def ee_pose_to_rot_ee_frame(pose):
    return ee_pose_to_rot_ee_frame_batch(pose[None, ...])[0]


def rot_ee_frame_to_ee_pose(pose_rot):
    return rot_ee_frame_to_ee_pose_batch(pose_rot[None, ...])[0]


def viz_rot_ee_pose(image, eepose, action_image_path, rot_image_path):
    """
    Save both cartesian-action and orientation-axis visualizations for an EVA
    action chunk using the same conventions as the debug path.
    """
    arr = np.asarray(eepose, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, ...]
    if arr.ndim != 2 or arr.shape[1] not in (12, 14):
        raise ValueError(f"Expected eepose shape (T, 12|14), got {arr.shape}")

    os.makedirs(os.path.dirname(action_image_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(rot_image_path) or ".", exist_ok=True)

    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(
            f"Expected image shape (H, W, 3) or (3, H, W), got {img.shape}"
        )
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if arr.shape[1] == 14:
        left_xyz = arr[:, :3]
        right_xyz = arr[:, 7:10]
    else:
        left_xyz = arr[:, :3]
        right_xyz = arr[:, 6:9]
    action_xyz = np.hstack([left_xyz, right_xyz]).astype(np.float32, copy=False)

    im_action = visualize_actions(
        img.copy(),
        action_xyz,
        Eva.EXTRINSICS,
        Human.INTRINSICS,
        arm="both",
    )
    cv2.imwrite(action_image_path, im_action)

    eva_viz_batch = {
        "observations.images.front_img_1": torch.from_numpy(img[None, ...]),
        "actions_cartesian": torch.from_numpy(arr[None, ...]),
    }
    im_rot = Eva.viz_transformed_batch(eva_viz_batch, mode="palm_axes")
    cv2.imwrite(rot_image_path, im_rot)
    return im_action, im_rot


GRIPPER_WIDTH = 0.09
# Control parameters
DEFAULT_FREQUENCY = 30  # Hz
QUERY_FREQUENCY = 30
DEFAULT_RESAMPLE_LENGTH = 45

RIGHT_CAM_SERIAL = ""
LEFT_CAM_SERIAL = ""

EMBODIMENT_MAP = {
    "both": 8,
    "left": 7,
    "right": 6,
}

TEMP_DIR = "/home/robot/temp_dir"


def _build_robot_interface(arms_list, offline_debug=False, offline_episode_path=None):
    if offline_debug:
        from robot_interface import OfflineARXInterface

        return OfflineARXInterface(arms=arms_list, dataset_path=offline_episode_path)

    from robot_interface import ARXInterface

    return ARXInterface(arms=arms_list)


def _get_model_xml_path():
    candidates = [
        "/home/robot/robot_ws/egomimic/resources/model_x5.xml",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "resources", "model_x5.xml")
        ),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


class _KeyPoll:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # no Enter needed
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def getch(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class Rollout(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def rollout_step(self, i):
        pass


class ReplayRollout(Rollout):
    def __init__(self, dataset_path, cartesian):
        super().__init__()
        self.dataset_path = dataset_path
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"HDF5 not found: {self.dataset_path}")
        with h5py.File(self.dataset_path, "r") as f:
            if cartesian:
                self.actions = np.asarray(f["actions"]["eepose"][...], dtype=np.float32)
            else:
                self.actions = np.asarray(
                    f["observations"]["joint_positions"][...], dtype=np.float32
                )

    def rollout_step(self, i):
        if i < self.actions.shape[0]:
            return self.actions[i]
        else:
            return None


class PolicyRollout(Rollout):
    def __init__(
        self,
        arm,
        policy_path,
        query_frequency,
        cartesian,
        resampled_action_len=None,
        debug=False,
        annotation_path=None,
    ):
        super().__init__()
        self.arm = arm
        self.policy_path = policy_path
        self.query_frequency = query_frequency
        self.cartesian = cartesian
        self.embodiment_id = EMBODIMENT_MAP[self.arm]
        self.embodiment_name = get_embodiment(self.embodiment_id)
        self.extrinsics = Eva.EXTRINSICS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_device = self.device
        print(f"[rollout] Loading policy from {self.policy_path}")
        self.policy = self._load_policy()
        self.debug_actions = None
        self.resampled_action_len = resampled_action_len
        self.debug = debug
        self.transform_list = Eva.get_transform_list(mode="cartesian_wristframe_ypr")
        self.annotation = None
        self._tokenizer = None
        self.collate_fn = default_collate
        if annotation_path is not None:
            if not os.path.isfile(annotation_path):
                print(
                    f"[rollout] WARNING: annotation file not found: {annotation_path}  (continuing without annotation)"
                )
            else:
                with open(annotation_path, "r") as f:
                    self.annotation = f.read().strip()
                self.collate_fn = build_tokenized_collate(
                    max_length=128,
                    model_name="google/paligemma-3b-mix-224",
                    sampling_mode="first",
                    annotation_key="annotations",
                    default_prompt=self.annotation,
                )

    LOCAL_WEIGHT_PATH = (
        "/home/robot/robot_ws/egomimic/algo/pi_checkpoints/pi05_base_pytorch"
    )

    @classmethod
    def _patch_checkpoint_paths(cls, ckpt_path):
        """Rewrite pytorch_weight_path in the checkpoint's saved config
        to point to the local base model weights."""
        import torch as _torch
        from omegaconf import DictConfig, OmegaConf

        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ht = ckpt.get("hyper_parameters", {}).get("config_tree")
        if ht is None:
            return ckpt_path
        if isinstance(ht, DictConfig):
            cfg = OmegaConf.to_container(ht, resolve=True)
        else:
            cfg = ht
        # Navigate to pytorch_weight_path in the config
        robomimic = cfg.get("model", {}).get("robomimic_model", {})
        config = robomimic.get("config", {})
        old_path = config.get("pytorch_weight_path")
        if old_path is None or old_path == cls.LOCAL_WEIGHT_PATH:
            return ckpt_path
        print(
            f"[rollout] Patching pytorch_weight_path: {old_path} -> {cls.LOCAL_WEIGHT_PATH}"
        )
        config["pytorch_weight_path"] = cls.LOCAL_WEIGHT_PATH
        ckpt["hyper_parameters"]["config_tree"] = OmegaConf.create(cfg)
        patched_path = ckpt_path + ".patched"
        _torch.save(ckpt, patched_path)
        print(f"[rollout] Patched checkpoint saved to {patched_path}")
        return patched_path

    def _load_policy(self):
        patched_path = self._patch_checkpoint_paths(self.policy_path)
        policy = ModelWrapper.load_from_checkpoint(
            patched_path, weights_only=False, map_location="cpu"
        )
        policy = policy.to(self.policy_device)
        policy.eval()
        policy.model.device = self.policy_device

        # Unwrap torch.compile on sample_actions to avoid massive first-call
        # compilation overhead (~50s). The compiled version (instance attribute)
        # shadows the original class method; deleting it restores the fast
        # uncompiled path which is sufficient for real-time rollout.
        pi0 = policy.model.nets["policy"]
        if "sample_actions" in vars(pi0):
            del pi0.sample_actions
            print(
                "[rollout] Disabled torch.compile on sample_actions for rollout inference"
            )

        # Verify model is on GPU
        try:
            p = next(pi0.parameters())
            print(f"[rollout] Model device: {p.device}, dtype: {p.dtype}")
            if not p.is_cuda:
                print(
                    "[rollout] WARNING: model is NOT on GPU — inference will be very slow!"
                )
        except StopIteration:
            pass

        if getattr(policy.model, "diffusion", False):
            for head in policy.model.nets.policy.heads:
                if isinstance(policy.model.nets.policy.heads[head], DenoisingPolicy):
                    policy.model.nets.policy.heads[head].num_inference_steps = 10
        return policy

    def _downsample_chunk(self, chunk: np.ndarray, target_len: int) -> np.ndarray:
        if target_len is None or target_len <= 0 or chunk.shape[0] == target_len:
            return chunk.astype(np.float32, copy=False)

        # chunk: (T, D) -> (1, T, D) and back
        if self.cartesian:
            if self.arm == "both":
                left = chunk[:, :7]
                right = chunk[:, 7:14]
                left_r = interpolate_arr_euler(left[None, ...], target_len)[0]
                right_r = interpolate_arr_euler(right[None, ...], target_len)[0]
                out = np.hstack([left_r, right_r])
            else:
                out = interpolate_arr_euler(chunk[None, ...], target_len)[0]
        else:
            out = interpolate_arr(chunk[None, ...], target_len)[0]

        return out.astype(np.float32, copy=False)

    def rollout_step(self, i, obs):
        if i % self.query_frequency == 0:
            start_infer_t = time.time()
            transform_list_batch = self.process_obs_for_transform_list(obs)
            for transform in self.transform_list:
                transform_list_batch = transform.transform(transform_list_batch)
            transform_list_batch = self.collate_fn([transform_list_batch])
            if self.arm == "both":
                embodiment_name = "eva_bimanual"
            elif self.arm == "right":
                embodiment_name = "eva_right_arm"

            elif self.arm == "left":
                embodiment_name = "eva_left_arm"
            batch = {
                embodiment_name: transform_list_batch,
            }
            processed_batch = self.policy.model.process_batch_for_training(batch)
            preds = self.policy.model.forward_eval(processed_batch)[
                f"{embodiment_name}_actions_cartesian"
            ]
            self.actions = preds.detach().cpu().numpy().squeeze()
            self.debug_actions = self.actions.copy()
            if self.cartesian:
                if self.arm == "both":
                    left_actions = self.actions[:, :7]
                    right_actions = self.actions[:, 7:]

                    transformed_left = cam_frame_to_base_frame(
                        left_actions[:, :6].copy(), self.extrinsics["left"]
                    )
                    transformed_right = cam_frame_to_base_frame(
                        right_actions[:, :6].copy(), self.extrinsics["right"]
                    )
                    transformed_left = rot_ee_frame_to_ee_pose_batch(transformed_left)
                    transformed_right = rot_ee_frame_to_ee_pose_batch(transformed_right)
                    gripper_left = left_actions[:, 6:7]
                    gripper_right = right_actions[:, 6:7]
                    if left_actions.shape[1] == 7:
                        left_actions = np.hstack([transformed_left, gripper_left])
                    else:
                        left_actions = transformed_left
                    if right_actions.shape[1] == 7:
                        right_actions = np.hstack([transformed_right, gripper_right])
                    else:
                        right_actions = transformed_right
                    self.actions = np.hstack([left_actions, right_actions])
                else:
                    eepose = rot_ee_frame_to_ee_pose_batch(self.actions[:, :6].copy())
                    self.actions[:, :6] = eepose
                    transformed_6dof = cam_frame_to_base_frame(
                        self.actions[:, :6].copy(), self.extrinsics[self.arm]
                    )
                    # Preserve gripper if present (7th value)
                    gripper = self.actions[:, 6:7]
                    if self.actions.shape[1] == 7:
                        self.actions = np.hstack([transformed_6dof, gripper])
                    else:
                        self.actions = transformed_6dof

            if self.resampled_action_len is not None:
                self.actions = self._downsample_chunk(
                    self.actions, self.resampled_action_len
                )
            # print(f"actions: {self.actions[6:7]}, debug_actions: {self.debug_actions[6:7]}")

            print(f"Inference time: {(time.time() - start_infer_t)}s")

        act_i = i % self.query_frequency
        return self.actions[act_i]

    def process_obs_for_transform_list(self, obs):
        # front camera: obs["front_img_1"] is BGR, shape [H, W, 3]
        front = torch.from_numpy(obs["front_img_1"][None, ...])  # [1, H, W, 3]
        front = front[..., [2, 1, 0]]  # BGR -> RGB
        front = front.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        front = front.squeeze()
        data = {
            # Keep rollout-local keys, PI schematic aliases, and canonical
            # dataset zarr keys so checkpoints with different data schematics
            # can all resolve the same image tensor.
            "front_img_1": front,
            "base_0_rgb": front,
            "observations.images.front_img_1": front,
            "pad_mask": torch.ones((1, 100, 1), dtype=torch.bool),
        }

        eepose = obs["ee_poses"]

        if self.arm in ["right", "both"]:
            right = torch.from_numpy(
                obs["right_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            right = right[..., [2, 1, 0]]  # BGR -> RGB
            right = right.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
            data["right_wrist_img"] = right.squeeze()
            data["right_wrist_0_rgb"] = data["right_wrist_img"]
            data["observations.images.right_wrist_img"] = data["right_wrist_img"]
            right_ee_pose = eepose[7:13]
            right_ee_pose = ee_pose_to_rot_ee_frame(right_ee_pose)
            right_ypr = right_ee_pose[..., 3:6]
            right_xyzw = R.from_euler("ZYX", right_ypr).as_quat()
            right_wxyz = xyzw_to_wxyz(right_xyzw)
            right_xyzwxyz = np.concatenate([eepose[7:10], right_wxyz], axis=-1)
            data["right.obs_ee_pose"] = torch.from_numpy(right_xyzwxyz).reshape(-1)
            data["right.obs_gripper"] = torch.from_numpy(eepose[13:14]).reshape(-1)
            right_gripper = torch.from_numpy(eepose[13:14]).view(1, 1).repeat(45, 1)
            data["right.cmd_gripper"] = right_gripper
            right_cmd_ee_pose = torch.from_numpy(right_xyzwxyz).view(1, 7).repeat(45, 1)
            data["right.cmd_ee_pose"] = right_cmd_ee_pose

        if self.arm in ["left", "both"]:
            left = torch.from_numpy(
                obs["left_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            left = left[..., [2, 1, 0]]  # BGR -> RGB
            left = left.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
            data["left_wrist_img"] = left.squeeze()
            data["left_wrist_0_rgb"] = data["left_wrist_img"]
            data["observations.images.left_wrist_img"] = data["left_wrist_img"]
            left_ee_pose = eepose[0:6]
            left_ee_pose = ee_pose_to_rot_ee_frame(left_ee_pose)
            left_ypr = left_ee_pose[..., 3:6]
            left_xyzw = R.from_euler("ZYX", left_ypr).as_quat()
            left_wxyz = xyzw_to_wxyz(left_xyzw)
            left_xyzwxyz = np.concatenate([eepose[:3], left_wxyz], axis=-1)
            data["left.obs_ee_pose"] = torch.from_numpy(left_xyzwxyz).reshape(-1)
            data["left.obs_gripper"] = torch.from_numpy(eepose[6:7]).reshape(-1)
            left_gripper = torch.from_numpy(eepose[6:7]).view(1, 1).repeat(45, 1)
            data["left.cmd_gripper"] = left_gripper
            left_cmd_ee_pose = torch.from_numpy(left_xyzwxyz).view(1, 7).repeat(45, 1)
            data["left.cmd_ee_pose"] = left_cmd_ee_pose

        if self.arm == "both":
            data["embodiment"] = ["eva_bimanual"]
        elif self.arm == "right":
            data["embodiment"] = ["eva_right_arm"]
        elif self.arm == "left":
            data["embodiment"] = "eva_left_arm"

        if self.annotation is not None:
            data["annotations"] = [self.annotation]

        return data

    def load_annotation(self, annotation_path):
        """Load a new annotation file, building the tokenized collate only if needed.

        The annotation text flows through data["annotations"] at each inference
        step, so updating self.annotation is sufficient when the tokenized
        collate already exists.  We only build it when the collate is still the
        plain default_collate (i.e. no annotation was provided at init time).

        Returns True on success, False if the file could not be loaded.
        """
        if not os.path.isfile(annotation_path):
            print(f"[rollout] WARNING: annotation file not found: {annotation_path}")
            return False
        with open(annotation_path, "r") as f:
            self.annotation = f.read().strip()
        if self.collate_fn is default_collate:
            self.collate_fn = build_tokenized_collate(
                max_length=128,
                model_name="google/paligemma-3b-mix-224",
                sampling_mode="first",
                annotation_key="annotations",
                default_prompt=self.annotation,
            )
        print(
            f"[rollout] Loaded new annotation from {annotation_path}: '{self.annotation}'"
        )
        return True

    def reset(self):
        self.actions = None
        self.debug_actions = None
        self.policy.eval()


def debug_policy(actions, front_img, step_i):
    os.makedirs("debug", exist_ok=True)

    if isinstance(front_img, torch.Tensor):
        if front_img.dim() == 4:
            front_img = front_img[0].permute(1, 2, 0).cpu().numpy()
        elif front_img.dim() == 3:
            if front_img.shape[0] == 3:
                front_img = front_img.permute(1, 2, 0).cpu().numpy()
            else:
                front_img = front_img.cpu().numpy()
    elif front_img.ndim == 3 and front_img.shape[0] == 3:
        front_img = front_img.transpose(1, 2, 0)
    front_img = front_img.astype(np.uint8)

    actions = actions.squeeze()
    eva_viz_batch = {
        "observations.images.front_img_1": torch.from_numpy(front_img[None, ...]),
        "actions_cartesian": torch.from_numpy(
            actions.astype(np.float32, copy=False)[None, ...]
        ),
    }
    im_viz = Eva.viz_transformed_batch(eva_viz_batch, mode="traj+rotation")

    cv2.imwrite(f"debug/debug_{step_i}.png", im_viz)
    breakpoint()


def reset_rollout(ri, policy):
    print("Resetting rollout: going home + clearing policy state")
    if isinstance(policy, ReplayRollout):
        return
    ri.set_home()
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "actions"):
        policy.actions = None
    if hasattr(policy, "debug_actions"):
        policy.debug_actions = None


def main(
    arms,
    frequency,
    cartesian,
    query_frequency=None,
    policy_path=None,
    dataset_path=None,
    debug=False,
    resampled_action_len=None,
    offline_debug=False,
    offline_episode_path=None,
    annotation_path=None,
):
    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]

    if offline_episode_path is not None and not offline_debug:
        raise ValueError("--offline-episode-path requires --offline-debug.")
    if policy_path is not None and offline_debug and offline_episode_path is None:
        raise ValueError(
            "--policy-path requires --offline-episode-path in --offline-debug mode."
        )

    ri = _build_robot_interface(
        arms_list=arms_list,
        offline_debug=offline_debug,
        offline_episode_path=offline_episode_path,
    )

    if policy_path is not None:
        rollout_type = "policy"
        policy = PolicyRollout(
            arm=arms,
            policy_path=policy_path,
            query_frequency=query_frequency,
            cartesian=cartesian,
            resampled_action_len=resampled_action_len,
            debug=debug,
            annotation_path=annotation_path,
        )
    elif dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
    else:
        raise ValueError(
            "Must provide either --policy-path or --dataset-path (and optionally --repo-id)."
        )

    print(f"Cartesian value {cartesian}")

    def _enter_intervention(kp, policy, rollout_type):
        """Pause rollout and wait for user command.

        Restores the terminal to cooked mode so the user can type full
        commands, then re-enters cbreak mode before returning.

        Returns one of:
            "continue"  – resume rollout
            "restart"   – restart rollout
            "quit"      – exit program
        """
        # Restore normal terminal so the user can type freely
        termios.tcsetattr(kp.fd, termios.TCSADRAIN, kp.old)
        print("\n--- INTERVENTION (rollout paused) ---")
        print("  c            : continue rollout")
        print("  a <path>     : load new annotation file")
        print("  r            : restart rollout")
        print("  q            : quit")

        while True:
            try:
                cmd = input("> ").strip()
            except EOFError:
                tty.setcbreak(kp.fd)
                return "quit"

            if cmd == "c":
                print("Resuming rollout.")
                tty.setcbreak(kp.fd)
                return "continue"
            elif cmd == "q":
                tty.setcbreak(kp.fd)
                return "quit"
            elif cmd == "r":
                tty.setcbreak(kp.fd)
                return "restart"
            elif cmd.startswith("a "):
                ann_path = cmd[2:].strip()
                if not ann_path:
                    print("Usage: a <annotation_path>")
                    continue
                if rollout_type != "policy" or not isinstance(policy, PolicyRollout):
                    print("Annotation loading is only supported for policy rollouts.")
                    continue
                policy.load_annotation(ann_path)
            else:
                print(f"Unknown command: '{cmd}'. Use c / a <path> / r / q.")

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)
            # Enter intervention at startup so the user decides when to begin
            result = _enter_intervention(kp, policy, rollout_type)
            if result == "quit":
                print("Quit requested.")
                return
            if result == "restart":
                reset_rollout(ri, policy)

            while True:  # restartable
                with RateLoop(frequency=frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch is not None:
                            # Any key press triggers intervention
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                print("Quit requested.")
                                return
                            elif result == "restart":
                                print("Restart requested.")
                                reset_rollout(ri, policy)
                                result = _enter_intervention(kp, policy, rollout_type)
                                if result == "quit":
                                    return
                                if result == "restart":
                                    reset_rollout(ri, policy)
                                break
                            if hasattr(policy, "actions"):
                                policy.actions = None
                            break

                        actions = None
                        if rollout_type == "policy":
                            obs = ri.get_obs()
                            actions = policy.rollout_step(step_i, obs)
                        elif rollout_type == "replay":
                            actions = policy.rollout_step(step_i)
                        elif rollout_type == "replay_lerobot":
                            actions = policy.rollout_step(step_i)
                        else:
                            raise ValueError(f"Invalid rollout type: {rollout_type}")

                        if actions is None:
                            print("Finish rollout.")
                            reset_rollout(ri, policy)
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                return
                            if result == "restart":
                                reset_rollout(ri, policy)
                            break

                        if (
                            debug
                            and rollout_type == "policy"
                            and step_i % query_frequency == 0
                        ):
                            debug_actions = policy.debug_actions
                            front_img = obs["front_img_1"]
                            debug_policy(
                                debug_actions,
                                front_img,
                                step_i,
                            )

                        for arm in arms_list:
                            arm_offset = 7 if (arm == "right" and arms == "both") else 0
                            arm_action = actions[arm_offset : arm_offset + 7]
                            if cartesian:
                                ri.set_pose(arm_action, arm)
                            else:
                                ri.set_joints(arm_action, arm)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting rollout.")
        return


def build_arg_parser(description="Rollout robot model."):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--arms",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="Which arm(s) to control",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_FREQUENCY,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--query_frequency",
        type=int,
        default=QUERY_FREQUENCY,
        help="Frames which model does inference",
    )
    parser.add_argument("--policy-path", type=str, help="policy checkpoint path")
    parser.add_argument("--dataset-path", type=str, help="dataset path for replay")
    parser.add_argument(
        "--offline-debug",
        action="store_true",
        help="use the offline dummy robot interface for rollout debugging",
    )
    parser.add_argument(
        "--offline-episode-path",
        type=str,
        help="local EVA Zarr episode path used as observation source in offline debug mode",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="control in cartesian space instead of joint space",
    )
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=DEFAULT_RESAMPLE_LENGTH,
        help="Resample each predicted action chunk to this length (e.g., 100 -> 45). Euler if --cartesian.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug visualization of actions on images",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        help="path to the annotation file",
    )
    return parser


def run_from_args(args):
    print(f"Resampling actions to {args.resampled_action_len}")
    return main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        cartesian=args.cartesian,
        debug=args.debug,
        resampled_action_len=args.resampled_action_len,
        offline_debug=args.offline_debug,
        offline_episode_path=args.offline_episode_path,
        annotation_path=args.annotation_path,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args)
