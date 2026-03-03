import os
import sys
import time
from abc import ABC, abstractmethod

import cv2
import h5py
import numpy as np
import torch
from robot_utils import RateLoop
from scipy.spatial.transform import Rotation as R

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.embodiment.eva import Eva
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver
from egomimic.utils.egomimicUtils import (
    CameraTransforms,
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

from robot_interface import ARXInterface


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

    camera_transforms = CameraTransforms(
        intrinsics_key="base", extrinsics_key="x5Dec13_2"
    )
    im_action = visualize_actions(
        img.copy(),
        action_xyz,
        camera_transforms.extrinsics,
        camera_transforms.intrinsics,
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
        extrinsics_key,
        resampled_action_len=None,
        debug=False,
    ):
        super().__init__()
        self.arm = arm
        self.policy_path = policy_path
        self.policy = ModelWrapper.load_from_checkpoint(policy_path, weights_only=False)
        self.query_frequency = query_frequency
        self.cartesian = cartesian
        self.embodiment_id = EMBODIMENT_MAP[self.arm]
        self.embodiment_name = get_embodiment(self.embodiment_id)
        if getattr(self.policy.model, "diffusion", False):
            for head in self.policy.model.nets.policy.heads:
                if isinstance(
                    self.policy.model.nets.policy.heads[head], DenoisingPolicy
                ):
                    self.policy.model.nets.policy.heads[head].num_inference_steps = 10
        self.extrinsics = CameraTransforms(
            intrinsics_key="base", extrinsics_key=extrinsics_key
        ).extrinsics
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy.to(self.policy_device)
        self.debug_actions = None
        self.resampled_action_len = resampled_action_len
        self.debug = debug

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
            for transform in Eva.get_transform_list():
                transform_list_batch = transform.transform(transform_list_batch)
            for k, v in transform_list_batch.items():
                if hasattr(v, "unsqueeze"):
                    transform_list_batch[k] = v.unsqueeze(0)
                elif isinstance(v, np.ndarray):
                    transform_list_batch[k] = v[None, ...]
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
        front = front.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0

        data = {
            "front_img_1": front.squeeze(),
            "pad_mask": torch.ones((1, 100, 1), device=self.device, dtype=torch.bool),
        }

        eepose = obs["ee_poses"]

        if self.arm in ["right", "both"]:
            right = torch.from_numpy(
                obs["right_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            right = right[..., [2, 1, 0]]  # BGR -> RGB
            right = (
                right.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0
            )
            data["right_wrist_img"] = right.squeeze()
            right_ee_pose = eepose[7:13]
            right_ee_pose = ee_pose_to_rot_ee_frame(right_ee_pose)
            right_ypr = right_ee_pose[..., 3:6]
            right_xyzw = R.from_euler("ZYX", right_ypr).as_quat()
            right_wxyz = xyzw_to_wxyz(right_xyzw)
            right_xyzwxyz = np.concatenate([eepose[:3], right_wxyz], axis=-1)
            data["right.obs_ee_pose"] = torch.from_numpy(right_xyzwxyz).reshape(-1)
            data["right.obs_gripper"] = torch.from_numpy(eepose[13:14]).reshape(-1)
            right_gripper = torch.from_numpy(eepose[13:14]).view(1, 1).repeat(45, 1)
            data["right.gripper"] = right_gripper
            # dummy command ee pose
            right_cmd_ee_pose = torch.from_numpy(right_xyzwxyz).view(1, 7).repeat(45, 1)
            data["right.cmd_ee_pose"] = right_cmd_ee_pose

        if self.arm in ["left", "both"]:
            left = torch.from_numpy(
                obs["left_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            left = left[..., [2, 1, 0]]  # BGR -> RGB
            left = left.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0
            data["left_wrist_img"] = left.squeeze()
            left_ee_pose = eepose[0:6]
            left_ee_pose = ee_pose_to_rot_ee_frame(left_ee_pose)
            left_ypr = left_ee_pose[..., 3:6]
            left_xyzw = R.from_euler("ZYX", left_ypr).as_quat()
            left_wxyz = xyzw_to_wxyz(left_xyzw)
            left_xyzwxyz = np.concatenate([eepose[:3], left_wxyz], axis=-1)
            data["left.obs_ee_pose"] = torch.from_numpy(left_xyzwxyz).reshape(-1)
            data["left.obs_gripper"] = torch.from_numpy(eepose[6:7]).reshape(-1)
            left_gripper = torch.from_numpy(eepose[6:7]).view(1, 1).repeat(45, 1)
            data["left.gripper"] = left_gripper
            # dummy command ee pose
            left_cmd_ee_pose = torch.from_numpy(left_xyzwxyz).view(1, 7).repeat(45, 1)
            data["left.cmd_ee_pose"] = left_cmd_ee_pose

        if self.arm == "both":
            data["embodiment"] = ["eva_bimanual"]
            data["metadata.robot_name"] = ["eva_bimanual"]
        elif self.arm == "right":
            data["embodiment"] = ["eva_right_arm"]
            data["metadata.robot_name"] = ["eva_right_arm"]
        elif self.arm == "left":
            data["embodiment"] = ["eva_left_arm"]
            data["metadata.robot_name"] = ["eva_left_arm"]

        return data

    def reset(self):
        self.actions = None
        self.debug_actions = None
        self.policy = ModelWrapper.load_from_checkpoint(
            self.policy_path, weights_only=False
        )
        if getattr(self.policy.model, "diffusion", False):
            for head in self.policy.model.nets.policy.heads:
                if isinstance(
                    self.policy.model.nets.policy.heads[head], DenoisingPolicy
                ):
                    self.policy.model.nets.policy.heads[head].num_inference_steps = 10


def debug_policy(
    obs, camera_transforms, policy, step_i, cartesian, arms, kinematics_solver
):
    os.makedirs("debug", exist_ok=True)
    if isinstance(obs["front_img_1"], torch.Tensor):
        if obs["front_img_1"].dim() == 4:
            img = obs["front_img_1"][0].permute(1, 2, 0).cpu().numpy()
        elif obs["front_img_1"].dim() == 3:
            img = obs["front_img_1"].permute(1, 2, 0).cpu().numpy()
        else:
            img = obs["front_img_1"].cpu().numpy()
    else:
        img = obs["front_img_1"]
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)

    if cartesian:
        if arms == "both":
            left_actions = policy.debug_actions[:, :3]
            right_actions = policy.debug_actions[:, 7:10]
            action_xyz = np.hstack([left_actions, right_actions])
        else:
            action_xyz = policy.debug_actions[:, :3]
    else:
        jnts = policy.actions[:, :7]
        actions_xyz = np.zeros((jnts.shape[0], 3), dtype=np.float32)
        for j in range(actions_xyz.shape[0]):
            pos, _rot = kinematics_solver.fk(jnts[j][:6])
            actions_xyz[j] = pos
        action_xyz = actions_xyz

    im_viz = visualize_actions(
        img,
        action_xyz,
        camera_transforms.extrinsics,
        camera_transforms.intrinsics,
        arm=arms,
    )
    cv2.imwrite(f"debug/debug_{step_i}.png", im_viz)
    if (
        cartesian
        and arms == "both"
        and policy.debug_actions is not None
        and policy.debug_actions.ndim == 2
        and policy.debug_actions.shape[1] in (12, 14)
    ):
        eva_viz_batch = {
            "observations.images.front_img_1": torch.from_numpy(
                obs["front_img_1"][None, ...]
            ),
            "actions_cartesian": torch.from_numpy(
                policy.debug_actions.astype(np.float32, copy=False)[None, ...]
            ),
        }
        im_axes = Eva.viz_transformed_batch(eva_viz_batch, mode="palm_axes")
        cv2.imwrite(f"debug/debug_axes_{step_i}.png", im_axes)


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
):
    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]

    ri = ARXInterface(arms=arms_list)

    if policy_path is not None:
        rollout_type = "policy"
        policy = PolicyRollout(
            arm=arms,
            policy_path=policy_path,
            query_frequency=query_frequency,
            cartesian=cartesian,
            extrinsics_key="x5Dec13_2",
            resampled_action_len=resampled_action_len,
            debug=debug,
        )
    elif dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
    else:
        raise ValueError(
            "Must provide either --policy-path or --dataset-path (and optionally --repo-id)."
        )

    print(f"Cartesian value {cartesian}")

    camera_transforms = CameraTransforms(
        intrinsics_key="base", extrinsics_key="x5Dec13_2"
    )
    kinematics_solver = EvaMinkKinematicsSolver(
        model_path="/home/robot/robot_ws/egomimic/resources/model_x5.xml"
    )

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)

            while True:  # restartable
                with RateLoop(frequency=frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch == "q":
                            print("Quit requested.")
                            return
                        if ch == "r":
                            print("Restart requested.")
                            reset_rollout(ri, policy)
                            time.sleep(10.0)
                            break  # restart RateLoop

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
                            print(
                                "Finish rollout. Press 'r' to restart or 'q' to quit."
                            )
                            while True:
                                ch2 = kp.getch()
                                if ch2 == "q":
                                    return
                                if ch2 == "r":
                                    reset_rollout(ri, policy)
                                    time.sleep(10.0)
                                    break
                                time.sleep(0.01)
                            break

                        if debug and rollout_type == "policy":
                            debug_policy(
                                obs,
                                camera_transforms,
                                policy,
                                step_i,
                                cartesian,
                                arms,
                                kinematics_solver,
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rollout robot model.")
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

    args = parser.parse_args()

    print(f"Resampling actions to {args.resampled_action_len}")
    main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        cartesian=args.cartesian,
        debug=args.debug,
        resampled_action_len=args.resampled_action_len,
    )
