import os
import time
import h5py
import torch
import sys
import numpy as np
import cv2
from abc import ABC, abstractmethod

from egomimic.rldb.utils import (
    EMBODIMENT,
    get_embodiment,
    get_embodiment_id,
    RLDBDataset,
)
from egomimic.pl_utils.pl_model import ModelWrapper

from robot_utils import RateLoop
from egomimic.utils.egomimicUtils import (
    CameraTransforms,
    draw_actions,
    cam_frame_to_base_frame,
    ee_pose_to_cam_frame,
    base_frame_to_cam_frame,
)
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver

sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))
from robot_interface import *


# from stream_aria import AriaRecorder
# from stream_d405 import RealSenseRecorder
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


# Control parameters
DEFAULT_FREQUENCY = 30  # Hz
QUERY_FREQUENCY = 30

RIGHT_CAM_SERIAL = ""
LEFT_CAM_SERIAL = ""

EMBODIMENT_MAP = {
    "both": 8,
    "left": 7,
    "right": 6,
}


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


# TODO: Work with all types of arms
class ReplayRolloutLerobot(Rollout):
    def __init__(
        self,
        dataset_path,
        repo_id,
        cartesian,
        extrinsics_key,
        episodes=[1],
        arm="right",
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.cartesian = cartesian
        self.extrinsics = CameraTransforms(
            intrinsics_key="base", extrinsics_key=extrinsics_key
        ).extrinsics
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_actions = None
        self.arm = arm

        dataset = RLDBDataset(
            repo_id=repo_id,
            root=dataset_path,
            local_files_only=True,
            episodes=episodes,
            mode="sample",
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        self.iter = iter(data_loader)
        self.data_loader = data_loader
        self.i = 0
        self.actions_key = "actions_cartesian" if cartesian else "actions_joints"
        self.actions = np.empty((0, 7), dtype=np.float32)

    def rollout_step(self, i):
        while i >= self.i:
            batch = next(self.iter)
            cur_actions = (
                batch[self.actions_key].cpu().numpy()[:, 0, :]
            )  # (B, N, 7) -> (B, 7)
            if self.cartesian:
                gripper_pose = (
                    cur_actions[:, 6:7] * 0.08
                )  # TODO: make this in robot config
                cur_actions = cam_frame_to_base_frame(
                    cur_actions, self.extrinsics[self.arm]
                )
                cur_actions = np.concatenate([cur_actions, gripper_pose], axis=1)

            self.actions = np.concatenate([self.actions, cur_actions], axis=0)
            self.i += cur_actions.shape[0]
        return self.actions[i]


class PolicyRollout(Rollout):
    def __init__(self, arm, policy_path, query_frequency, cartesian, extrinsics_key):
        super().__init__()
        self.arm = arm
        self.policy = ModelWrapper.load_from_checkpoint(policy_path)
        self.query_frequency = query_frequency
        self.cartesian = cartesian
        self.embodiment_id = EMBODIMENT_MAP[self.arm]
        self.embodiment_name = get_embodiment(self.embodiment_id)
        self.extrinsics = CameraTransforms(
            intrinsics_key="base", extrinsics_key=extrinsics_key
        ).extrinsics
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_actions = None

    def rollout_step(self, i, obs):
        if i % self.query_frequency == 0:
            start_infer_t = time.time()
            batch = self.process_obs_for_policy(obs)
            preds = self.policy.model.forward_eval(batch)
            ac_key = self.policy.model.ac_keys[self.embodiment_id]
            actions = preds[f"{self.embodiment_name.lower()}_{ac_key}"]
            self.actions = actions.detach().cpu().numpy().squeeze()
            self.debug_actions = self.actions.copy()
            if self.cartesian:
                transformed_6dof = cam_frame_to_base_frame(
                    self.actions[:, :6].copy(), self.extrinsics[self.arm]
                )
                # Preserve gripper if present (7th value)
                if self.actions.shape[1] == 7:
                    self.actions = np.hstack([transformed_6dof, self.actions[:, 6:7]])
                else:
                    self.actions = transformed_6dof

            print(f"Inference time: {(time.time() - start_infer_t)}s")

        # TODO check gripper if we are using 0 to 0.08 or 0 to 1
        act_i = i % self.query_frequency
        return self.actions[act_i]

    def process_obs_for_policy(self, obs):
        # front camera: obs["front_img_1"] is BGR, shape [H, W, 3]
        front = torch.from_numpy(obs["front_img_1"][None, ...])        # [1, H, W, 3]
        front = front[..., [2, 1, 0]]                                  # BGR -> RGB
        front = front.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0

        data = {
            "front_img_1": front,
            "pad_mask": torch.ones((1, 100, 1), device=self.device, dtype=torch.bool),
        }

        if self.arm == "right":
            right = torch.from_numpy(obs["right_wrist_img"][None, ...])  # [1, H, W, 3] BGR
            right = right[..., [2, 1, 0]]                                # BGR -> RGB
            right = (
                right.permute(0, 3, 1, 2)
                .to(self.device, dtype=torch.float32)
                / 255.0
            )
            data["right_wrist_img"] = right
            joint_positions = obs["joint_positions"][7:]

        elif self.arm == "left":
            left = torch.from_numpy(obs["left_wrist_img"][None, ...])    # [1, H, W, 3] BGR
            left = left[..., [2, 1, 0]]                                  # BGR -> RGB
            left = (
                left.permute(0, 3, 1, 2)
                .to(self.device, dtype=torch.float32)
                / 255.0
            )
            data["left_wrist_img"] = left
            joint_positions = obs["joint_positions"][:7]

        elif self.arm == "both":
            right = torch.from_numpy(obs["right_wrist_img"][None, ...])
            right = right[..., [2, 1, 0]]
            right = (
                right.permute(0, 3, 1, 2)
                .to(self.device, dtype=torch.float32)
                / 255.0
            )
            left = torch.from_numpy(obs["left_wrist_img"][None, ...])
            left = left[..., [2, 1, 0]]
            left = (
                left.permute(0, 3, 1, 2)
                .to(self.device, dtype=torch.float32)
                / 255.0
            )
            data["right_wrist_img"] = right
            data["left_wrist_img"] = left
            joint_positions = obs["joint_positions"]

        data["joint_positions"] = torch.from_numpy(joint_positions).reshape(1, 1, -1)
        data["embodiment"] = torch.tensor([self.embodiment_id], dtype=torch.int64)

        if not self.cartesian:
            data["actions_joints"] = torch.zeros_like(data["joint_positions"])
        else:
            data["actions_cartesian"] = torch.zeros_like(data["joint_positions"])

        processed_batch = {self.embodiment_id: data}

        # move non-image tensors to device and float32 (images already are)
        for key, val in data.items():
            if key not in ("front_img_1", "right_wrist_img", "left_wrist_img"):
                data[key] = val.to(self.device, dtype=torch.float32)

        processed_batch[self.embodiment_id] = (
            self.policy.model.data_schematic.normalize_data(
            processed_batch[self.embodiment_id], self.embodiment_id
            )
        )

        return processed_batch



def main(
    arms,
    frequency,
    cartesian,
    query_frequency=None,
    policy_path=None,
    dataset_path=None,
    repo_id=None,
    episodes=[1],
    debug=False,
):
    ri = None
    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]
    ri = ARXInterface(arms=arms_list)

    if policy_path is None and dataset_path is not None and repo_id is not None:
        rollout_type = "replay_lerobot"
        policy = ReplayRolloutLerobot(
            dataset_path=dataset_path,
            repo_id=repo_id,
            cartesian=cartesian,
            extrinsics_key="x5Nov18_3",
            episodes=episodes,
            arm=arms,
        )
    elif policy_path is not None:
        rollout_type = "policy"
        policy = PolicyRollout(
            arm=arms,
            policy_path=policy_path,
            query_frequency=query_frequency,
            cartesian=cartesian,
            extrinsics_key="x5Nov18_3",
        )
    elif dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)

    print(f"Cartesian value {cartesian}")
    ri.set_home()

    camera_transforms = CameraTransforms(
        intrinsics_key="base", extrinsics_key="x5Nov18_3"
    )
    kinematics_solver = EvaMinkKinematicsSolver(
        model_path="/home/robot/robot_ws/egomimic/resources/model_x5.xml"
    )
    try:
        with RateLoop(frequency=frequency, verbose=True) as loop:
            for i in loop:
                actions = None
                if rollout_type == "policy":
                    obs = ri.get_obs()
                    actions = policy.rollout_step(i, obs)
                elif rollout_type == "replay":
                    actions = policy.rollout_step(i)
                elif rollout_type == "replay_lerobot":
                    actions = policy.rollout_step(i)
                else:
                    raise ValueError(f"Invalid rollout type: {rollout_type}")

                if actions is None:
                    print("Finish rollout")
                    break

                if debug and rollout_type == "policy":
                    os.makedirs("debug", exist_ok=True)
                    if isinstance(obs["front_img_1"], torch.Tensor):
                        if obs["front_img_1"].dim() == 4:  # (B, C, H, W)
                            img = obs["front_img_1"][0].permute(1, 2, 0).cpu().numpy()
                        elif obs["front_img_1"].dim() == 3:  # (C, H, W)
                            img = obs["front_img_1"].permute(1, 2, 0).cpu().numpy()
                        else:  # (H, W, C)
                            img = obs["front_img_1"].cpu().numpy()
                    else:
                        img = obs["front_img_1"]
                        if img.ndim == 3 and img.shape[0] == 3:  # (C, H, W)
                            img = img.transpose(1, 2, 0)
                    img = img.astype(np.uint8)

                    for arm in arms_list:
                        arm_offset = 0
                        if arm == "right":
                            arm_offset = 7
                        arm_action = actions[arm_offset : arm_offset + 7]

                        if cartesian:
                            action_xyz = policy.debug_actions[:, :3]
                        else:
                            jnts = policy.actions[:, :7]
                            actions_xyz = np.zeros((jnts.shape[0], 3))
                            for i in range(action_xyz.shape[0]):
                                pos, rot = kinematics_solver.fk(jnts[i][:6])
                                actions_xyz[i] = pos
                            # TODO fk later

                        im_viz = visualize_actions(
                            img,
                            action_xyz,
                            camera_transforms.extrinsics,
                            camera_transforms.intrinsics,
                            arm=arm,
                        )
                        cv2.imwrite(
                            f"debug/debug_{arm}_{i}.png",
                            cv2.cvtColor(im_viz, cv2.COLOR_RGB2BGR),
                        )

                for arm in arms_list:
                    arm_offset = 0
                    if arm == "right":
                        arm_offset = 7
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
    parser.add_argument("--repo-id", type=str, help="repo id for replay")
    parser.add_argument("--episodes", type=int, nargs="+", help="episodes to replay")
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="control in cartesian space instead of joint space",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug visualization of actions on images",
    )

    args = parser.parse_args()

    main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        repo_id=args.repo_id,
        episodes=args.episodes,
        cartesian=args.cartesian,
        debug=args.debug,
    )
