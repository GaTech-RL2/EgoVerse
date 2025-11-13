import os
import time
import h5py
import torch
import sys
import numpy as np

from abc import ABC, abstractmethod

from egomimic.rldb.utils import EMBODIMENT, get_embodiment, get_embodiment_id
from egomimic.pl_utils.pl_model import ModelWrapper

from robot_utils import RateLoop

sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))
from robot_interface import *
# from stream_aria import AriaRecorder
# from stream_d405 import RealSenseRecorder

# Control parameters
DEFAULT_FREQUENCY = 30  # Hz
QUERY_FREQUENCY = 20

RIGHT_CAM_SERIAL = ""
LEFT_CAM_SERIAL = ""

EMBODIMENT_MAP = {
    "both": 2,
    "left": 1,
    "right": 0,
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
                self.actions = np.asarray(f["action"][...], dtype=np.float32)
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
    def __init__(self, arm, policy_path, query_frequency):
        super().__init__()
        self.arm = arm
        self.policy = ModelWrapper.load_from_checkpoint(policy_path)
        self.query_frequency = query_frequency
        self.embodiment_id = EMBODIMENT_MAP[self.arm]
        self.embodiment_name = get_embodiment(self.embodiment_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def rollout_step(self, i, obs):
        if i % self.query_frequency == 0:
            start_infer_t = time.time()
            batch = self.process_obs_for_policy(obs)
            preds = self.policy.model.forward_eval(batch)
            ac_key = self.policy.model.ac_keys[self.embodiment_id]
            actions = preds[f"{self.embodiment_name.lower()}_{ac_key}"]
            self.actions = actions.detach().cpu().numpy().squeeze()
            print(f"Inference time: {(time.time() - start_infer_t)}s")

        # TODO check gripper if we are using 0 to 0.08 or 0 to 1
        act_i = i % self.query_frequency
        return self.actions[act_i]

    def process_obs_for_policy(self, obs):
        data = {
            "front_img_1": (torch.from_numpy(obs["front_img_1"][None, :]))
            .permute(0, 3, 1, 2)
            .to(torch.uint8)
            / 255.0,
            "pad_mask": torch.ones((1, 100, 1)).to(self.device).bool(),
        }
        # check dim since it seems to take in a batch
        if self.arm == "right":
            data["right_wrist_img"] = (
                torch.from_numpy(obs["right_wrist_img"][None, :])
                .permute(0, 3, 1, 2)
                .to(torch.uint8)
                / 255.0
            )

        elif self.arm == "left":
            data["left_wrist_img"] = (
                torch.from_numpy(obs["left_wrist_img"][None, :])
                .permute(0, 3, 1, 2)
                .to(torch.uint8)
                / 255.0
            )

        elif self.arm == "both":
            data["right_wrist_img"] = (
                torch.from_numpy(obs["right_wrist_img"][None, :])
                .permute(0, 3, 1, 2)
                .to(torch.uint8)
                / 255.0
            )
            data["left_wrist_img"] = (
                torch.from_numpy(obs["left_wrist_img"][None, :])
                .permute(0, 3, 1, 2)
                .to(torch.uint8)
                / 255.0
            )
        data["joint_positions"] = torch.from_numpy(obs["joint_positions"]).reshape(
            (1, 1, -1)
        )  # arm joint logic already implemented in robot interface
        data["embodiment"] = torch.tensor([self.embodiment_id], dtype=torch.int64)
        if not self.cartesian:
            data["actions_joints"] = torch.zeros_like(data["joint_positions"])
        else:
            data["actions_cartesian"] = torch.zeros_like(data["joint_positions"])

        processed_batch = {}
        processed_batch[self.embodiment_id] = data
        for key, val in data.items():
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
):
    ri = None
    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]
    ri = ARXInterface(arms=arms_list)

    rollout_type = "replay" if policy_path is None else "policy"
    if rollout_type == "policy":
        policy = PolicyRollout(
            arm=arm, policy_path=policy_path, query_frequency=query_frequency
        )
    elif rollout_type == "replay":
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
    else:
        raise RuntimeError("Invalid rollout type")

    print(f"Cartesian value {cartesian}")
    ri.set_home()
    with RateLoop(frequency=frequency, verbose=True) as loop:
        for i in loop:
            actions = None
            if rollout_type == "policy":
                obs = ri.get_obs()
                actions = policy.rollout_step(i, obs)
            elif rollout_type == "replay":
                actions = policy.rollout_step(i)
            # not sure if I need throw exception here

            if actions is None:
                print("Finish rollout")
                break
            for arm in arms_list:
                arm_offset = 0
                if arm == "right":
                    arm_offset = 7
                arm_action = actions[arm_offset : arm_offset + 7]
                if cartesian:
                    ri.set_pose(arm_action, arm)
                else:
                    ri.set_joints(arm_action, arm)


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
        type=float,
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

    args = parser.parse_args()

    main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        cartesian=args.cartesian,
    )
