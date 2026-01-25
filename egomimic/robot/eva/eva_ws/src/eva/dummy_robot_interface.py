import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from robot_interface import Robot_Interface
from egomimic.rldb.utils import S3RLDBDataset
from egomimic.rldb.utils import EMBODIMENT


keys_translation = {
    "base_0_rgb": "front_img_1",
    "right_wrist_0_rgb": "right_wrist_img",
    "left_wrist_0_rgb": "left_wrist_img",
}

def translate_if_necessary(key):
    if key in keys_translation:
        return keys_translation[key]
    return key
class dummyArxInterface(Robot_Interface):
    def __init__(self, arms, embodiment_id,dataset_path=None, data_schematic=None):
        # Skip Robot_Interface config loading to keep this fully local/offline.
        self.arms = arms
        self.recorders = {}
        self._joint_positions = {arm: np.zeros(7, dtype=np.float64) for arm in arms}
        self._ee_pose = {arm: np.zeros(7, dtype=np.float64) for arm in arms}
        self.dataset_path = dataset_path
        self.dataset = None
        self.embodiment_id = embodiment_id
        if self.dataset_path is not None:
            filters = {"episode_hash": "2025-11-27-03-39-50-378000"}
            self.dataset = S3RLDBDataset(embodiment="eva_right_arm", mode="total", filters=filters, cache_root='scratch/.cache')
            self.dataset_step = 0
        if data_schematic is not None:
            self.data_schematic = data_schematic
        else:
            self.data_schematic = None
            print("No data schematic provided")
            

    def _create_controllers(self, cfg):
        return None

    def set_joints(self, desired_position, arm):
        if desired_position.shape != (7,):
            raise ValueError(
                "For Eva, desired position must be of shape (7,) for single arm"
            )
        self._joint_positions[arm] = desired_position.astype(np.float64)

    def set_pose(self, pose, arm):
        if pose.shape != (7,):
            raise ValueError(
                f"For Eva, target position must be of shape (7,), current shape: {pose.shape}"
            )
        self._ee_pose[arm] = pose.astype(np.float64)
        joints = np.zeros(7, dtype=np.float64)
        joints[6] = pose[6]
        self._joint_positions[arm] = joints
        return joints

    def get_obs(self):
        if self.dataset is not None:
            data = self.dataset[self.dataset_step] #TODO from dataschematic instead of hardcoding
            obs = {}
            for key in self.data_schematic.keys_of_type("camera_keys"):
                lerobot_key = self.data_schematic.keyname_to_lerobot_key(key, self.embodiment_id)
                if lerobot_key is not None:
                    image = data[lerobot_key]
                    if image.ndim == 4:
                        image = (image.permute(0,2, 3, 1).squeeze().cpu().numpy() * 255.0).astype(np.uint8)
                        image = image[..., [2, 1, 0]]
                    else:
                        image = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                        image = image[..., [2, 1, 0]]
                    obs[translate_if_necessary(key)] = image
            for key in self.data_schematic.keys_of_type("proprio_keys"):
                lerobot_key = self.data_schematic.keyname_to_lerobot_key(key, self.embodiment_id)
                if lerobot_key is not None:
                    proprio = data[lerobot_key].cpu().numpy()
                    if key == "joint_positions":
                        if self.embodiment_id == EMBODIMENT.EVA_RIGHT_ARM.value:
                            zero_proprio = np.zeros(14)
                            zero_proprio[7:] = proprio
                            proprio = zero_proprio
                        elif self.embodiment_id == EMBODIMENT.EVA_LEFT_ARM.value:
                            zero_proprio = np.zeros(14)
                            zero_proprio[:7] = proprio
                            proprio = zero_proprio
                    obs[translate_if_necessary(key)] = proprio

            self.dataset_step += 1
            return obs
        else:
            obs = {}
            joint_positions = np.zeros(14, dtype=np.float64)
            ee_poses = np.zeros(14, dtype=np.float64)
            for arm in self.arms:
                arm_offset = 0
                if arm == "right":
                    arm_offset = 7
                joint_positions[arm_offset : arm_offset + 7] = self.get_joints(arm)
                xyz, rot = self.get_pose(arm, se3=False)
                ee_poses[arm_offset : arm_offset + 7] = np.concatenate(
                    [xyz, rot.as_euler("ZYX", degrees=False), [joint_positions[arm_offset + 6]]]
                )
            obs["joint_positions"] = joint_positions
            obs["ee_poses"] = ee_poses
            return obs

    def solve_ik(self, ee_pose, arm):
        if ee_pose.shape != (6,):
            raise ValueError(
                "For Eva, target position must be of shape (6,) for single arm"
            )
        return np.zeros(6, dtype=np.float64)

    def get_joints(self, arm):
        return self._joint_positions[arm].copy()

    def get_pose(self, arm, se3=False):
        pose = self._ee_pose[arm]
        pos = pose[:3].copy()
        rot = R.from_euler("ZYX", pose[3:6], degrees=False)
        if se3:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = pos
            return T
        return pos, rot

    def set_home(self):
        for arm in self.arms:
            self._joint_positions[arm] = np.zeros(7, dtype=np.float64)
            self._ee_pose[arm] = np.zeros(7, dtype=np.float64)