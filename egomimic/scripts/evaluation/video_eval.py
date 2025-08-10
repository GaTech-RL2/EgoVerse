import argparse
import numpy as np
import time
import os

import torch
from torchvision.utils import save_image
import torchvision.io as tvio
import cv2
# from interbotix_common_modules.common_robot.robot import (
#     create_interbotix_global_node,
#     robot_shutdown,
#     robot_startup,
# )

# from eve.constants import DT, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE

from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    general_unnorm,
    miniviewer,
    nds,
    ARIA_INTRINSICS,
    EXTRINSICS,
    ee_pose_to_cam_frame,
    AlohaFK,
    AlohaIK
)

# from eve.robot_utils import move_grippers, move_arms  # requires EgoMimic-eve
# from eve.real_env import make_real_env  # requires EgoMimic-eve

# # from egomimic.utils.realUtils import *

# from eve.constants import DT, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE

from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    general_unnorm,
    miniviewer,
    nds,
    ARIA_INTRINSICS,
    EXTRINSICS,
    ee_pose_to_cam_frame,
    AlohaFK,
    draw_actions,
    TemporalAgg,
    init_pybullet,
    cam_cartesian_to_base_quat,
    rollout_ee_pose_arm
)


from omegaconf import DictConfig, OmegaConf
import hydra

CURR_INTRINSICS = ARIA_INTRINSICS
CURR_EXTRINSICS = EXTRINSICS["ariaJun7"]
TEMPORAL_AGG = False

from rldb.utils import EMBODIMENT, get_embodiment, get_embodiment_id

import egomimic
from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval

from egomimic.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

extrinsics = EXTRINSICS['ariaJun7']
class VideoEval(Eval):
    def __init__(
        self,
        eval_path,
        ckpt_path,
        embodiment_name,
        display_arm,
        **kwargs
    ):
        super().__init__(eval_path)
        self.ckpt_path = ckpt_path
        log.info(f"Instantiating model from checkpoint<{ckpt_path}>")
        self.model = ModelWrapper.load_from_checkpoint(ckpt_path)
        self.embodiment_name = embodiment_name
        self.cartesian = True
        self.arm = display_arm

        self.ik = AlohaIK()
        
        self.plot_pred_freq = kwargs.get("plot_pred_freq", None)
        # if self.data_schematic is None:
        #     raise ValueError("Data schematic is needed to be passed in.")
        self.model.eval()
        # node = create_interbotix_global_node('aloha')
        # self.env = make_real_env(node, active_arms=self.arm, setup_robots=True)

        # robot_startup(node)

        self.embodiment_id = get_embodiment_id(self.embodiment_name)
        
        self.repo_id = "rpuns/test"
        
        self.urdf_path = os.path.join(
            os.path.dirname(egomimic.__file__), "resources/aloha_vx300s.urdf"
        )
        
        self.aloha_fk = AlohaFK()
        self.robot_id = init_pybullet(self.urdf_path)

    def batch_to_device(self, batch, device):
        for embodiment_id, batch_embodiment in batch.items():
            for key, val in batch_embodiment.items():
                if hasattr(val, "device"):
                    batch[embodiment_id][key] = val.to(device)
            
    def process_batch_for_eval(self, batch):
        obs = batch
        processed_batch = {}
        qpos = np.array(obs["qpos"])
        qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(self.device)

        data = {
            "front_img_1" : (
                torch.from_numpy(
                obs["images"]["cam_high"][None, :]
            )).permute(0,3,1,2).to(torch.uint8) / 255.0,
            "pad_mask": torch.ones((1, 100, 1)).to(self.device).bool(),

        }

        if self.arm == "right":
            data["right_wrist_img"] = torch.from_numpy(obs["images"]["cam_right_wrist"][None, :]).permute(0,3,1,2).to(torch.uint8) / 255.0
            data["joint_positions"] =  qpos[..., 7:].reshape((1, 1, -1))
        elif self.arm == "left":
            data["left_wrist_img"] = torch.from_numpy(obs["images"]["cam_left_wrist"][None, :]).permute(0,3,1,2).to(torch.uint8) / 255.0
            data["joint_positions"] = qpos[..., :7].reshape((1, 1, -1))
        elif self.arm == "both":
            data["right_wrist_img"] = torch.from_numpy(obs["images"]["cam_right_wrist"][None, :]).permute(0,3,1,2).to(torch.uint8) / 255.0
            data["left_wrist_img"] = torch.from_numpy(obs["images"]["cam_left_wrist"][None, :]).permute(0,3,1,2).to(torch.uint8) / 255.0
            data["joint_positions"] = qpos[..., :].reshape((1, 1, -1))
        data["embodiment"] = torch.tensor([self.embodiment_id], dtype=torch.int64)
        if not self.cartesian:
            data["actions_joints"] = torch.zeros_like(data["joint_positions"])
        else:
            data["actions_cartesian"] = torch.zeros_like(data["joint_positions"])
        processed_batch[self.embodiment_id] = data
        for key, val in data.items():
            data[key] = val.to(self.device)
        processed_batch[self.embodiment_id] = self.model.model.data_schematic.normalize_data(processed_batch[self.embodiment_id], self.embodiment_id)
    
        return processed_batch
    
    def solve_ik(self, cartesians, joint_positions, arm="right"):
        """
        for each arm, right or left for extrinsics (works for batch and unbatched)
        """
        cartesians = cartesians[..., :6]
        gripper = cartesians[..., [7]]
        fk_ee_pose = cam_cartesian_to_base_quat(cartesians, extrinsics[arm])
        
        joint_positions_reconstructed = rollout_ee_pose_arm(self.robot_id, fk_ee_pose, joint_positions)

        joint_positions_reconstructed = np.stack(joint_positions_reconstructed).astype(np.float32)
        joint_positions_reconstructed = np.concatenate([joint_positions_reconstructed, gripper], axis=-1)
        return joint_positions_reconstructed
        
    def run_eval(self):
        self.device = torch.device("cuda")
        self.model.to(self.device)
        start_time = time.time()
        if TEMPORAL_AGG:
            TA = TemporalAgg()
        imgs = []
        trainloader = self.datamodule.val_dataloader()
        for i, batch in enumerate(trainloader):
            with torch.inference_mode():
                proc_batch = self.model.model.process_batch_for_training(batch[0])
                self.batch_to_device(proc_batch, self.device)
                preds = self.model.model.forward_eval(proc_batch)
                ac_key = self.model.model.ac_keys[self.embodiment_id]
                actions = preds[f"{self.embodiment_name}_{ac_key}"]
                actions = actions.cpu().numpy()

                if TEMPORAL_AGG:
                    TA.add_action(actions[0])
                    actions = TA.smoothed_action()[None, :]
                data_dict = proc_batch[self.embodiment_id]
                cur_imgs = data_dict['front_img_1'].cpu().numpy()
                for j in range(cur_imgs.shape[0]):
                    im = cur_imgs[j]
                    im = np.transpose(im, (1, 2, 0))
                    if im.dtype != np.uint8:
                        im = (im * 255).astype(np.uint8)
                    pred_type = None
                    if not self.cartesian:
                        pred_type = "joints"
                    else:
                        pred_type = "xyz" # haven't test debug in cartesian mode
                    color = "Purples"
                    viz_actions = actions[j]
                    extrinsics = self.model.model.camera_transforms.extrinsics
                    intrinsics = self.model.model.camera_transforms.intrinsics
                    drawn_im = draw_actions(im, pred_type, color, viz_actions, extrinsics, intrinsics, self.arm)
                    imgs.append(torch.from_numpy(drawn_im))
        imgs = torch.stack(imgs)
        video_path = os.path.join(self.eval_path, "eval_video.mp4")
        tvio.write_video(video_path, imgs, fps=30, video_codec="h264")
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Took {elapsed:.2f} seconds")
        return
