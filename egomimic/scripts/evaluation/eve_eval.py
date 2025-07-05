import argparse
import numpy as np
import time
import os

import torch
from torchvision.utils import save_image
import cv2
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

from eve.constants import DT, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE

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
)

from eve.robot_utils import move_grippers, move_arms  # requires EgoMimic-eve
from eve.real_env import make_real_env  # requires EgoMimic-eve

# from egomimic.utils.realUtils import *

from eve.constants import DT, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE

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
)

from omegaconf import DictConfig, OmegaConf
import hydra

CURR_INTRINSICS = ARIA_INTRINSICS
CURR_EXTRINSICS = EXTRINSICS["ariaJun7"]
TEMPORAL_AGG = False

from rldb.utils import EMBODIMENT, get_embodiment, get_embodiment_id

from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval
from egomimic.scripts.evaluation.utils import TemporalAgg, save_image, transformation_matrix_to_pose, batched_euler_to_rot_matrix

from egomimic.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

class EveEval(Eval):
    def __init__(
        self,
        eval_path,
        ckpt_path,
        arm,
        query_frequency,
        num_rollouts,
        debug=False,
        **kwargs
    ):
        super().__init__(eval_path, **kwargs)

        log.info(f"Instantiating model from checkpoint<{ckpt_path}>")
        self.model = ModelWrapper.load_from_checkpoint(ckpt_path)
        self.arm = arm
        self.query_frequency = query_frequency
        self.num_rollouts = num_rollouts
        self.debug = debug
        self.cartesian = True
        
        self.plot_pred_freq = kwargs.get("plot_pred_freq", None)
        # if self.data_schematic is None:
        #     raise ValueError("Data schematic is needed to be passed in.")
        self.model.eval()
        node = create_interbotix_global_node('aloha')
        self.env = make_real_env(node, active_arms=self.arm, setup_robots=True)

        robot_startup(node)

        if not os.path.exists(self.eval_path) and self.debug:
            os.makedirs(self.eval_path)

        if self.arm == "right":
            self.embodiment_name = "eve_right_arm"
            self.embodiment_id = get_embodiment_id(self.embodiment_name)
        elif self.arm == "left":
            self.embodiment_name = "eve_left_arm"
            self.embodiment_id = get_embodiment_id(self.embodiment_name)
        elif self.arm == "both":
            self.embodiment_name = "eve_bimanual"
            self.embodiment_id = get_embodiment_id(self.embodiment_name)
        else:
            raise ValueError("Invalid arm inputted")

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
        data["actions_joints"] = torch.zeros_like(data["joint_positions"])
        processed_batch[self.embodiment_id] = data
        for key, val in data.items():
            data[key] = val.to(self.device)
        processed_batch[self.embodiment_id] = self.model.model.data_schematic.normalize_data(processed_batch[self.embodiment_id], self.embodiment_id)
    
        return processed_batch
    
    def solve_ik(self, actions_cartesian, current_joint_positions, arm="right"):
        """
        for each arm, right or left for extrinsics (works for batch and unbatched)
        """
        breakpoint()
        actions_cartesian = actions_cartesian
        actions_gripper = actions_cartesian[..., -1]
        actions_pos = actions_cartesian[..., :3]
        actions_ypr = actions_cartesian[..., 3:6]
        actions_rotmat = batched_euler_to_rot_matrix(actions_ypr)
        extrinsics = torch.from_numpy(self.model.model.camera_transforms.extrinsics[arm]).to(actions_cartesian.device).float()
        batch_shape = actions_cartesian.shape[:-1]
        T = torch.zeros(*batch_shape, 4, 4, device=actions_cartesian.device)
        T[..., :3, :3] = actions_rotmat
        T[..., :3, 3] = actions_pos
        T[..., 3, 3] = 1.0

        T_base = extrinsics @ T
        
        # actions_rotmat = T_base[..., :3, :3]
        # wxyz -> xyzw

        # actions_quat = matrix_to_quaternion(actions_rotmat)
        # actions_quat = torch.cat([actions_quat[..., 1:], actions_quat[..., :1]], dim=-1)

        actions = []
        for b in range(T_base.shape[0]):
            actions.append(transformation_matrix_to_pose(T_base[b].cpu().numpy()))

        actions = np.stack(actions)

        actions_pos = actions[..., :3]
        actions_quat = actions[..., 3:]
        target_joint_positions = self.ik.solve(target_pos=actions_pos, 
                                               target_orientation=actions_quat, 
                                               target_gripper=actions_gripper.cpu().numpy(),
                                               current_joints=current_joint_positions)
        
        return target_joint_positions[None, :, :]
    
    def run_eval(self):
        self.device = torch.device("cuda")
        self.model.to(self.device)
        aloha_fk = AlohaFK()
        qpos_t, actions_t = [], []

        if TEMPORAL_AGG:
            TA = TemporalAgg()

        ts = self.env.reset()
        t0 = time.time()

        for rollout_id in range(self.num_rollouts):
            with torch.inference_mode():
                rollout_images = []
                for t in range(1000):
                    time.sleep(max(0, DT*2 - (time.time() - t0)))
                    t0 = time.time()
                    obs = ts.observation
                    inference_t = time.time()

                    if t % self.query_frequency == 0:
                        batch = self.process_batch_for_eval(obs)
                        preds = self.model.model.forward_eval(batch)
                        
                        ac_key = self.model.model.ac_keys[self.embodiment_id]
                        actions = preds[f"{self.embodiment_name}_{ac_key}"].cpu().numpy()
                        if self.cartesian:
                            if self.arm == "right":
                                qpos = torch.from_numpy(np.array(obs["qpos"])).float().to(self.device)
                                current_joint_positions = qpos[7:]
                                actions = self.solve_ik(actions, current_joint_positions, "right")
                            elif self.arm == "left":
                                qpos = torch.from_numpy(np.array(obs["qpos"])).float().to(self.device)
                                current_joint_positions = qpos[:7]
                                actions = self.solve_ik(actions, current_joint_positions, "left")
                            else:
                                qpos = torch.from_numpy(np.array(obs["qpos"])).float().to(self.device)
                                actions_left = actions[..., :7]
                                actions_right = actions[..., 7:]
                                current_joint_left = qpos[:7]
                                current_joint_right = qpos[7:]
                                solved_left = self.solve_ik(actions_left, current_joint_left, "left")
                                solved_right = self.solve_ik(actions_right, current_joint_right, "right")
                                actions = np.concatenate([solved_left, solved_right], axis=-1)

                        if TEMPORAL_AGG:
                            TA.add_action(actions[0])
                            actions = TA.smoothed_action()[None, :]

                        print(f"Inference time: {time.time() - inference_t}")
                        if self.debug:
                            breakpoint()
                            data_dict = batch[self.embodiment_id]
                            im = data_dict['front_img_1'].squeeze(0).permute(1, 2, 0).cpu().numpy()
                            if im.dtype != np.uint8:
                                im = (im * 255).astype(np.uint8)
                            pred_type = "joints"
                            color = "Purples"
                            viz_actions = preds[f'{self.embodiment_name}_actions_joints'].squeeze(0).cpu().numpy()
                            viz_actions = viz_actions[:100, :]
                            extrinsics = self.model.model.camera_transforms.extrinsics
                            intrinsics = self.model.model.camera_transforms.intrinsics
                            drawn_im = draw_actions(im, pred_type, color, viz_actions, extrinsics, intrinsics, self.arm)
                            save_image(drawn_im, os.path.join(self.eval_path, f'image_{t}.png'))

                    raw_action = actions[:, t % self.query_frequency]
                    raw_action = raw_action[0]
                    target_qpos = raw_action

                    if self.arm == "right":
                        target_qpos = np.concatenate([np.zeros(7), target_qpos])
                    
                    ts = self.env.step(target_qpos)
                    qpos_t.append(ts.observation["qpos"])
                    actions_t.append(target_qpos)

            log.info("Moving Robot")

            if self.arm == "right":
                move_grippers(
                [self.env.follower_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN], moving_time=0.5
                )  # open
                move_arms([self.env.follower_bot_right], [START_ARM_POSE[:6]], moving_time=1.0)
            elif self.arm == "left":
                move_grippers(
                [self.env.follower_bot_left], [FOLLOWER_GRIPPER_JOINT_OPEN], moving_time=0.5
                ) 
                move_arms([self.env.follower_bot_left], [START_ARM_POSE[:6]], moving_time=1.0)
            elif self.arm == "both":
                move_grippers(
                    [self.env.follower_bot_left, self.env.follower_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN]*2, moving_time=0.5
                )  # open
                move_arms([self.env.follower_bot_left, self.env.follower_bot_right], [START_ARM_POSE[:6]]*2, moving_time=1.0)
        return
        
