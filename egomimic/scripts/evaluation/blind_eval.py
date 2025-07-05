from sre_constants import JUMP
import pandas as pd
import argparse
import numpy as np
import time
import os
import string
import random
from pathlib import Path

import gc
import torch
import robomimic.utils.obs_utils as ObsUtils
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

from egomimic.utils.realUtils import *

from omegaconf import DictConfig, OmegaConf
import hydra

CURR_INTRINSICS = ARIA_INTRINSICS
CURR_EXTRINSICS = EXTRINSICS["ariaJul29R"]
TEMPORAL_AGG = False

from rldb.utils import EMBODIMENT, get_embodiment, get_embodiment_id

from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval
from egomimic.scripts.evaluation.utils import TemporalAgg

from egomimic.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

from egomimic.scripts.evaluation.eval import Eval

class BlindEval(Eval):
    def __init__(
        self,
        **kwargs
    ):
        self.models = kwargs.get("models", None)
        self.blind_eval_name = kwargs.get("blind_eval_name", None)
        self.blind_eval_path = kwargs.get("blind_eval_path", None)
        self.rollout_time = kwargs.get("rollout_time", 1000)
        
        self.cur_ckpt = None
        self.cur_rollout_id = None
        self.cur_blind_id = None

        if self.models is not None and self.blind_eval_path is not None:
            raise ValueError("Only models or blind_eval_path should be specified")
        if self.models is None and self.blind_eval_path is None:
            raise ValueError("One of models or blind eval path needs to be specified")
        if (self.blind_eval_name is not None) ^ (self.blind_eval_path is not None):
            raise ValueError("Can only either use existing blind eval with (blind_eval_path) or create new blind eval (use blind_eval_name).")
        
        self.properties = ['ckpt_path', 'arm', 'frequency', 'cartesian']
        eval_dir = Path(self.eval_path).parent
        self.blind_eval_path = os.path.join(eval_dir, self.blind_eval_name)
        os.makedirs(self.blind_eval_path, exist_ok=True)

        if self.models:
            rows = []
            num_models = len(self.models)
            letters = list(string.ascii_uppercase[:num_models])
            random.shuffle(letters)
            for i, (key, value) in enumerate(self.models.items()):
                row = [value[prop] for prop in self.properties] + [letters[i]]
                rows.append(row)
            self.properties.append('blind_id')
            self.model_df = pd.DataFrame(rows, columns=self.properties)
            self.model_df.to_pickle(os.path.join(self.blind_eval_path, 'info.pkl'))
            self.result_df = pd.DataFrame([], columns=['episode_id', 'blind_id', 'success'])
            self.result_df.to_csv(os.path.join(self.blind_eval_path, 'results.csv'))
            breakpoint()
        else:
            self.df = pd.read_pickle(os.path.join(self.blind_eval_path, 'info.pkl'))
            self.result_df = pd.read_csv(os.path.join(self.blind_eval_path, 'results.csv')) 
    
    def ckpt_from_blind_id(self, blind_id):
        row = self.model_df[self.model_df['blind_id'] == blind_id]
        return row['ckpt_path'].iloc[0]
    
    def blind_ids(self):
        return self.model_df["blind_id"].unique()

    def select_models_episode(self):
        options_list = self.df['name']
        blind_id_valid = False
        while not blind_id_valid:
            blind_id = input('Choose models to eval' + ', '.join(options_list))
        if blind_id not in self.model_df.columns.unique():
            print("Invalid model inputted")
        else:
            blind_id_valid = True
        self.cur_blind_id = blind_id
        ckpt = self.ckpt_from_blind_id(blind_id)
        if ckpt != self.cur_ckpt:
            if self.model is not None:
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("New cuda memory being cleaned: " + torch.cuda.memory_allocated())

            self.model = ModelWrapper.load_from_checkpoint(ckpt)
            self.cur_ckpt = ckpt
        
        rollout_id_valid = False
        added_string = f'prev id was {self.cur_rollout_id}' if self.cur_rollout_id else ''
        while not rollout_id_valid:
            rollout_id = input(f'Choose episode id {added_string}: ')
            if rollout_id.isdigit() and int(rollout_id) > 0:
                self.cur_rollout_id = int(rollout_id)
        
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
    
    def run_eval(self):
        aloha_fk = AlohaFK()
        qpos_t, actions_t = [], []

        if TEMPORAL_AGG:
            TA = TemporalAgg()

        ts = self.env.reset()
        t0 = time.time()

        while True:
            keep_rollout = True
            self.select_models_episode()
            print("Can interrupt rollout with any keyboard keys")
            with torch.inference_mode():
                rollout_images = []
                try:
                    for t in range(self.rollout_time):
                        time.sleep(max(0, DT*2 - (time.time() - t0)))
                        t0 = time.time()
                        obs = ts.observation
                        inference_t = time.time()

                        if t % self.query_frequency == 0:
                            batch = self.process_batch_for_eval(obs)
                            preds = self.model.model.forward_eval(batch)
                            
                            ac_key = self.model.model.ac_keys[self.embodiment_id]
                            actions = preds[f"{self.embodiment_name}_{ac_key}"].cpu().numpy()

                            if TEMPORAL_AGG:
                                TA.add_action(actions[0])
                                actions = TA.smoothed_action()[None, :]

                            print(f"Inference time: {time.time() - inference_t}")

                        raw_action = actions[:, t % self.query_frequency]
                        raw_action = raw_action[0]
                        target_qpos = raw_action

                        if self.arm == "right":
                            target_qpos = np.concatenate([np.zeros(7), target_qpos])
                        
                        ts = self.env.step(target_qpos)
                        qpos_t.append(ts.observation["qpos"])
                        actions_t.append(target_qpos)
                except KeyboardInterrupt:
                    cancel_type_valid = False
                    while not cancel_type_valid:
                        cancel_type = input("keyboard interrupted answer k to keep run and t to throw away run")
                        if cancel_type == 'k':
                            keep_rollout = True
                            cancel_type_valid = True
                        elif cancel_type == 't':
                            keep_rollout = False
                            cancel_type_valid = True
                        else:
                            print("Enter a valid input\n")
                        
            if keep_rollout:
                score = input("enter the score: ")
                row = {'blind_id': self.cur_blind_id, 'rollout_id': self.cur_rollout_id, 'score': score}
                self.result_df = pd.concat([self.result_df, pd.DataFrame(row)], ignore_index=True)

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
        
            
        