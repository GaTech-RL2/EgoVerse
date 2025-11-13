import argparse
import numpy as np
import time
import os

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

from egomimic.rldb.utils import EMBODIMENT, get_embodiment, get_embodiment_id

from egomimic.evaluation.eval import Eval

from egomimic.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TemporalAgg:
    def __init__(self):
        self.recent_actions = []

    def add_action(self, action):
        """
        actions: (100, 7) tensor
        """
        self.recent_actions.append(action)
        if len(self.recent_actions) > 4:
            del self.recent_actions[0]

    def smoothed_action(self):
        """
        returns smooth action (100, 7)
        """
        mask = []
        count = 0

        shifted_actions = []
        # breakpoint()

        for ac in self.recent_actions[::-1]:
            basic_mask = np.zeros(100)
            basic_mask[: 100 - count] = 1
            mask.append(basic_mask)
            shifted_ac = ac[count:]
            shifted_ac = np.concatenate([shifted_ac, np.zeros((count, 7))], axis=0)
            shifted_actions.append(shifted_ac)
            count += 25

        mask = mask[::-1]
        mask = ~(np.array(mask).astype(bool))
        recent_actions = shifted_actions[::-1]
        recent_actions = np.array(recent_actions)
        # breakpoint()
        mask = np.repeat(mask[:, :, None], 7, axis=2)
        smoothed_action = np.ma.array(recent_actions, mask=mask).mean(axis=0)

        # PLOT_JOINT = 0
        # for i in range(recent_actions.shape[0]):
        #     plt.plot(recent_actions[i, :, PLOT_JOINT], label=f"index{i}")
        # plt.plot(smoothed_action[:, PLOT_JOINT], label="smooth")
        # plt.legend()
        # plt.savefig("smoothing.png")
        # plt.close()
        # breakpoint()

        return smoothed_action


class Eve(Eval):
    def __init__(self, config, ckpt_path, mode, arm, eval_path, debug=True):
        super().__init__(config, ckpt_path)
        self.mode = mode
        self.arm = arm

        model: LightningModule = hydra.utils.instantiate(cfg.model)
        model.load_from_checkpoint(self.ckpt_path)

        self.model = model
        self.model.eval()

        log.info("Instantiated model!")

        node = create_interbotix_global_node("aloha")
        self.env = make_real_env(node, active_arms=self.arm, setup_robots=True)

        self.data_schematic = self.model.model.data_schematic

        robot_startup(node)

        rollout_dir = os.path.dirname(os.path.dirname(eval_path))
        self.rollout_dir = os.path.join(rollout_dir, "rollouts")

        if not os.path.exists(rollout_dir):
            os.mkdir(rollout_dir)

        if not debug:
            self.rollout_dir = None

    def process_batch_for_eval(self, batch):
        obs = batch
        processed_batch = {}
        qpos = np.array(obs["qpos"])
        qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = {
            "right_wrist_img": (
                torch.from_numpy(obs["images"]["cam_right_wrist"][None, None, :])
            ).to(torch.uint8)
            / 255.0,
            "front_img_1": (
                torch.from_numpy(obs["images"]["cam_high"][None, None, :])
            ).to(torch.uint8)
            / 255.0,
            "pad_mask": torch.ones((1, 100, 1)).to(device).bool(),
            "joint_positions": qpos[..., 7:].reshape((1, 1, -1)),
        }

        if self.arm == "right":
            data["joint_positions"] = qpos[..., 7:].reshape((1, 1, -1))
            embodiment_id = get_embodiment_id("eve_right_arm")
            data["embodiment"] = torch.Tensor([embodiment_id], dtype=torch.int64)
            processed_batch[embodiment_id] = data
            processed_batch = self.model.model.data_schematic.normalize_data(
                processed_batch, embodiment_id
            )

        elif self.arm == "both":
            data["left_wrist_img"] = (
                torch.from_numpy(obs["images"]["cam_left_wrist"][None, None, :]).to(
                    torch.uint8
                )
                / 255.0
            )
            data["joint_positions"] = qpos[..., :].reshape((1, 1, -1))
            embodiment_id = get_embodiment_id("eve_bimanual")
            data["embodiment"] = torch.Tensor([embodiment_id], dtype=torch.int64)
            processed_batch[embodiment_id] = data
            processed_batch = self.model.model.data_schematic.normalize_data(
                processed_batch, embodiment_id
            )

        return processed_batch

    def perfom_eval(self):
        """ """
        if self.mode == "real":
            self.eval_real(self.model.model, self.env, self.rollout_dir, self.arm)
        else:
            raise ValueError("Sim evaluation is not set up for this robot")

    def eval_real(self, model, env, rollout_dir, arm):
        device = torch.device("cuda")
        aloha_fk = AlohaFK()
        query_frequency = 25

        qpos_t, actions_t = [], []
        num_rollouts = 50

        for rollout_id in range(num_rollouts):
            if TEMPORAL_AGG:
                TA = TemporalAgg()

            ts = env.reset()
            t0 = time.time()

            with torch.inference_mode():
                rollout_images = []
                for t in range(1000):
                    time.sleep(max(0, DT * 2 - (time.time() - t0)))
                    t0 = time.time()
                    obs = ts.observation
                    inference_t = time.time()

                    if t % query_frequency == 0:
                        batch = self.process_batch_for_eval(obs)
                        preds = self.model.model.forward_eval(batch)

                        if self.arm == "right":
                            embodiment_name = "eve_right_arm"
                            embodiment_id = get_embodiment_id(embodiment_name)
                        else:
                            embodiment_name = "eve_bimanual"
                            embodiment_id = get_embodiment_id(embodiment_name)
                        ac_key = self.model.model.ac_keys[embodiment_id]
                        actions = preds[f"{embodiment_name}_{ac_key}"].cpu().numpy()

                        if TEMPORAL_AGG:
                            TA.add_action(actions[0])
                            actions = TA.smoothed_action()[None, :]

                        print(f"Inference time: {time.time() - inference_t}")

                    raw_action = actions[:, t % query_frequency]
                    raw_action = raw_action[0]
                    target_qpos = raw_action

                    if self.arm == "right":
                        target_qpos = np.concatenate([np.zeros(7), target_qpos])

                    ts = env.step(target_qpos)
                    qpos_t.append(ts.observation["qpos"])
                    actions_t.append(target_qpos)

            rollout_images = []
            log.info("Moving Robot")

            if self.arm == "right":
                move_grippers(
                    [env.follower_bot_right],
                    [FOLLOWER_GRIPPER_JOINT_OPEN],
                    moving_time=0.5,
                )  # open
                move_arms(
                    [env.follower_bot_right], [START_ARM_POSE[:6]], moving_time=1.0
                )

            elif self.arm == "both":
                move_grippers(
                    [env.follower_bot_left, env.follower_bot_right],
                    [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
                    moving_time=0.5,
                )  # open
                move_arms(
                    [env.follower_bot_left, env.follower_bot_right],
                    [START_ARM_POSE[:6]] * 2,
                    moving_time=1.0,
                )

            time.sleep(12.0)
        return
