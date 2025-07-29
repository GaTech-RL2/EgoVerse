from typing import Any, Dict, List, Optional, Tuple

import os
import time
import itertools
import copy
import math
import numpy as np

import hydra
import torch
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer

from omegaconf import DictConfig, OmegaConf

from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper
from egomimic.utils.egomimicUtils import *

from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval

from egomimic.scripts.evaluation.utils import TemporalAgg, save_image, transformation_matrix_to_pose, batched_euler_to_rot_matrix

from egomimic.scripts.evaluation.test2 import cam_cartesian_to_base_quat, rollout_ee_pose_offline, init_pybullet, print_joint_info

log = RankedLogger(__name__, rank_zero_only=True)

from rldb.utils import *

@task_wrapper
def eval(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating data schematic <{cfg.multirun_cfg.data_schematic._target_}>")
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.multirun_cfg.data_schematic)
    datamodule = None
    if cfg.datasets is not None:
        
        if cfg.datasets == "multirun":
            log.info(f"Using multirun validation datasets")
            eval_datasets = cfg.multirun_cfg.data.valid_datasets
            datasets_target = cfg.multirun_cfg.data._target_
        elif "eval_datasets" in cfg.datasets and cfg.datasets.eval_datasets is not None:
            log.info(f"Using specified yaml evaluation datasets")
            eval_datasets = cfg.datasets.data.eval_datasets
            datasets_target = cfg.datasets.data._target_
        elif "valid_datasets" in cfg.datasets and cfg.datasets.valid_datasets is not None:
            log.ingo(f"Using specified yaml validation datasets")
            eval_datasets = cfg.datasets.data.valid_datasets
            datasets_target = cfg.datasets.data._target_
        
        eval_datasets_dict = {}
        for dataset_name in eval_datasets:
            eval_datasets[dataset_name] = hydra.utils.instantiate(
                eval_datasets_dict[dataset_name]
            )
    
        log.info(f"Instantiating datamodule <{datasets_target}>")
        assert "MultiDataModuleWrapper" in datasets_target, "cfg.data._target_ must be 'MultiDataModuleWrapper'"
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, valid_datasets=eval_datasets_dict)
    
        for dataset_name, dataset in datamodule.valid_datasets.items():
            log.info(f"Inferring shapes for dataset <{dataset_name}>")
            data_schematic.infer_shapes_from_batch(dataset[0])
            data_schematic.infer_norm_from_dataset(dataset)
    
    
    repo_id = "rpuns/test"
    
    urdf_path = os.path.join(
            os.path.dirname(egomimic.__file__), "resources/aloha_vx300s.urdf"
        )   
    robot_id = init_pybullet(urdf_path)
    print_joint_info(robot_id)
    
    eval = hydra.utils.instantiate(cfg.eval)
    eval.datamodule = datamodule
    eval.data_schematic = data_schematic # unsure if this is necessary to pass in
    dataset_path = '/home/rl2-eve/EgoVerse/logs/lerobot/lerobot'
    repo_id = "rpuns/test"
    episodes = [0]  
    dataset = RLDBDataset(repo_id=repo_id, root=dataset_path, local_files_only=True, episodes=episodes, mode="sample")
    joint_positions = torch.stack([
            sample["actions_joints"][0] for sample in dataset
    ])
    cartesians = torch.stack([
        sample["actions_cartesian"][0] for sample in dataset
    ])
    extrinsics = EXTRINSICS['ariaJun7']
    aloha_fk = AlohaFK()
    left_out = aloha_fk.fk(joint_positions[..., :6]).numpy()
    right_out = aloha_fk.fk(joint_positions[..., 7:13]).numpy()
    # right_fk_ee_pose = np.zeros((right_out.shape[0], 7))
    # left_fk_ee_pose = np.zeros((left_out.shape[0], 7))
    # for i, r in enumerate(right_out):
    #     right_fk_ee_pose[i] = transformation_matrix_to_pose(r)
    # for i, l in enumerate(left_out):
    #     left_fk_ee_pose[i] = transformation_matrix_to_pose(l)
    
    cartesians_left = cartesians[..., :6]
    cartesians_right = cartesians[..., 7:]
    
    left_fk_ee_pose = cam_cartesian_to_base_quat(cartesians_left, extrinsics['left'])
    right_fk_ee_pose = cam_cartesian_to_base_quat(cartesians_right, extrinsics['right'])

    joint_positions_reconstructed = rollout_ee_pose_offline(robot_id, left_fk_ee_pose, right_fk_ee_pose, joint_positions)
    ###
    breakpoint()
    joint_positions_reconstructed = np.stack(joint_positions_reconstructed).astype(np.float32)
    right_reconstructed_ee_pose = aloha_fk.fk(joint_positions_reconstructed[..., 7:13])
    left_reconstructed_ee_pose = aloha_fk.fk(joint_positions_reconstructed[..., :6])

    tensor_left_out = torch.from_numpy(left_out).float()
    tensor_right_out = torch.from_numpy(right_out).float()

    left_diff = tensor_left_out - left_reconstructed_ee_pose
    right_diff = tensor_right_out - right_reconstructed_ee_pose

    print(f'Left diff mean: {left_diff.mean(dim=0)}')
    print(f'Left diff std: {left_diff.std(dim=0)}')
    print(f'Right diff mean: {right_diff.mean(dim=0)}')
    print(f'Right diff std: {right_diff.std(dim=0)}')

    p.disconnect()
    
@hydra.main(version_base="1.3", config_path="../../hydra_configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    
    extras(cfg)

    # if 'multirun_path' not in cfg:
    #     raise ValueError("Multirun path is required.")
    # if not os.path.exists(cfg.multirun_path):
    #     raise FileNotFoundError(f"Cannot locate multirun.yaml at {cfg.multirun_path}")
    if 'multirun_cfg' in cfg:
        multi_cfg = OmegaConf.load(cfg.multirun_cfg)
        OmegaConf.set_struct(cfg, False)
        cfg["multirun_cfg"] = copy.deepcopy(multi_cfg)
        OmegaConf.set_struct(cfg, True)
    
    print(OmegaConf.to_yaml(cfg))
    
    eval(cfg)

if __name__ == '__main__':
    main()