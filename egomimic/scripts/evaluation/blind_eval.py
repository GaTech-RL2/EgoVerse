from sre_constants import JUMP
import pandas as pd
import argparse
import numpy as np
import time
import os
import string
import random
from pathlib import Path
from multiprocessing import Process

import pickle

import gc
import torch
from torchvision.utils import save_image
import cv2
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
    TemporalAgg
)

from omegaconf import DictConfig, OmegaConf
import hydra

CURR_INTRINSICS = ARIA_INTRINSICS
CURR_EXTRINSICS = EXTRINSICS["ariaJul29R"]
TEMPORAL_AGG = False

from rldb.utils import EMBODIMENT, get_embodiment, get_embodiment_id

from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval

from egomimic.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

from egomimic.scripts.evaluation.eval import Eval

class BlindEval(Eval):
    def __init__(
        self,
        eval_path,
        models,
        **kwargs
    ):
        super().__init__(eval_path, **kwargs)
        self.models = models
        
        self.datamodule = None
        self.data_schematic = None
        
        self.blind_eval_path = kwargs.get("blind_eval_path", None)
        
        self.cur_rollout_id = None
        self.cur_blind_id = None
        self.cur_eval = None

        if self.models is not None and self.blind_eval_path is not None:
            raise ValueError("Only models or blind_eval_path should be specified")
        if self.models is None and self.blind_eval_path is None:
            raise ValueError("One of models or blind eval path needs to be specified")
        
        self.properties = ['filepath', 'class_name', 'model_name', 'blind_id']
        if self.blind_eval_path is None:
            self.blind_eval_path = self.eval_path
        os.makedirs(self.blind_eval_path, exist_ok=True)
        
        if self.models:
            rows = []
            num_models = len(self.models)
            letters = list(string.ascii_uppercase[:num_models])
            random.shuffle(letters)
            for i, (key, value) in enumerate(self.models.items()):
                filepath = os.path.join(self.eval_path, f"{letters[i]}.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(self.models[key], f)
                row = [filepath, self.models[key].__class__.__name__, key, letters[i]]
                rows.append(row)
            self.model_df = pd.DataFrame(rows, columns=self.properties)
            self.model_df.to_pickle(os.path.join(self.blind_eval_path, 'info.pkl'))
            self.result_df = pd.DataFrame([], columns=['episode_id', 'blind_id', 'success'])
            self.result_df.to_csv(os.path.join(self.blind_eval_path, 'results.csv'))
        else:
            self.model_df = pd.read_pickle(os.path.join(self.blind_eval_path, 'info.pkl'))
            self.result_df = pd.read_csv(os.path.join(self.blind_eval_path, 'results.csv')) 
    
    def eval_from_blind_id(self, blind_id):
        row = self.model_df[self.model_df['blind_id'] == blind_id].iloc[0]
        filepath = row['filepath']
        with open(filepath, "rb") as f:
            eval_model = pickle.load(f)
        return eval_model
    
    def blind_ids(self):
        return self.model_df["blind_id"].unique()

    def select_models_episode(self):
        options_list = self.model_df['blind_id']
        blind_id_valid = False
        valid_ids = set(self.model_df['blind_id'].astype(str))
        while not blind_id_valid:
            blind_id = input('Choose models to eval ' + ', '.join(options_list) + ': ')
            if blind_id not in valid_ids:
                print("Invalid model inputted")
            else:
                blind_id_valid = True
        if self.cur_blind_id != blind_id:
            if self.cur_eval is not None:
                del self.cur_eval
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("New cuda memory being cleaned: " + str(torch.cuda.memory_allocated()))

        self.cur_eval = self.eval_from_blind_id(blind_id)
        self.cur_blind_id = blind_id
        rollout_id_valid = False
        added_string = f'prev id was {self.cur_rollout_id}' if self.cur_rollout_id else ''
        while not rollout_id_valid:
            rollout_id = input(f'Choose episode id {added_string}: ')
            if rollout_id.isdigit() and int(rollout_id) > 0:
                self.cur_rollout_id = int(rollout_id)
                rollout_id_valid = True
            else:
                print("Episode id must be a positive integer.")
    
    def run_eval(self):
        if TEMPORAL_AGG:
            TA = TemporalAgg()

        while True:
            self.select_models_episode()
            if hasattr(self.cur_eval, "datamodule") and self.cur_eval.datamodule is None:
                self.cur_eval.datamodule = self.datamodule
            proc = Process(target=self.cur_eval.run_eval())
            proc.start()
            print("Interrupt rollout with any keyboard keys") #needed to be added on the actual laptop
            while proc.is_alive() is None:
                # add the code the interrupt eval here
                continue

            success_value = input("Input the success value: ")
            new_row = pd.DataFrame(
                [[self.cur_rollout_id, self.cur_blind_id, success_value]],  # Double brackets for single row
                columns=['episode_id', 'blind_id', 'success']
            )
            self.result_df = pd.concat([self.result_df, new_row], ignore_index=True)
            status_valid = False
            while not status_valid:
                status = input("stop recording reply s, otherwise reply r: ")
                if status == "s" or status == "r":
                    status_valid = True
                else:
                    print("The replied is not valid, reply again")
            
            if status == "s":
                break
        self.result_df.to_csv(os.path.join(self.blind_eval_path, 'results.csv'))
        print("results has been saved")
        return
        
            
        