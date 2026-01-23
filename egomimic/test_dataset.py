import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

OmegaConf.register_new_resolver("eval", eval)

from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper, get_metric_value

from egomimic.scripts.evaluation.eval import Eval

import numpy as np

log = RankedLogger(__name__, rank_zero_only=True)

from egomimic.rldb.utils import DataSchematic


def test_dataset(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating data schematic <{cfg.data_schematic._target_}>")
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)
    
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )
    
    # valid_datasets = {}
    # for dataset_name in cfg.data.valid_datasets:
    #     valid_datasets[dataset_name] = hydra.utils.instantiate(
    #         cfg.data.valid_datasets[dataset_name]
    #     )
    
    # import pdb; pdb.set_trace()
    
    cprint("=========Finish=========", "green", attrs=["bold"])



@hydra.main(version_base="1.3", config_path="./hydra_configs", config_name="train.yaml")
def main(cfg: DictConfig):
    extras(cfg)
    
    test_dataset(cfg)


if __name__ == "__main__":
    main()