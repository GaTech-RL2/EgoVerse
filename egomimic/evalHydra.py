from typing import Any, Dict, List, Optional, Tuple

import os
import copy

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer


from omegaconf import DictConfig, OmegaConf

from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper

from egomimic.pl_utils.pl_model import ModelWrapper

from egomimic.scripts.evaluation.eval import Eval

log = RankedLogger(__name__, rank_zero_only=True)

from rldb.utils import DataSchematic

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
            datasets_instance = cfg.multirun_cfg.data
        elif "eval_datasets" in cfg.datasets and cfg.datasets.eval_datasets is not None:
            log.info(f"Using specified yaml evaluation datasets")
            eval_datasets = cfg.datasets.data.eval_datasets
            datasets_target = cfg.datasets.data._target_
            datasets_instance = cfg.datasets.data
        elif "valid_datasets" in cfg.datasets and cfg.datasets.valid_datasets is not None:
            log.ingo(f"Using specified yaml validation datasets")
            eval_datasets = cfg.datasets.data.valid_datasets
            datasets_target = cfg.datasets.data._target_
            datasets_instance = cfg.datasets.data
        
        eval_datasets_dict = {}
        for dataset_name in eval_datasets:
            eval_datasets_dict[dataset_name] = hydra.utils.instantiate(
               eval_datasets[dataset_name]
            )
    
        log.info(f"Instantiating datamodule <{datasets_target}>")
        assert "MultiDataModuleWrapper" in datasets_target, "cfg.data._target_ must be 'MultiDataModuleWrapper'"
        datamodule: LightningDataModule = hydra.utils.instantiate(datasets_instance, valid_datasets=eval_datasets_dict)
    
        for dataset_name, dataset in datamodule.valid_datasets.items():
            log.info(f"Inferring shapes for dataset <{dataset_name}>")
            data_schematic.infer_shapes_from_batch(dataset[0])
            data_schematic.infer_norm_from_dataset(dataset)
    
    eval = hydra.utils.instantiate(cfg.eval)
    eval.datamodule = datamodule
    eval.data_schematic = data_schematic # unsure if this is necessary to pass in

    log.info("Starting evaluation!")
    eval.run_eval()
    return None, None

@hydra.main(version_base="1.3", config_path="./hydra_configs", config_name="eval.yaml")
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