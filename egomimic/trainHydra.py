import copy
import os
import signal
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from tabulate import tabulate

from egomimic.eval.eval import Eval
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper

OmegaConf.register_new_resolver("eval", eval)
log = RankedLogger(__name__, rank_zero_only=True)


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and isinstance(model_cfg.robomimic_model, DictConfig)
        and "data_schematic" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.data_schematic = None
    return OmegaConf.create({"model": model_cfg})


def _log_dataset_frame_counts(train_datasets: dict, valid_datasets: dict) -> None:
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    for name, ds in valid_datasets.items():
        rows.append(("valid", name, len(ds)))
    if valid_datasets:
        rows.append(
            ("TOTAL", "(valid)", sum(len(ds) for ds in valid_datasets.values()))
        )
    table = tabulate(
        rows,
        headers=["Split", "Dataset", "Frames"],
        tablefmt="rounded_outline",
        intfmt=",",
    )
    log.info("Dataset frame counts:\n" + table)


def _propagate_data_schematic_to_datasets(data_schematic, datasets):
    """
    Set the shared data schematic on all top-level datasets.
    """
    split_datasets = datasets
    for dataset_name, dataset in split_datasets.items():
        if not isinstance(dataset, MultiDataset):
            raise ValueError(
                f"{dataset_name} is not a MultiDataset. All top level datasets in data config should be MultiDataset"
            )
        dataset.set_data_schematic(data_schematic)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()
    # log.info(f"Instantiating data schematic <{cfg.data_schematic._target_}>")

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    # Modify dataset configs to include `data_schematic` dynamically at runtime
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    valid_datasets = {}
    for dataset_name in cfg.data.valid_datasets:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.valid_datasets[dataset_name]
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    assert (
        "MultiDataModuleWrapper" in cfg.data._target_
    ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
    )

    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        data_schematic.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

        # this remove annotation and image keys from the keymap
        km["norm_mode"] = True

        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        # infer_norm_from_dataset: load from precomputed JSON/dir if set, else compute (no disk write).
        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )
        # Cache norm stats if save_cache_dir is set
        save_cache_dir = OmegaConf.select(
            cfg, "norm_stats.save_cache_dir", default=None
        )
        if save_cache_dir:
            data_schematic.cache_stats(save_cache_dir=save_cache_dir)

    if cfg.reject_outliers:
        # Propagate the shared data schematic to top-level MultiDatasets for bounds checks.
        # Use datamodule.train_datasets (null entries already filtered by the wrapper).
        _propagate_data_schematic_to_datasets(
            data_schematic,
            datamodule.train_datasets,
        )
    viz_func = cfg.visualization
    viz_func_dict = {}
    for embodiment_name, embodiment_viz_func in viz_func.items():
        viz_func_dict[embodiment_name] = hydra.utils.instantiate(embodiment_viz_func)

    viz_func_dict = {}
    for embodiment_name, embodiment_viz_func in cfg.visualization.items():
        viz_func_dict[embodiment_name] = hydra.utils.instantiate(embodiment_viz_func)

    # NOTE: We also pass the data_schematic_dict into the robomimic model's instatiation now that we've initialzied the shapes and norm stats.  In theory, upon loading the PL checkpoint, it will remember this, but let's see.
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        data_schematic_state=data_schematic.to_state(),
        viz_func=viz_func_dict,
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    _log_dataset_frame_counts(datamodule.train_datasets, datamodule.valid_datasets)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Resolve mode: support both new `mode` key and legacy `train`/`eval` booleans
    if cfg.get("mode") is not None:
        mode = cfg.mode
    elif cfg.get("train", False):
        mode = "train"
    elif cfg.get("eval", False):
        mode = "eval"
    else:
        raise ValueError("Config must specify either `mode` or `train`/`eval` booleans")

    # In eval mode, apply trainer overrides from the eval object and disable logger
    if mode == "eval":
        eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
        log.info(
            "Eval mode: applying trainer overrides from eval config, disabling logger"
        )
        with open_dict(cfg):
            for k, v in eval_obj.override_dict.items():
                cfg.trainer[k] = v
            cfg.trainer.devices = 1
            cfg.trainer.num_nodes = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.logger = None

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        # auto_requeue=False: skip Lightning's built-in handler. Its handler
        # calls save_checkpoint inside the signal context, which wedges on a
        # NCCL barrier when many ranks (e.g. 16 at 4×4) all enter the handler
        # mid-batch. We replace it with a should_stop-based path below.
        plugins.append(SLURMEnvironment(auto_requeue=False))
        print("SLURM REQUEUE ENABLED (custom should_stop handler)")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins
    )

    requeue_requested = {"flag": False}
    if os.environ.get("SLURM_JOB_ID"):
        def _sigusr1_handler(signum, _frame):
            requeue_requested["flag"] = True
            # Override min_epochs/min_steps so `trainer.should_stop` isn't
            # silently ignored by Lightning's `_can_stop_early` guard ("Trainer
            # was signaled to stop but min_epochs has not been met"). The
            # config sets min_epochs=2000; without this override SIGUSR1 has
            # no effect. min_epochs is on FitLoop; min_steps is on its inner
            # epoch_loop (FitLoop.min_steps is a read-only property).
            # NOTE: stop happens at the next epoch boundary, not mid-batch —
            # there can be up to one full epoch of delay before fit() returns.
            trainer.fit_loop.min_epochs = 0
            trainer.fit_loop.epoch_loop.min_steps = None
            trainer.should_stop = True
            log.info(
                f"[rank {trainer.global_rank}] SIGUSR1 received; "
                "requested graceful stop and requeue"
            )
        signal.signal(signal.SIGUSR1, _sigusr1_handler)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        import glob as _glob
        hpc_ckpts = _glob.glob(
            os.path.join(trainer.default_root_dir, "hpc_ckpt_*.ckpt")
        )
        last_ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        if hpc_ckpts:
            latest_hpc = max(
                hpc_ckpts,
                key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]),
            )
            log.info(f"Detected SLURM requeue — resuming from hpc ckpt: {latest_hpc}")
            cfg.ckpt_path = latest_hpc
        elif os.path.isfile(last_ckpt_path):
            log.info("Detected SLURM requeue — resuming from 'last.ckpt'")
            cfg.ckpt_path = last_ckpt_path
        else:
            log.warning(
                "Detected SLURM requeue but no checkpoint found — fresh start"
            )

    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    if mode == "train":
        if cfg.get("evaluator") is not None:
            eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            model.evaluator = eval_obj
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )

        if requeue_requested["flag"]:
            log.info(
                "Training stopped early due to SIGUSR1; saving hpc_ckpt then "
                "calling 'scontrol requeue' so SLURM relaunches the job."
            )
            hpc_save_path = os.path.join(
                trainer.default_root_dir,
                f"hpc_ckpt_{trainer.global_step}.ckpt",
            )
            trainer.save_checkpoint(hpc_save_path)
            log.info(f"Saved hpc ckpt to {hpc_save_path}")

            if trainer.global_rank == 0:
                job_id = os.environ.get("SLURM_JOB_ID", "")
                if job_id:
                    import subprocess as _sp
                    log.info(f"Calling scontrol requeue {job_id} ...")
                    try:
                        rc = _sp.call(["scontrol", "requeue", job_id])
                    except FileNotFoundError:
                        rc = _sp.call(
                            f"scontrol requeue {job_id}", shell=True
                        )
                    if rc == 0:
                        log.info(f"Requeued SLURM job {job_id}")
                    else:
                        log.warning(
                            f"scontrol requeue {job_id} failed (rc={rc}); "
                            "job will exit without auto-requeue"
                        )

            import sys as _sys
            _sys.exit(1)
    elif mode == "eval":
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        model.evaluator = eval_obj
        # Load checkpoint weights manually so we can reset the epoch counter
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            log.info(f"Loaded weights from {ckpt_path}")
        log.info("Starting evaluation!")
        trainer.validate(model=model, datamodule=datamodule)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    test_metrics = {}  # my stub
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path="./hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    print(OmegaConf.to_yaml(cfg))

    # cfg = OmegaConf.resolve(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    main()
