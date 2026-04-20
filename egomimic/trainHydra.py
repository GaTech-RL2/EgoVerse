import copy
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
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


# ---------------------------------------------------------------------------
# Modal integration
# ---------------------------------------------------------------------------


class ModalAutoRestartCallback(Callback):
    """Save a checkpoint and spawn a continuation job ~30 min before the Modal
    container times out, then stop the current run gracefully.

    Required env vars (set automatically by egomimic/modal/run.py):
        MODAL_TIMEOUT_SECONDS   container timeout in seconds
        MODAL_START_TIME        unix timestamp when the container started
        MODAL_HYDRA_ARGS        JSON-encoded list of the original hydra overrides
        MODAL_GIT_REMOTE        remote URL used to clone the repo
        MODAL_GIT_COMMIT        git SHA checked out in the container
    """

    _RESTART_MARGIN_SEC = 1800  # save + spawn 30 min before timeout

    def __init__(self) -> None:
        super().__init__()
        self._triggered = False
        self._start = float(os.environ.get("MODAL_START_TIME", time.time()))
        self._timeout = int(os.environ.get("MODAL_TIMEOUT_SECONDS", 86400))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._triggered:
            return
        remaining = self._timeout - (time.time() - self._start)
        if remaining < self._RESTART_MARGIN_SEC:
            self._triggered = True
            self._auto_restart(trainer)

    def _auto_restart(self, trainer: "Trainer") -> None:
        ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "modal_auto_restart.ckpt"
        )
        trainer.save_checkpoint(ckpt_path)
        log.info(f"[ModalAutoRestart] Checkpoint saved → {ckpt_path}")

        # Grab the live WandB run ID so the continuation logs to the same run
        wandb_run_id = None
        for lgr in trainer.loggers:
            if hasattr(lgr, "experiment") and hasattr(lgr.experiment, "id"):
                wandb_run_id = lgr.experiment.id
                break

        raw_args: list = json.loads(os.environ.get("MODAL_HYDRA_ARGS", "[]"))
        new_args = [
            a
            for a in raw_args
            if not a.startswith("ckpt_path=") and not a.startswith("wandb_run_id=")
        ]
        new_args.append(f"ckpt_path={ckpt_path}")
        if wandb_run_id:
            new_args.append(f"wandb_run_id={wandb_run_id}")

        git_remote = os.environ.get("MODAL_GIT_REMOTE", "")
        git_commit = os.environ.get("MODAL_GIT_COMMIT", "")
        wandb_api_key = os.environ.get("WANDB_API_KEY", "")

        if trainer.is_global_zero:
            try:
                import modal as _modal

                fn = _modal.Function.from_name(
                    "egoverse-train",
                    "run_hydra_train",
                    environment_name=os.environ.get("MODAL_ENVIRONMENT", "main"),
                )
                handle = fn.spawn(
                    tuple(new_args), git_remote, git_commit, wandb_api_key
                )
                log.info(f"[ModalAutoRestart] Spawned continuation: {handle.object_id}")
            except Exception as exc:
                log.error(f"[ModalAutoRestart] Failed to spawn continuation: {exc}")

        trainer.should_stop = True
        log.info(
            "[ModalAutoRestart] Stopping current run — continuation job is running"
        )


def _submit_to_modal(cfg: DictConfig) -> None:
    """Delegate training to Modal via the `modal run` CLI and exit.

    Uses a subprocess so the modal package is imported in a clean process and
    the egomimic/modal/ directory does not shadow the installed modal package.

    Supported CLI overrides (stripped before forwarding to the container):
        +modal_gpu=H100          GPU type: A100, H100, A10G, A100:4 (4×A100), etc.
        +modal_cpu=16            Number of CPU cores
        +modal_memory_gb=128     RAM in GB  (or +modal_memory_mb=131072)
    """
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent

    modal_env = os.environ.copy()

    _MODAL_KEYS = {"modal_gpu", "modal_cpu", "modal_memory_gb", "modal_memory_mb"}
    container_overrides = []
    gpu_count = 1
    for override in HydraConfig.get().overrides.task:
        key = override.lstrip("+").split("=")[0]
        if key in _MODAL_KEYS:
            val = override.split("=", 1)[1]
            if key == "modal_gpu":
                modal_env["MODAL_GPU"] = val
                gpu_count = int(val.split(":")[1]) if ":" in val else 1
            elif key == "modal_cpu":
                modal_env["MODAL_CPU"] = val
            elif key == "modal_memory_gb":
                modal_env["MODAL_MEMORY_GB"] = val
            elif key == "modal_memory_mb":
                modal_env["MODAL_MEMORY_MB"] = val
        else:
            container_overrides.append(override)

    # Sync launch_params.gpus_per_node with the requested GPU count so DDP
    # uses all available GPUs (devices: ${launch_params.gpus_per_node} in ddp_modal.yaml)
    container_overrides = [
        a
        for a in container_overrides
        if not a.lstrip("+").startswith("launch_params.gpus_per_node=")
    ]
    container_overrides.append(f"launch_params.gpus_per_node={gpu_count}")

    modal_environment = modal_env.get("MODAL_ENVIRONMENT", "main")

    cmd = [
        sys.executable,
        "-m",
        "modal",
        "run",
        "--detach",
        "--env",
        modal_environment,
        "egomimic/modal/run.py::submit",
        "--",
        *container_overrides,
    ]

    gpu = modal_env.get("MODAL_GPU", "A100")
    cpu = modal_env.get("MODAL_CPU", "12")
    mem_gb = modal_env.get("MODAL_MEMORY_GB") or str(
        int(modal_env.get("MODAL_MEMORY_MB", "65536")) // 1024
    )
    print(f"Modal resources: gpu={gpu}  cpu={cpu}  memory={mem_gb}GB")
    print(f"Submitting to Modal via: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(repo_root), env=modal_env)
    sys.exit(result.returncode)


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

    # Register Modal auto-restart callback when running inside a Modal container
    if os.environ.get("MODAL_IS_REMOTE") == "1" and os.environ.get(
        "MODAL_TIMEOUT_SECONDS"
    ):
        callbacks.append(ModalAutoRestartCallback())
        log.info("[ModalAutoRestart] Callback registered")

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

    # When resuming from a Modal auto-restart, wire wandb_run_id into the W&B logger
    # so the continuation logs to the same W&B run.
    wandb_run_id = cfg.get("wandb_run_id")
    if wandb_run_id and cfg.get("logger") and hasattr(cfg.logger, "wandb"):
        with open_dict(cfg):
            cfg.logger.wandb.id = wandb_run_id
            cfg.logger.wandb.resume = "allow"

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(
            SLURMEnvironment(requeue_signal=[signal.SIGUSR1, signal.SIGUSR2])
        )
        print("SLURM REQUEUE ENABLED")

    with open_dict(cfg):
        cfg.trainer.pop("_modal", None)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

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
        last_ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        log.info("Detected SLURM requeue — resuming from 'last.ckpt'")
        cfg.ckpt_path = last_ckpt_path

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

    # If trainer._modal is set and we're not already inside a Modal container,
    # submit the job to Modal using the raw Hydra CLI overrides and exit.
    if OmegaConf.select(cfg, "trainer._modal", default=False):
        if os.environ.get("MODAL_IS_REMOTE") != "1":
            try:
                _submit_to_modal(cfg)
            except ImportError:
                raise RuntimeError(
                    "trainer._modal=true requires the 'modal' package. "
                    "Install it with: pip install modal"
                )
            return  # _submit_to_modal calls sys.exit, but keep for type checkers

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
