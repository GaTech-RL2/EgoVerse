import copy
import os
import signal
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from fsspec.implementations.local import LocalFileSystem
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from tabulate import tabulate

import egomimic.utils.hydra_resolvers  # noqa: F401  -- registers OmegaConf resolvers
from egomimic.eval.eval import Eval
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import set_global_seed
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
        and "norm_stats" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.norm_stats = None
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


class MmapCheckpointIO(TorchCheckpointIO):
    """``TorchCheckpointIO`` that memory-maps tensor storages instead of reading them.

    Every DDP rank loads the checkpoint independently and with no rank guard
    (``checkpoint_connector.resume_start``), so a plain read holds N private
    copies of the file in host RAM at once -- N x (weights + optimizer moments),
    and the moments are 2x the weights for AdamW. Mapping the file instead lets
    ranks on a node share page-cache pages, so one physical copy backs them all.

    ``pl_load`` does not forward ``mmap``, hence the reimplementation. Falls back
    to the normal read for checkpoints that predate torch's zipfile format (they
    cannot be mapped).
    """

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Any] = lambda storage, loc: storage,
        weights_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        if not isinstance(fs, LocalFileSystem):
            # mmap needs a real local file; let pl_load handle fsspec/URL paths.
            return pl_load(path, map_location=map_location, weights_only=weights_only)
        try:
            return torch.load(
                path, map_location=map_location, weights_only=weights_only, mmap=True
            )
        except (RuntimeError, ValueError, NotImplementedError) as e:
            log.warning(
                f"mmap load of {path} failed ({e}); falling back to a full read"
            )
            return pl_load(path, map_location=map_location, weights_only=weights_only)


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

    # Stats-only MultiDataset (no graph of its own; explicitly populated from
    # datamodule.train_datasets). MultiDataset now owns NormStats's role too.
    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats.populate_from_datasets(datamodule.train_datasets)

    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        norm_stats.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

        # this remove annotation and image keys from the keymap
        km["norm_mode"] = True

        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        # infer_norm_from_dataset: load from precomputed JSON/dir if set, else compute (no disk write).
        norm_stats.infer_norm_from_dataset(
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
            norm_stats.cache_stats(save_cache_dir=save_cache_dir)

    # Wire each training/valid MultiDataset to the stats-only ``norm_stats``
    # by reference. Bounds-check + normalize run at the MultiDataset level in
    # ``__getitem__`` — not as per-leaf transforms — which avoids the shared
    # transform_list aliasing trap.
    for ds in datamodule.train_datasets.values():
        ds.set_norm_stats_from(norm_stats)
    for ds in datamodule.valid_datasets.values():
        ds.set_norm_stats_from(norm_stats)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        norm_stats_state=norm_stats.to_state(),
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
    if cfg.get("mmap_checkpoint", True):
        plugins.append(MmapCheckpointIO())
    if os.environ.get("SLURM_JOB_ID"):
        # requeue_signal is a single signal, not a list -- SignalConnector passes
        # it straight to signal.getsignal(), which raises TypeError on a list.
        plugins.append(SLURMEnvironment(requeue_signal=signal.SIGUSR1))
        print("SLURM REQUEUE ENABLED")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins or None
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

    # ``load_weights_only``: pre-load model state_dict from ``ckpt_path``, then
    # pass ``ckpt_path=None`` to ``trainer.fit()`` so Lightning skips the
    # optimizer/scheduler/epoch-state restore. Fixes a DDP-wide hang where
    # ``restore_optimizers`` sits on a size-1 ALLREDUCE that only some ranks
    # post (torch's ``load_optimizer_state_dict`` traversal diverges on 5B-param
    # ckpts under find_unused_parameters_true). Trade-off: AdamW moments start
    # fresh — sub-optimal for a resume but preserves training continuity better
    # than a fully-fresh run.
    #
    # WANDB CONTINUITY: after skipping Lightning's ckpt restore the trainer's
    # ``global_step`` counter starts at 0, but the existing wandb run already
    # has data through the ckpt's global_step. Logging with step<max_step causes
    # visible backtracking in the wandb UI. We extract ``global_step`` from the
    # ckpt and monkey-patch each ``WandbLogger.log_metrics`` to add that offset
    # so future metrics land at ``ckpt_step + trainer.global_step`` — same
    # wandb run, monotonically increasing step, no backtracking.
    if cfg.get("load_weights_only", False) and cfg.get("ckpt_path"):
        _ckpt_path = cfg.ckpt_path
        log.info(f"[load_weights_only] Pre-loading state_dict from {_ckpt_path}")
        _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
        _missing, _unexpected = model.load_state_dict(_ckpt["state_dict"], strict=False)
        _ckpt_global_step = int(_ckpt.get("global_step", 0))
        _ckpt_epoch = int(_ckpt.get("epoch", 0))
        log.info(
            f"[load_weights_only] state_dict loaded: "
            f"{len(_missing)} missing / {len(_unexpected)} unexpected keys | "
            f"ckpt.global_step={_ckpt_global_step} ckpt.epoch={_ckpt_epoch}"
        )
        del _ckpt
        import gc as _gc

        _gc.collect()
        cfg.ckpt_path = None  # skip Lightning's full-ckpt restore

        # Patch WandbLogger(s) so metrics get offset by ckpt_global_step. Have
        # to walk both ``logger`` (list of Loggers) and ``trainer.loggers`` since
        # Lightning may re-wrap them. Also set ``WANDB_STEP_OFFSET`` env var so
        # any code that opens wandb.log directly can pick it up.
        os.environ["WANDB_STEP_OFFSET"] = str(_ckpt_global_step)
        from lightning.pytorch.loggers.wandb import WandbLogger as _WL

        for _lg in list(logger or []) + list(getattr(trainer, "loggers", []) or []):
            if isinstance(_lg, _WL) and not getattr(_lg, "_step_offset_patched", False):
                _lg._wandb_step_offset = _ckpt_global_step
                _orig_log_metrics = _lg.log_metrics

                def _offset_log_metrics(
                    metrics, step=None, _orig=_orig_log_metrics, _lg=_lg
                ):
                    if step is not None:
                        step = int(step) + int(_lg._wandb_step_offset)
                    return _orig(metrics, step=step)

                _lg.log_metrics = _offset_log_metrics
                _lg._step_offset_patched = True
                log.info(
                    f"[load_weights_only] Patched WandbLogger with "
                    f"step_offset={_ckpt_global_step}"
                )

    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    if mode == "train":
        if cfg.get("evaluator") is not None:
            eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            model.evaluator = eval_obj
        log.info("Starting training!")

        _dbg_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", -1)))
        _dbg_local = int(
            os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", -1))
        )
        _dbg_ws = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", -1)))
        _dbg_node = os.environ.get("SLURMD_NODENAME", "?")
        print(
            f"[DBG BEFORE_FIT] rank={_dbg_rank} local={_dbg_local} ws={_dbg_ws} node={_dbg_node}",
            flush=True,
        )
        try:
            trainer.fit(
                model=model,
                datamodule=datamodule,
                ckpt_path=cfg.get("ckpt_path"),
                weights_only=False,
            )
        except BaseException as _dbg_e:
            print(
                f"[DBG FIT_EXCEPTION] rank={_dbg_rank} type={type(_dbg_e).__name__} "
                f"msg={_dbg_e!r}",
                flush=True,
            )
            raise
        finally:
            print(
                f"[DBG AFTER_FIT] rank={_dbg_rank} "
                f"global_step={getattr(trainer, 'global_step', '?')} "
                f"current_epoch={getattr(trainer, 'current_epoch', '?')} "
                f"state={getattr(trainer, 'state', '?')}",
                flush=True,
            )
    elif mode == "eval":
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        model.evaluator = eval_obj

        if hasattr(eval_obj, "run"):
            eval_obj.run(trainer, model, datamodule, cfg)
        else:
            # Default: load checkpoint + validate (unchanged from main)
            ckpt_path = cfg.get("ckpt_path")
            if ckpt_path:
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
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
