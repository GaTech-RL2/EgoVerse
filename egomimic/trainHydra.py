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
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.resolve_memo import resolve_once
from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, PinError
from egomimic.utils.checkpoint_utils import load_checkpoint_weights
from egomimic.utils.compile_cache import set_per_job_compile_cache_dir
from egomimic.utils.env import load_env
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper

OmegaConf.register_new_resolver("eval", eval)

log = RankedLogger(__name__, rank_zero_only=True)


_PI_WEIGHT_KEY = "model.robomimic_model.config.pytorch_weight_path"


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    """The config tree ``ModelWrapper`` receives. Only this copy skips the PI base
    weights (see ``_weights_from_checkpoint``); ``cfg`` keeps the real path for
    the logged hyperparameters."""
    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and isinstance(model_cfg.robomimic_model, DictConfig)
        and "norm_stats" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.norm_stats = None
    tree = OmegaConf.create({"model": model_cfg})
    has_weights = OmegaConf.select(tree, _PI_WEIGHT_KEY, default=None) is not None
    if has_weights and _weights_from_checkpoint(cfg):
        log.info(
            f"Loading every weight from {cfg.ckpt_path}: {_PI_WEIGHT_KEY}=null "
            "for the model (the base safetensors need not exist here)"
        )
        OmegaConf.update(tree, _PI_WEIGHT_KEY, None)
    return tree


def _requeue_resume_path(cfg: DictConfig) -> Optional[str]:
    """``<checkpoint dir>/last.ckpt`` when this process is a Slurm requeue
    (``SLURM_RESTART_COUNT`` > 0), else None. The dir is the ModelCheckpoint
    callback's ``dirpath`` (``<run>/checkpoints`` when unset), read from ``cfg``
    so this can run before the callbacks and Trainer exist."""
    if not os.environ.get("SLURM_JOB_ID"):
        return None
    if os.environ.get("SLURM_RESTART_COUNT", "0") == "0":
        return None
    ckpt_dir = OmegaConf.select(
        cfg, "callbacks.model_checkpoint.dirpath", default=None
    ) or os.path.join(cfg.trainer.default_root_dir, "checkpoints")
    return os.path.join(ckpt_dir, "last.ckpt")


def _prepare_checkpoint_resume(cfg: DictConfig) -> None:
    """Settle ``cfg.ckpt_path`` before the model config tree is built: a requeued
    job resumes from ``last.ckpt`` if it exists; one preempted before its first
    checkpoint keeps the launch-time ``ckpt_path`` (warning)."""
    requeue = _requeue_resume_path(cfg)
    if requeue is None:
        return
    if os.path.isfile(requeue):
        log.info(f"Detected SLURM requeue — resuming from {requeue}")
        cfg.ckpt_path = requeue
        return
    fallback = cfg.get("ckpt_path")
    log.warning(
        f"SLURM requeue detected but {requeue} does not exist; falling back to "
        f"ckpt_path={fallback}" + ("" if fallback else " (training starts over)")
    )


def _weights_from_checkpoint(cfg: DictConfig) -> bool:
    """True when ``cfg.ckpt_path`` is a checkpoint file whose state_dict will
    overwrite every weight, so PI need not read its base safetensors (it treats
    ``pytorch_weight_path=None`` as "no pretrained weights").

    Requires an existing file: Lightning's special values (``last``, ``best``,
    ``hpc``, ``registry:...``) may resolve to no checkpoint, which would then
    train from random init. ``pretrained=true`` is eval_latent's flag: there
    ``ckpt_path`` only routes the output dir and the base weights ARE the model
    under evaluation (elsewhere the flag is unused and just keeps them)."""
    ckpt_path = cfg.get("ckpt_path")
    return bool(ckpt_path) and os.path.isfile(ckpt_path) and not cfg.get("pretrained")


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


def _instantiate_dataset(*args, **kwargs):
    """``hydra.utils.instantiate`` wraps any exception raised by the target in an
    ``InstantiationException``. A bad pin must surface at launch as a bare
    ``PinError`` (not a generic Hydra error the caller has to unwrap), so this
    intercepts only that case and re-raises the original ``PinError`` -- keeping
    the ``InstantiationException`` as its context (``from e``, not ``from None``)
    so the Hydra target/config info isn't lost. Every other target exception is
    re-raised exactly as Hydra raised it."""
    try:
        return hydra.utils.instantiate(*args, **kwargs)
    except hydra.errors.InstantiationException as e:
        if isinstance(e.__cause__, PinError):
            raise e.__cause__ from e
        raise


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

    # One SQL pull / path resolution per dataset spec across train, valid and
    # the norm-stat copies; dropped on exit so nothing outlives this run.
    with resolve_once():
        train_datasets = {}
        for dataset_name in cfg.data.train_datasets:
            train_datasets[dataset_name] = _instantiate_dataset(
                cfg.data.train_datasets[dataset_name], dataset_name=dataset_name
            )

        valid_datasets = {}
        for dataset_name in cfg.data.valid_datasets:
            valid_datasets[dataset_name] = _instantiate_dataset(
                cfg.data.valid_datasets[dataset_name], dataset_name=dataset_name
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

        from egomimic.rldb.zarr import norm_cache

        sample_frac = OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0)
        explicit_path = OmegaConf.select(
            cfg, "norm_stats.precomputed_norm_path", default=None
        )
        # Code default is False so configs without the key keep the old behaviour
        # (always recompute); the shipped configs set norm_stats.use_cache=true.
        use_cache = bool(OmegaConf.select(cfg, "norm_stats.use_cache", default=False))
        cache_dir = OmegaConf.select(cfg, "norm_stats.cache_dir", default=None)

        for dataset_name, dataset in datamodule.train_datasets.items():
            log.info(f"Inferring shapes for dataset <{dataset_name}>")
            norm_stats.infer_shapes_from_batch(dataset[0])
            instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
            keymap_cfg = instantiate_copy.resolver.key_map
            km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

            # this remove annotation and image keys from the keymap
            km["norm_mode"] = True

            instantiate_copy.resolver.key_map = km
            norm_dataset = _instantiate_dataset(
                instantiate_copy, dataset_name=dataset_name
            )

            emb = get_embodiment_id(dataset_name)
            key = inputs = cached = None
            if explicit_path is None and use_cache and cache_dir:
                episodes = {
                    h: norm_cache.episode_fingerprint(getattr(ds, "episode_path", None))
                    for h, ds in dataset.datasets.items()
                }
                inputs = norm_cache.cache_inputs(
                    dataset_name,
                    episodes,
                    cfg.data.train_datasets[dataset_name],
                    sample_frac,
                )
                key = norm_cache.norm_cache_key(inputs)
                cached = norm_cache.find_cached(cache_dir, dataset_name, key, emb)
                if cached is not None:
                    log.info(f"norm stats for <{dataset_name}>: cache hit {cached}")

            # infer_norm_from_dataset: load from precomputed JSON/dir if set, else compute (no disk write).
            norm_stats.infer_norm_from_dataset(
                norm_dataset,
                dataset_name,
                sample_frac=sample_frac,
                num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
                precomputed_norm_path=explicit_path
                if explicit_path is not None
                else cached,
            )
            if key is not None and cached is None:
                if norm_stats.norm_stats.get(emb):
                    norm_cache.write_cached(
                        cache_dir,
                        dataset_name,
                        key,
                        inputs,
                        emb,
                        norm_stats.norm_stats[emb],
                        norm_stats._norm_run_metadata,
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

    _prepare_checkpoint_resume(cfg)

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

        if hasattr(eval_obj, "run"):
            eval_obj.run(trainer, model, datamodule, cfg)
        else:
            # Default: load checkpoint + validate (unchanged from main)
            ckpt_path = cfg.get("ckpt_path")
            if ckpt_path:
                load_checkpoint_weights(model, ckpt_path)
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
    # Here, not at import: a `-m` submitit launcher imports this module but only
    # the job runs main(), so each job keys the cache on its own SLURM_JOB_ID.
    set_per_job_compile_cache_dir()

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
