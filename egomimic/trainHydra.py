import copy
import hashlib
import os
import re
import signal
from pathlib import Path
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

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.eval.eval import Eval
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP, as_valid_groups
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.ema_callback import EMACallback
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.slurm_requeue import SaveOnlySignalCheckpoint
from egomimic.utils.utils import extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _slurm_auto_requeue(cfg: DictConfig) -> bool:
    """Resolve and cross-check the configured Slurm requeue owner."""

    configured_owner = str(
        OmegaConf.select(
            cfg,
            "runtime.slurm_requeue_owner",
            default="lightning",
        )
    )
    runner_owner = os.environ.get("ICE_REQUEUE_OWNER")
    child_requeue_disabled = os.environ.get("ICE_CHILD_REQUEUE_DISABLED")
    if configured_owner == "lightning" and (
        (runner_owner is None and child_requeue_disabled is None)
        or (runner_owner == "child" and child_requeue_disabled == "0")
    ):
        return True
    if (
        configured_owner == "runner"
        and runner_owner == "runner"
        and child_requeue_disabled == "1"
    ):
        return False
    raise RuntimeError(
        "inconsistent Slurm requeue ownership: "
        f"runtime.slurm_requeue_owner={configured_owner!r}, "
        f"ICE_REQUEUE_OWNER={runner_owner!r}, "
        f"ICE_CHILD_REQUEUE_DISABLED={child_requeue_disabled!r}; "
        "use lightning with no runner variables (or child/0), or runner "
        "with runner/1"
    )


def _slurm_environment(cfg: DictConfig) -> SLURMEnvironment:
    """Build the one Slurm environment plugin with an explicit requeue owner."""

    return SLURMEnvironment(
        auto_requeue=_slurm_auto_requeue(cfg),
        requeue_signal=signal.SIGUSR1,
    )


def _instantiate_slurm_callbacks(cfg: DictConfig) -> List[Callback]:
    """Add the save-only signal callback when an external runner owns requeue."""

    if not os.environ.get("SLURM_JOB_ID") or _slurm_auto_requeue(cfg):
        return []
    save_signal = str(
        OmegaConf.select(cfg, "runtime.slurm_save_signal", default="SIGUSR2")
    )
    if save_signal != "SIGUSR2":
        raise RuntimeError(
            f"runtime.slurm_save_signal must be 'SIGUSR2'; got {save_signal!r}"
        )
    checkpoint_dir = OmegaConf.select(
        cfg,
        "runtime.slurm_signal_checkpoint_dir",
        default=None,
    )
    if not checkpoint_dir:
        raise RuntimeError(
            "runtime.slurm_signal_checkpoint_dir is required for runner-owned requeue"
        )
    return [
        SaveOnlySignalCheckpoint(
            checkpoint_dir=str(checkpoint_dir),
            save_signal=signal.SIGUSR2,
        )
    ]


def _instantiate_trainer_plugins(cfg: DictConfig) -> List[Any]:
    """Instantiate each trainer plugin exactly once."""

    plugins: List[Any] = []
    if cfg.get("mmap_checkpoint", True):
        plugins.append(MmapCheckpointIO())
    if os.environ.get("SLURM_JOB_ID"):
        slurm_environment = _slurm_environment(cfg)
        plugins.append(slurm_environment)
        print(
            "SLURM ENVIRONMENT ENABLED "
            f"(Lightning auto_requeue={slurm_environment.auto_requeue})"
        )
    return plugins


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    return OmegaConf.create({"model": model_cfg})


def _validate_run_config(cfg: DictConfig) -> str:
    if cfg.get("model") is None:
        raise ValueError("Select a complete Pipeline model config")
    if OmegaConf.select(cfg, "model.pipeline", default=None) is None:
        raise ValueError("Model config must define model.pipeline")
    if cfg.get("data") is None:
        raise ValueError("Select a data config")
    mode = cfg.get("mode")
    if mode not in {"train", "eval"}:
        raise ValueError("Config mode must be 'train' or 'eval'")
    if mode == "eval" and cfg.get("evaluator") is None:
        raise ValueError("Evaluation mode requires an evaluator config")
    return mode


def _load_eval_checkpoint(model, checkpoint: dict, cfg: DictConfig):
    """Strictly restore a configured Pipeline for standalone evaluation."""
    algo = getattr(model, "model", None)
    if not isinstance(algo, PipelineAlgo):
        raise TypeError("Evaluation model must wrap PipelineAlgo")
    settings = OmegaConf.select(cfg, "eval_checkpoint", default=None)
    use_ema = settings.get("use_ema", False) if settings is not None else False
    if not isinstance(use_ema, bool):
        raise TypeError("eval_checkpoint.use_ema must be a boolean")
    strict_load_pipeline_checkpoint(
        algo,
        checkpoint,
        use_ema=use_ema,
    )
    return model


def _callbacks_for_mode(callbacks: List[Callback], mode: str) -> List[Callback]:
    """Keep training-time EMA swapping out of standalone evaluation."""
    if mode != "eval":
        return callbacks
    return [callback for callback in callbacks if not isinstance(callback, EMACallback)]


def _resolve_model_wrapper_class(cfg: DictConfig) -> type[ModelWrapper]:
    """Resolve a configured Lightning wrapper without constructing its model."""
    target = OmegaConf.select(cfg, "model._target_", default=None)
    if target is None:
        return ModelWrapper
    wrapper_class = hydra.utils.get_class(str(target))
    if not isinstance(wrapper_class, type) or not issubclass(
        wrapper_class, ModelWrapper
    ):
        raise TypeError(
            "cfg.model._target_ must resolve to a ModelWrapper subclass; "
            f"got {target!r}"
        )
    return wrapper_class


def _instantiate_model_wrapper(cfg: DictConfig) -> LightningModule:
    """Construct the configured wrapper with every wrapper-level control."""
    wrapper_class = _resolve_model_wrapper_class(cfg)
    return wrapper_class(
        config_tree=_build_model_config_tree(cfg),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
        enable_grad_norm=bool(cfg.model.get("enable_grad_norm", True)),
    )


def _log_dataset_frame_counts(train_datasets: dict, valid_entries) -> None:
    """Frame counts per split. ``valid_entries`` is the datamodule's
    ``iter_valid_datasets()`` -- (group, source, dataset) over EVERY val group,
    so a run with more than one group reports all of them rather than the one
    the flat alias happens to point at."""
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    valid_lengths = []
    for group, source, ds in valid_entries:
        rows.append((f"valid[{group}]", source, len(ds)))
        valid_lengths.append(len(ds))
    if valid_lengths:
        rows.append(("TOTAL", "(valid)", sum(valid_lengths)))
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


def _checkpoint_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runner_resume_checkpoint() -> tuple[str | None, int | None]:
    """Validate the exact checkpoint selected by the external ICE runner."""

    variable_names = (
        "ICE_RESUME_CHECKPOINT",
        "ICE_RESUME_CHECKPOINT_SHA256",
        "ICE_RESUME_GLOBAL_STEP",
    )
    missing = [name for name in variable_names if name not in os.environ]
    if missing:
        raise RuntimeError(
            "runner-owned requeue requires the runner resume environment: "
            + ", ".join(missing)
        )

    checkpoint_text = os.environ["ICE_RESUME_CHECKPOINT"]
    expected_sha256 = os.environ["ICE_RESUME_CHECKPOINT_SHA256"]
    expected_step_text = os.environ["ICE_RESUME_GLOBAL_STEP"]
    values = (checkpoint_text, expected_sha256, expected_step_text)
    if not any(values):
        restart_text = os.environ.get("SLURM_RESTART_COUNT", "0")
        if re.fullmatch(r"0|[1-9][0-9]*", restart_text) is None:
            raise RuntimeError(
                f"SLURM_RESTART_COUNT must be a nonnegative integer; got {restart_text!r}"
            )
        if int(restart_text) > 0:
            raise RuntimeError(
                "runner-owned Slurm restart has no ICE_RESUME_CHECKPOINT"
            )
        return None, None
    if not all(values):
        raise RuntimeError(
            "runner resume checkpoint, SHA-256, and global step must be supplied together"
        )
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise RuntimeError(
            "ICE_RESUME_CHECKPOINT_SHA256 must be 64 lowercase hex digits"
        )
    if re.fullmatch(r"0|[1-9][0-9]*", expected_step_text) is None:
        raise RuntimeError("ICE_RESUME_GLOBAL_STEP must be a nonnegative integer")

    checkpoint_path = Path(checkpoint_text)
    if not checkpoint_path.is_absolute():
        raise RuntimeError("ICE_RESUME_CHECKPOINT must be an absolute path")
    try:
        checkpoint_path = checkpoint_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ICE_RESUME_CHECKPOINT does not exist: {checkpoint_text}"
        ) from exc
    if not checkpoint_path.is_file():
        raise RuntimeError(f"ICE_RESUME_CHECKPOINT is not a file: {checkpoint_path}")

    identity_before = _checkpoint_file_identity(checkpoint_path)
    actual_sha256 = _sha256(checkpoint_path)
    identity_after_hash = _checkpoint_file_identity(checkpoint_path)
    if identity_after_hash != identity_before:
        raise RuntimeError("ICE resume checkpoint changed while hashing")
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "ICE_RESUME_CHECKPOINT_SHA256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )

    checkpoint = MmapCheckpointIO().load_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if _checkpoint_file_identity(checkpoint_path) != identity_before:
        raise RuntimeError("ICE resume checkpoint changed while validating global_step")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("ICE resume checkpoint must contain a mapping")
    actual_step = checkpoint.get("global_step")
    if (
        not isinstance(actual_step, int)
        or isinstance(actual_step, bool)
        or actual_step < 0
    ):
        raise RuntimeError(
            "ICE resume checkpoint global_step must be a nonnegative integer"
        )
    expected_step = int(expected_step_text)
    if actual_step != expected_step:
        raise RuntimeError(
            "ICE_RESUME_GLOBAL_STEP mismatch: "
            f"expected={expected_step} actual={actual_step}"
        )
    return str(checkpoint_path), actual_step


def _resolve_training_checkpoint(cfg: DictConfig, trainer: Trainer) -> str | None:
    """Resolve resume authority without mixing runner and Lightning semantics."""

    owner = str(
        OmegaConf.select(cfg, "runtime.slurm_requeue_owner", default="lightning")
    )
    _slurm_auto_requeue(cfg)

    if owner == "runner":
        selected_path, selected_step = _validate_runner_resume_checkpoint()
        configured_path = cfg.get("ckpt_path")
        if configured_path:
            configured_resolved = Path(str(configured_path)).expanduser().resolve()
            if selected_path is None or configured_resolved != Path(selected_path):
                raise RuntimeError(
                    "cfg.ckpt_path disagrees with runner-selected ICE_RESUME_CHECKPOINT"
                )
        with open_dict(cfg):
            cfg.ckpt_path = selected_path
        if selected_path is None:
            log.info("ICE runner selected a clean start with no resume checkpoint")
        else:
            log.info(
                "ICE runner resume checkpoint validated: "
                f"path={selected_path} global_step={selected_step}"
            )
        return selected_path

    configured_path = cfg.get("ckpt_path")
    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
        and not configured_path
    ):
        configured_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        with open_dict(cfg):
            cfg.ckpt_path = configured_path
        log.info("Detected Lightning-owned Slurm requeue — using 'last.ckpt'")
    elif (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        log.info("Detected Lightning-owned Slurm requeue — using configured checkpoint")
    return configured_path


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    mode = _validate_run_config(cfg)

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

    # `valid_datasets` is either flat ({source: dataset-config}) or grouped
    # ({group: {source: dataset-config}}). Instantiate one level deeper for the
    # grouped shape; `MultiDataModuleWrapper` normalises both to groups.
    valid_group_configs = as_valid_groups(cfg.data.valid_datasets)
    valid_datasets = {
        group: {
            source: hydra.utils.instantiate(dataset_config)
            for source, dataset_config in members.items()
        }
        for group, members in valid_group_configs.items()
    }
    if list(valid_datasets) == [DEFAULT_VALID_GROUP]:
        # Hand the flat shape back unchanged so a single-group config produces
        # exactly the mapping it did before groups existed.
        valid_datasets = valid_datasets[DEFAULT_VALID_GROUP]

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    assert "MultiDataModuleWrapper" in cfg.data._target_, (
        "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    )
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

    if cfg.get("norm_stats_only", False):
        if not OmegaConf.select(cfg, "norm_stats.save_cache_dir", default=None):
            raise ValueError("norm_stats_only requires norm_stats.save_cache_dir")
        log.info("Normalization-only mode complete")
        return {}, {"cfg": cfg, "norm_stats": norm_stats}

    # Wire each training/valid MultiDataset to the stats-only ``norm_stats``
    # by reference. Bounds-check + normalize run at the MultiDataset level in
    # ``__getitem__`` — not as per-leaf transforms — which avoids the shared
    # transform_list aliasing trap.
    for ds in datamodule.train_datasets.values():
        ds.set_norm_stats_from(norm_stats)
    # EVERY val group, via the datamodule's own iterator. Do NOT iterate
    # `datamodule.valid_datasets` here: it is a back-compat alias for a single
    # group (the default if present, else the first), so it silently skips the
    # rest. Those datasets would then emit unnormalised samples that the
    # evaluator unnormalises again, and the resulting metrics look plausible.
    for _group, _source, _ds in datamodule.iter_valid_datasets():
        _ds.set_norm_stats_from(norm_stats)

    # Fail loudly rather than silently mis-normalising a split: every train and
    # val dataset must now share the stats object by reference.
    _unwired = [
        f"train/{name}"
        for name, ds in datamodule.train_datasets.items()
        if getattr(ds, "norm_stats", None) is not norm_stats.norm_stats
    ] + [
        f"{group}/{source}"
        for group, source, ds in datamodule.iter_valid_datasets()
        if getattr(ds, "norm_stats", None) is not norm_stats.norm_stats
    ]
    if _unwired:
        raise RuntimeError(
            "norm stats were not wired to every dataset; these would emit "
            f"unnormalised samples that the evaluator then unnormalises again: {_unwired}"
        )
    log.info(
        f"norm stats wired to {len(datamodule.train_datasets)} train and "
        f"{sum(1 for _ in datamodule.iter_valid_datasets())} valid datasets "
        f"across {len(datamodule.valid_groups)} val group(s): "
        f"{list(datamodule.valid_groups)}"
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = _instantiate_model_wrapper(cfg)

    _log_dataset_frame_counts(
        datamodule.train_datasets, datamodule.iter_valid_datasets()
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    if mode == "train":
        callbacks.extend(_instantiate_slurm_callbacks(cfg))

    callbacks = _callbacks_for_mode(callbacks, mode)
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
    plugins = _instantiate_trainer_plugins(cfg)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins or None
    )

    if mode == "train":
        _resolve_training_checkpoint(cfg, trainer)

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

    if mode == "train":
        if cfg.get("evaluator") is not None:
            eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            eval_obj.bind_data_context(normalizer=norm_stats)
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
        eval_obj.bind_data_context(normalizer=norm_stats)
        model.evaluator = eval_obj

        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _load_eval_checkpoint(model, checkpoint, cfg)
            log.info(f"Loaded weights from {ckpt_path}")
        log.info("Starting evaluation!")
        trainer.validate(model=model, datamodule=datamodule)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return trainer.callback_metrics, object_dict


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

    train(cfg)


if __name__ == "__main__":
    main()
