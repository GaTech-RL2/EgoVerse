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
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from tabulate import tabulate

import egomimic.utils.hydra_resolvers  # noqa: F401  -- registers OmegaConf resolvers
from egomimic.eval.eval import Eval
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.resolve_memo import resolve_once
from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import (
    EvenStrideDataset,
    MultiDataset,
    PinError,
    pinned_episode_subset,
)
from egomimic.utils.checkpoint_utils import load_checkpoint_weights
from egomimic.utils.compile_cache import set_per_job_compile_cache_dir
from egomimic.utils.env import load_env
from egomimic.utils.gpu_orphans import arm_rank_pdeathsig, reap_orphans
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


def _model_trainer_defaults(cfg: DictConfig) -> dict:
    """Trainer kwargs a model config supplies: the gradient clip it was
    published with (pi0.5: openpi's 1.0), unless ``trainer.gradient_clip_val``
    is set explicitly."""
    clip = cfg.model.get("gradient_clip_val")
    if clip is None or cfg.trainer.get("gradient_clip_val") is not None:
        return {}
    return {"gradient_clip_val": clip}


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
    checkpoint keeps the launch-time ``ckpt_path`` (warning).

    ``last.ckpt`` is a symlink (``save_last: link``), so it can dangle (target
    deleted) or be missing while checkpoints exist (``rsync`` without ``-l``, a
    preemption between Lightning's unlink and relink). Both raise: falling back
    would silently restart a run that has checkpoints."""
    requeue = _requeue_resume_path(cfg)
    if requeue is None:
        return
    if os.path.isfile(requeue):
        log.info(f"Detected SLURM requeue — resuming from {requeue}")
        cfg.ckpt_path = requeue
        return
    if os.path.islink(requeue):
        raise FileNotFoundError(
            f"SLURM requeue: {requeue} links to {os.readlink(requeue)!r}, which "
            "does not exist. Refusing to restart a run that has checkpoints; "
            "restore the target or relink last.ckpt to an existing checkpoint."
        )
    ckpt_dir = os.path.dirname(requeue)
    existing = [
        os.path.join(ckpt_dir, f)
        for f in (os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else [])
        if f.endswith(".ckpt")
    ]
    if existing:
        newest = max(existing, key=os.path.getmtime)
        raise FileNotFoundError(
            f"SLURM requeue: {requeue} is missing but {ckpt_dir} holds "
            f"{len(existing)} checkpoint(s), newest {newest}. Refusing to restart "
            f"from scratch; link it with `ln -s {os.path.basename(newest)} "
            f"{requeue}` and requeue."
        )
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


def _log_dataset_frame_counts(
    train_datasets: dict,
    valid_datasets: dict,
    unseen_op_valid_datasets: dict | None = None,
) -> None:
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
    for name, ds in (unseen_op_valid_datasets or {}).items():
        rows.append(("unseen_op_valid", name, len(ds)))
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


def _resolve_mode(cfg: DictConfig) -> str:
    """``mode`` key, else the legacy ``train`` / ``eval`` booleans."""
    if cfg.get("mode") is not None:
        return cfg.mode
    if cfg.get("train", False):
        return "train"
    if cfg.get("eval", False):
        return "eval"
    raise ValueError("Config must specify either `mode` or `train`/`eval` booleans")


def _train_viz_enabled(cfg: DictConfig) -> bool:
    """The train_viz head (metrics + video on TRAIN data each validation) is on
    by default for training runs with an evaluator; ``train_viz=false`` turns
    it off, including the heads a data/top-level config declares itself."""
    return (
        bool(cfg.get("train_viz", True))
        and _resolve_mode(cfg) == "train"
        and cfg.get("evaluator") is not None
    )


def _train_viz_datasets(cfg: DictConfig, train_datasets: dict, instantiate):
    """Datasets for the train_viz loader and, when they are derived here, its
    loader params (None = leave ``data.train_viz_dataloader_params`` to Hydra).

    A data config's own ``train_viz_datasets`` win. Otherwise the train split
    itself is reused (same dataset objects, so no second resolve) through an
    unshuffled loader with the valid loader's params."""
    if not _train_viz_enabled(cfg):
        return {}, None
    explicit = cfg.data.get("train_viz_datasets")
    if explicit is not None:
        return {
            name: None if node is None else instantiate(node, dataset_name=name)
            for name, node in explicit.items()
        }, None
    valid_params = cfg.data.get("valid_dataloader_params") or {}
    params = {
        name: {**OmegaConf.to_container(valid_params[name]), "shuffle": False}
        for name in train_datasets
        if name in valid_params
    }
    return {name: train_datasets[name] for name in params}, params


def _build_train_viz_evaluator(cfg: DictConfig):
    """``train_viz_evaluator`` if configured, else a TrainVizEvalVideo around a
    second instance of the canonical evaluator (own frame buffers)."""
    if not _train_viz_enabled(cfg):
        return None
    if cfg.get("train_viz_evaluator") is not None:
        return hydra.utils.instantiate(cfg.train_viz_evaluator)
    from egomimic.eval.eval_train_viz import TrainVizEvalVideo

    return TrainVizEvalVideo(hydra.utils.instantiate(cfg.evaluator))


def _unseen_op_valid_datasets(cfg: DictConfig, instantiate) -> dict:
    """Datasets for the third (unseen_op_valid) val loader: a data config's own
    ``unseen_op_valid_datasets``, in training runs with an evaluator only (eval
    mode validates ``valid_datasets`` alone)."""
    explicit = cfg.data.get("unseen_op_valid_datasets")
    if (
        explicit is None
        or _resolve_mode(cfg) != "train"
        or cfg.get("evaluator") is None
    ):
        return {}
    return {
        name: None if node is None else instantiate(node, dataset_name=name)
        for name, node in explicit.items()
    }


def _trainer_world_size(cfg: DictConfig) -> int:
    """``devices * num_nodes`` as the config declares it, 1 if it cannot say.

    DistributedSampler strides rather than chunks, so W ranks between them walk
    the WHOLE split; the per-rank ``limit_val_batches`` window is therefore W
    times wider than it looks from one rank. Eval mode runs on one device
    (``train`` forces ``devices=1`` after the loaders are built), so it is 1.
    """
    legacy = "train" if cfg.get("train") else "eval" if cfg.get("eval") else None
    if (cfg.get("mode") or legacy) == "eval":  # _resolve_mode, minus its raise
        return 1
    devices = cfg.get("trainer", {}).get("devices", 1)
    if isinstance(devices, (list, ListConfig)):
        n = len(devices)
    elif isinstance(devices, int) and devices > 0:
        n = devices
    else:  # "auto", -1: resolved by lightning at runtime, unknown here
        log.warning(
            f"trainer.devices={devices!r} is not a count; treating the metric "
            "loaders as single-rank, which under-scores the val split."
        )
        n = 1
    nodes = cfg.get("trainer", {}).get("num_nodes", 1)
    return n * (int(nodes) if isinstance(nodes, int) and nodes > 0 else 1)


def _metric_frames_per_episode(
    cfg: DictConfig,
    head: str,
    n_episodes: int | None = None,
    batch_size: int | None = None,
) -> int | None:
    """Frames per episode to keep on this val head, or None for no subsampling.

    ``data.metric_frames_per_episode[head]`` is written for ONE rank -- it is
    ``floor(limit_val_batches * batch_size / n_episodes)`` -- so it is scaled by
    the world size here: the ranks stride through the subsampled set together
    and each still reads at most ``limit_val_batches`` batches. Without this a
    4-GPU run scores a quarter of the frames the same config scores on 1 GPU,
    which on the seen-val head is fewer than it scored before subsampling
    existed at all. EvenStrideDataset keeps a whole episode when K exceeds its
    length, so a split that fits entirely is not subsampled.

    ``auto`` computes that formula from the resolved split (``n_episodes``) and
    the head's loader ``batch_size``, for splits defined by live SQL filters
    whose episode count a hard-coded K would silently fall behind. With
    ``limit_val_batches=0`` (validation off) it subsamples nothing.
    """
    table = cfg.data.get("metric_frames_per_episode")
    if table is None:
        return None
    k = table.get(head)
    if k is None:
        return None
    if k == "auto":
        limit = cfg.get("trainer", {}).get("limit_val_batches")
        if limit == 0:  # validation is off; nothing to budget
            return None
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or not batch_size
            or not n_episodes
        ):
            raise ValueError(
                f"data.metric_frames_per_episode.{head}=auto needs an int "
                "trainer.limit_val_batches, the head's loader batch_size and a "
                f"resolved split; got limit_val_batches={limit!r}, "
                f"batch_size={batch_size!r}, episodes={n_episodes!r}"
            )
        k = max(1, limit * int(batch_size) // n_episodes)
    else:
        k = int(k)
        if k <= 0:
            raise ValueError(
                f"data.metric_frames_per_episode.{head} must be a positive int "
                f"or 'auto', got {k}"
            )
    return k * _trainer_world_size(cfg)


def _subsample_val_datasets(
    cfg: DictConfig, head: str, datasets: dict, loader_params=None
) -> dict:
    """Wrap each of this val head's datasets in ``EvenStrideDataset`` so the
    ``limit_val_batches`` window spans EVERY episode of the split.

    The val loaders are unshuffled (so the overlay video is coherent), which
    made the metric window the first ``limit_val_batches * batch_size * W``
    frames in episode-hash order: at W = 1 that is a third of the seen-val
    split and effectively one operator of the unseen split (lane-0 coverage
    report, 2026-09-15, measured single-rank). K evenly spaced frames per
    episode gives every episode -- and so every operator -- equal weight at
    the same batch count.

    ``loader_params`` is this head's ``{dataset_name: DataLoader kwargs}``,
    read only for an ``auto`` K. No ``metric_frames_per_episode`` entry for this
    head => datasets pass through untouched."""
    table = cfg.data.get("metric_frames_per_episode")
    if table is None or table.get(head) is None or not datasets:
        return datasets
    wrapped = {}
    for name, ds in datasets.items():
        if ds is None:
            wrapped[name] = None
            continue
        params = (loader_params or {}).get(name) or {}
        k = _metric_frames_per_episode(
            cfg, head, n_episodes=len(ds.datasets), batch_size=params.get("batch_size")
        )
        if k is None:
            wrapped[name] = ds
            continue
        sub = EvenStrideDataset(ds, frames_per_episode=k)
        log.info(
            f"val head '{head}' dataset '{name}': EvenStrideDataset "
            f"frames_per_episode={k} -> {len(sub)} / {len(ds)} frames over "
            f"{len(ds.datasets)} episodes"
        )
        wrapped[name] = sub
    return wrapped


def _video_datasets(cfg: DictConfig, heads: dict) -> dict:
    """``{head: {dataset_name: MultiDataset}}`` for the video-only val loaders.

    ``data.video_episodes[head]`` pins one episode hash per operator; each
    pinned episode is taken out of that head's ALREADY resolved split (no second
    SQL pull or path resolution -- see ``pinned_episode_subset``), contiguous
    and unsubsampled so the overlay video stays watchable.

    A head with no pins gets no video loader, so a config without the key keeps
    today's behaviour (metrics AND video on the one metric loader)."""
    table = cfg.data.get("video_episodes")
    if table is None:
        return {}
    out: dict = {}
    for head, datasets in heads.items():
        pins = [str(h) for h in (table.get(head) or [])]
        if not pins or not datasets:
            continue
        live = {name: ds for name, ds in datasets.items() if ds is not None}
        unplaced = sorted(set(pins) - {h for ds in live.values() for h in ds.datasets})
        if unplaced:
            raise PinError(
                f"{len(unplaced)} pinned video episode(s) for head '{head}' are in "
                f"none of its datasets {sorted(live)}: {unplaced}. The pin must "
                "name an episode of THIS head's split."
            )
        built = {}
        for name, ds in live.items():
            mine = [h for h in pins if h in ds.datasets]
            if not mine:
                continue
            built[name] = pinned_episode_subset(ds, mine, dataset_name=name)
            log.info(
                f"val head '{head}' video loader from dataset '{name}': "
                f"{len(mine)} pinned episode(s), {len(built[name])} frames"
            )
        out[head] = built
    return out


def _require_capped_video_heads(model, datamodule) -> None:
    """A pinned video loader renders on rank 0 alone while the other ranks wait
    in the val reduction, so an uncapped render (``viz_max_batches=None``) is
    bounded only by the 30-minute process-group timeout."""
    heads = model._val_heads()
    for head, datasets in (getattr(datamodule, "video_datasets", None) or {}).items():
        evaluator = heads.get(head)
        if (
            datasets
            and getattr(evaluator, "viz_func", None) is not None
            and evaluator.viz_max_batches is None
        ):
            raise ValueError(
                f"val head '{head}' has pinned video episodes "
                f"(data.video_episodes.{head}) but its evaluator sets no "
                "viz_max_batches: rank 0 would render every video batch while "
                "the other ranks wait on it. Set evaluator.viz_max_batches."
            )


def _build_unseen_op_valid_evaluator(cfg: DictConfig):
    """The canonical evaluator (fresh instance, own frame buffers) wrapped to
    log ``unseen_op_valid/`` and write ``videos_unseen_op_valid/``."""
    from egomimic.eval.eval_train_viz import TrainVizEvalVideo

    return TrainVizEvalVideo(
        hydra.utils.instantiate(cfg.evaluator), prefix="unseen_op_valid"
    )


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

    # One SQL pull / path resolution per dataset spec across train, valid,
    # train_viz and the norm-stat copies; dropped on exit so nothing outlives
    # this run.
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

        train_viz_datasets, train_viz_params = _train_viz_datasets(
            cfg, train_datasets, instantiate=_instantiate_dataset
        )
        unseen_op_valid_datasets = _unseen_op_valid_datasets(
            cfg, instantiate=_instantiate_dataset
        )

        # Split the val heads in two (lane A): a per-episode SUBSAMPLED metric
        # loader that covers the whole split inside limit_val_batches, and a
        # contiguous video-only loader over the pinned episodes. Build the video
        # subsets from the full splits BEFORE wrapping, and let each head opt in
        # independently (no keys in the data config => today's behaviour).
        video_datasets = _video_datasets(
            cfg,
            {
                "valid": valid_datasets,
                "train_viz": train_viz_datasets,
                "unseen_op_valid": unseen_op_valid_datasets,
            },
        )
        valid_datasets = _subsample_val_datasets(
            cfg,
            "valid",
            valid_datasets,
            cfg.data.get("valid_dataloader_params"),
        )
        train_viz_datasets = _subsample_val_datasets(
            cfg,
            "train_viz",
            train_viz_datasets,
            train_viz_params
            if train_viz_params is not None
            else cfg.data.get("train_viz_dataloader_params"),
        )
        unseen_op_valid_datasets = _subsample_val_datasets(
            cfg,
            "unseen_op_valid",
            unseen_op_valid_datasets,
            cfg.data.get("unseen_op_valid_dataloader_params"),
        )

        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        assert (
            "MultiDataModuleWrapper" in cfg.data._target_
        ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
        datamodule_kwargs = dict(
            train_datasets=train_datasets, valid_datasets=valid_datasets
        )
        # Always passed, so train_viz=false also drops a data config's own
        # train_viz_datasets.
        datamodule_kwargs["train_viz_datasets"] = train_viz_datasets
        if train_viz_params is not None:
            datamodule_kwargs["train_viz_dataloader_params"] = train_viz_params
        # Always passed too: hydra would otherwise instantiate the raw
        # config nodes itself (eval mode included).
        datamodule_kwargs["unseen_op_valid_datasets"] = unseen_op_valid_datasets
        # Built above; hydra has nothing to instantiate for these.
        datamodule_kwargs["video_datasets"] = video_datasets
        datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg.data, **datamodule_kwargs
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
        pool_horizon = bool(
            OmegaConf.select(cfg, "norm_stats.pool_horizon", default=False)
        )
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
            if "proprio_history" in km:
                # Current-step proprio stats even on a history (K > 1) config:
                # every history step is normalized with the CURRENT-step,
                # per-channel (D,) stats, so a K > 1 run and the K = 1 baseline
                # share exactly one set of stats. Same rule as
                # scripts/precompute_norm_stats.py.
                km["proprio_history"] = 1

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
                    pool_horizon,
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
                pool_horizon=pool_horizon,
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
    for ds in getattr(datamodule, "train_viz_datasets", {}).values():
        ds.set_norm_stats_from(norm_stats)
    for ds in getattr(datamodule, "unseen_op_valid_datasets", {}).values():
        ds.set_norm_stats_from(norm_stats)
    # The video subsets share leaves with their head's split but are separate
    # MultiDatasets, so they need their own wiring (EvenStrideDataset forwards
    # the call to the base it indexes into).
    for head_datasets in getattr(datamodule, "video_datasets", {}).values():
        for ds in head_datasets.values():
            ds.set_norm_stats_from(norm_stats)

    _prepare_checkpoint_resume(cfg)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        norm_stats_state=norm_stats.to_state(),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
        enable_grad_norm=cfg.model.get("enable_grad_norm", True),
    )

    _log_dataset_frame_counts(
        datamodule.train_datasets,
        datamodule.valid_datasets,
        datamodule.unseen_op_valid_datasets,
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    mode = _resolve_mode(cfg)

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
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins or None,
        **_model_trainer_defaults(cfg),
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
            # data.valid_prefix (e.g. seen_op_valid in the opsplit configs)
            # renames the canonical head: `<prefix>/Valid/...` metrics and
            # `videos_<prefix>/` instead of `Valid/...` and `videos/`.
            valid_prefix = cfg.data.get("valid_prefix")
            if valid_prefix:
                from egomimic.eval.eval_train_viz import TrainVizEvalVideo

                eval_obj = TrainVizEvalVideo(eval_obj, prefix=valid_prefix)
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            model.evaluator = eval_obj
        train_viz_eval_obj = (
            _build_train_viz_evaluator(cfg) if datamodule.train_viz_datasets else None
        )
        if train_viz_eval_obj is not None:
            train_viz_eval_obj.trainer = trainer
            train_viz_eval_obj.model = model.model
            model.train_viz_evaluator = train_viz_eval_obj
        if datamodule.unseen_op_valid_datasets:
            unseen_eval_obj = _build_unseen_op_valid_evaluator(cfg)
            unseen_eval_obj.trainer = trainer
            unseen_eval_obj.model = model.model
            model.unseen_op_valid_evaluator = unseen_eval_obj
        _require_capped_video_heads(model, datamodule)
        model.val_loader_names = datamodule.val_loader_names()
        # Pre-fit baseline val. Skipped on requeues AND checkpoint resumes:
        # trainer.validate here runs BEFORE fit restores ckpt_path weights, so
        # on a resume it would score the un-resumed base model.
        # (_prepare_checkpoint_resume already folded a requeue into ckpt_path.)
        if cfg.get("val_at_start", False) and not cfg.get("ckpt_path"):
            log.info(
                "val_at_start: running validation at epoch 0 (pre-fit baseline; "
                "the overlay video follows evaluator.viz_every_n_epochs)"
            )
            trainer.validate(model=model, datamodule=datamodule)
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
        # Without the names, loader indices fall back to [valid, train_viz] and
        # the pinned video loader is misrouted.
        model.val_loader_names = datamodule.val_loader_names()

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

    # Before CUDA init: a cancelled job can leave dataloader workers -- or whole
    # ranks -- holding GPU contexts, and nothing else on this cluster sweeps them.
    reap_orphans()
    arm_rank_pdeathsig()

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
