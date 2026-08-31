"""
Offline val-metric sweep across a folder of checkpoints.

Given a training run's ``checkpoints/`` folder (e.g. the mecka+eva or aria+eva
cotrain), enumerate every ``epoch_epoch=*.ckpt``, run a validation pass on the
training run's OWN val split (same seed-split via ``_force_ood_split``), and
log the resulting metrics to a fresh wandb run keyed by epoch. Uses the same
TF rolling (``_sample_rolling_tf``) that offline eval uses, and the same
longeva pattern (eva ``cam_horizon`` bumped ×3 so eva episodes get their full
length reconstructed).

Design:
  * Instantiate the datamodule + model + norm_stats ONCE (expensive), then loop
    checkpoints and swap in state_dicts. A fresh trainer per ckpt so
    ``trainer.callback_metrics`` doesn't carry across.
  * Wandb logging is direct (``wandb.init`` / ``wandb.log(step=epoch)``) rather
    than through Lightning's ``WandbLogger`` so we control the step axis
    (epoch, not global_step which resets on load).
  * Videos are written by ``WAMEvalVideo`` per ckpt but overwritten each pass —
    the primary artifact of the sweep is wandb metric curves, not the mp4s.

Hydra config: reuses ``train_zarr_human_wam_wan22_5b.yaml`` for structure and
takes these extra keys off the command line:
  * ``checkpoints_dir``  — folder holding ``epoch_epoch=*.ckpt`` files
  * ``num_val_episodes`` — per-embodiment val episodes (default 20)
  * ``wandb_project`` / ``wandb_run_name`` — wandb dest
"""

from __future__ import annotations

import copy
import gc
import glob
import os
import re

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

import egomimic.utils.hydra_resolvers  # noqa: F401  -- registers OmegaConf resolvers
import wandb
from egomimic.eval.eval import Eval
from egomimic.eval.eval_dreamzero import (
    _apply_eval_trainer_overrides,
    _build_model_config_tree,
    _force_ood_split,
    _patch_algo_use_sample_rolling,
    _restrict_to_first_n_episodes,
)
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.pylogger import RankedLogger

OmegaConf.register_new_resolver("eval", eval, replace=True)

log = RankedLogger(__name__, rank_zero_only=True)


def _list_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    """Return sorted ``[(epoch, path), ...]`` for every ``epoch_epoch=N.ckpt``.

    Skips ``last.ckpt`` — it's a duplicate of the highest-epoch checkpoint,
    and gets overwritten by any resume so it's not a stable point in time.
    """
    pattern = os.path.join(ckpt_dir, "epoch_epoch=*.ckpt")
    out: list[tuple[int, str]] = []
    for p in glob.glob(pattern):
        m = re.search(r"epoch=(\d+)\.ckpt$", p)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _metrics_to_wandb(metrics: dict) -> dict:
    """Flatten trainer.callback_metrics (Tensors) → floats for wandb."""
    result = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            try:
                result[k] = float(v.detach().cpu().item())
            except Exception:
                continue
        elif isinstance(v, (int, float)):
            result[k] = float(v)
    return result


@hydra.main(
    version_base="1.3",
    config_path="../hydra_configs",
    config_name="train_zarr_human_wam_wan22_5b",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()

    checkpoints_dir = cfg.get("checkpoints_dir")
    if not checkpoints_dir:
        raise ValueError(
            "checkpoints_dir must be provided (path to a training run's "
            "``checkpoints/`` folder holding ``epoch_epoch=*.ckpt`` files)."
        )
    ckpts = _list_checkpoints(checkpoints_dir)
    if not ckpts:
        raise ValueError(f"No ``epoch_epoch=*.ckpt`` files under {checkpoints_dir}")
    log.info(
        f"[val_sweep] Found {len(ckpts)} checkpoints: "
        f"epochs {[e for e, _ in ckpts]}"
    )

    num_val_episodes: int = int(cfg.get("num_val_episodes", 20))
    valid_ratio: float = float(cfg.get("valid_ratio", 0.2))
    valid_mode: str = str(cfg.get("valid_mode", "valid"))
    wandb_project: str = str(cfg.get("wandb_project", "wam"))
    wandb_run_name: str = str(
        cfg.get("wandb_run_name", f"val_sweep_{os.path.basename(checkpoints_dir)}")
    )

    # ---- config surgery: OOD-split so valid_datasets is disjoint from train
    _force_ood_split(cfg, valid_ratio=valid_ratio, valid_mode=valid_mode)

    # ---- datasets --------------------------------------------------------
    train_datasets = {
        name: hydra.utils.instantiate(cfg.data.train_datasets[name])
        for name in cfg.data.train_datasets
    }
    valid_datasets = {
        name: hydra.utils.instantiate(cfg.data.valid_datasets[name])
        for name in cfg.data.valid_datasets
    }

    total_windows = 0
    for name, mds in valid_datasets.items():
        _restrict_to_first_n_episodes(mds, num_val_episodes)
        total_windows += len(mds)
    log.info(
        f"[val_sweep] Sweeping {total_windows} val windows per checkpoint "
        f"(× {num_val_episodes} episodes per dataset)."
    )

    # ---- datamodule ------------------------------------------------------
    assert "MultiDataModuleWrapper" in cfg.data._target_
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
    )

    # ---- norm_stats ------------------------------------------------------
    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats.populate_from_datasets(datamodule.train_datasets)
    for dataset_name, dataset in datamodule.train_datasets.items():
        norm_stats.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        km = OmegaConf.to_container(instantiate_copy.resolver.key_map, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )
    for ds in datamodule.train_datasets.values():
        ds.set_norm_stats_from(norm_stats)
    for ds in datamodule.valid_datasets.values():
        ds.set_norm_stats_from(norm_stats)

    # ---- model (instantiated ONCE — reused across ckpts via load_state_dict)
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        norm_stats_state=norm_stats.to_state(),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    # ---- one-epoch validate-only trainer overrides -----------------------
    _apply_eval_trainer_overrides(cfg, limit_val_batches=total_windows or 1)

    # ---- wandb -----------------------------------------------------------
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "checkpoints_dir": checkpoints_dir,
            "num_val_episodes": num_val_episodes,
            "n_checkpoints": len(ckpts),
            "epochs": [e for e, _ in ckpts],
        },
    )

    # ---- checkpoint loop -------------------------------------------------
    for epoch, ckpt_path in ckpts:
        log.info(f"[val_sweep] === epoch {epoch}: loading {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        log.info(
            f"[val_sweep] load_state_dict: {len(missing)} missing / "
            f"{len(unexpected)} unexpected keys"
        )
        del checkpoint
        gc.collect()

        # Fresh trainer per ckpt so callback_metrics doesn't carry over and
        # WAMEvalVideo's per-eid buffers start empty.
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=None, logger=None
        )
        os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

        eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        model.evaluator = eval_obj

        # Route rolling through the same TF path the offline eval uses so the
        # per-block reconditioning matches what the training val loop should do.
        _patch_algo_use_sample_rolling(model.model, teacher_force=True)

        trainer.validate(model=model, datamodule=datamodule)

        metrics = _metrics_to_wandb(trainer.callback_metrics)
        log.info(f"[val_sweep] epoch {epoch} metrics: {metrics}")
        wandb.log(metrics, step=epoch)

        # Move the videos this ckpt produced into an epoch-scoped subdir so
        # subsequent ckpts don't overwrite them.
        src_videos = os.path.join(trainer.default_root_dir, "videos", "epoch_0")
        dst_videos = os.path.join(
            trainer.default_root_dir, "videos", f"ckpt_epoch_{epoch}"
        )
        if os.path.isdir(src_videos) and not os.path.exists(dst_videos):
            os.rename(src_videos, dst_videos)

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    wandb.finish()
    log.info("[val_sweep] Done.")


if __name__ == "__main__":
    main()
