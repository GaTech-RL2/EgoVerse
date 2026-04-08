"""Compute and cache normalization statistics for the configured datasets.

Reuses the Hydra config tree from ``trainHydra.py`` but only runs the data
schematic + norm-stat portion of the pipeline — no model, no trainer, no GPU.
Intended to be launched via the Hydra submitit launcher on PACE (typically the
CPU-only ``submitit_pace_cpu`` launcher).
"""

import copy
from typing import Optional

import hydra
import lightning as L
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf

from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras

OmegaConf.register_new_resolver(
    "eval", eval, replace=True
)  # idempotent if trainHydra was imported elsewhere
log = RankedLogger(__name__, rank_zero_only=True)


def compute_norm_stats(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    # Norm stats are only computed over train datasets; skip valid to save time.
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    assert (
        "MultiDataModuleWrapper" in cfg.data._target_
    ), "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, train_datasets=train_datasets, valid_datasets={}
    )

    save_cache_dir = OmegaConf.select(cfg, "norm_stats.save_cache_dir", default=None)

    for dataset_name, dataset in datamodule.train_datasets.items():
        if not isinstance(dataset, MultiDataset):
            raise ValueError(
                f"{dataset_name} is not a MultiDataset. All top level datasets in data config should be MultiDataset"
            )

        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        data_schematic.infer_shapes_from_batch(dataset[0])

        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)

        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
            checkpoint_every_frac=OmegaConf.select(
                cfg, "norm_stats.checkpoint_every_frac", default=None
            ),
            checkpoint_save_dir=save_cache_dir,
        )

        if save_cache_dir:
            data_schematic.cache_stats(save_cache_dir=save_cache_dir)
            log.info(
                f"[NormStats] Cached stats for <{dataset_name}> under {save_cache_dir}"
            )

    if not save_cache_dir:
        log.warning(
            "norm_stats.save_cache_dir is not set — computed stats were NOT written to disk."
        )


@hydra.main(
    version_base="1.3",
    config_path="./hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    print(OmegaConf.to_yaml(cfg))
    compute_norm_stats(cfg)


if __name__ == "__main__":
    main()
