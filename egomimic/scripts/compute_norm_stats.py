"""Compute and cache norm stats for a data config — no model, no trainer.

Accepts the same Hydra composition as ``trainHydra.py`` (``--config-name``,
``data=``, ``model=``, etc.) so existing run recipes can be reused. The
``model=`` override is harmless — the model is never instantiated.

Writes ``norm_stats/norm_stats.json`` under ``cfg.norm_stats.save_cache_dir``
(defaults to the Hydra run dir). Point a follow-up training run at that
directory via ``norm_stats.precomputed_norm_path=<dir>`` to skip recompute.
"""

import copy
import os
from typing import Optional

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras

OmegaConf.register_new_resolver("eval", eval, replace=True)
log = RankedLogger(__name__, rank_zero_only=True)


def compute_norm_stats(cfg: DictConfig) -> None:
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

    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats.populate_from_datasets(train_datasets)

    save_cache_dir = OmegaConf.select(cfg, "norm_stats.save_cache_dir", default=None)
    if not save_cache_dir:
        raise ValueError(
            "norm_stats.save_cache_dir must be set so the computed stats are persisted."
        )

    for dataset_name, dataset in train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        norm_stats.infer_shapes_from_batch(dataset[0])

        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)

        log.info(f"Computing norm stats for dataset <{dataset_name}>")
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=None,
        )
        norm_stats.cache_stats(save_cache_dir=save_cache_dir)

    out_path = os.path.join(save_cache_dir, "norm_stats", "norm_stats.json")
    log.info(f"Wrote norm stats to: {out_path}")
    log.info(
        f"To reuse, pass: norm_stats.precomputed_norm_path={os.path.join(save_cache_dir, 'norm_stats')}"
    )


@hydra.main(
    version_base="1.3",
    config_path="../hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    print(OmegaConf.to_yaml(cfg))
    compute_norm_stats(cfg)


if __name__ == "__main__":
    main()
