"""Compute per-embodiment norm stats offline (CPU) and save norm_stats.json.

Replicates the stats block of ``trainHydra`` (MultiDataset.populate_from_datasets
-> infer_norm_from_dataset per train dataset -> cache_stats) over a standalone
data config, so a RICL run can point BOTH the query path
(``norm_stats.precomputed_norm_path``) and the bank provider
(``data.bank_norm_path``) at the same file. This is also the only way to get
stats for a bank embodiment the training run never loads as a dataloader
(e.g. aria for D1 — include an aria dataset in the stats config).

The config must be *standalone-composable*: ``load_config_from_path`` resolves
its ``defaults:`` chain but there is no root ``paths`` group, so resolver
``folder_path`` must be absolute (see ``data/ricl_stats_d0.yaml``).

Usage:
  python -m egomimic.ricl.scripts.compute_norm_stats \\
      --data-config egomimic/hydra_configs/data/ricl_stats_d0.yaml \\
      --out ricl_artifacts/stats_d0

Writes ``<out>/norm_stats/norm_stats.json``.
"""

from __future__ import annotations

import argparse
import copy
import logging

import hydra
from omegaconf import OmegaConf

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.hydra_utils import load_config_from_path

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-config",
        required=True,
        help="standalone data config with a train_datasets dict (absolute "
        "resolver folder_path; no ${paths.*} interpolations)",
    )
    ap.add_argument("--out", required=True, help="dir for norm_stats/norm_stats.json")
    ap.add_argument("--norm-mode", default="quantile")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    cfg = load_config_from_path(args.data_config)

    train_datasets = {}
    for name in cfg.train_datasets:
        if cfg.train_datasets[name] is None:
            continue
        logger.info("Instantiating dataset <%s>", name)
        train_datasets[name] = hydra.utils.instantiate(cfg.train_datasets[name])

    if not train_datasets:
        raise SystemExit(f"{args.data_config} defines no train_datasets")

    norm_stats = MultiDataset(state={}, norm_mode=args.norm_mode)
    norm_stats.populate_from_datasets(train_datasets)

    # Mirror trainHydra: per dataset, re-instantiate a norm-mode copy (keymap
    # with norm_mode=True strips image/annotation keys -> cheap CPU iteration)
    # and accumulate stats into the shared ``norm_stats``.
    for name, dataset in train_datasets.items():
        logger.info("Inferring shapes for dataset <%s>", name)
        norm_stats.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.train_datasets[name])
        km = OmegaConf.to_container(instantiate_copy.resolver.key_map, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            name,
            sample_frac=args.sample_frac,
            num_workers=args.num_workers,
        )

    norm_stats.cache_stats(save_cache_dir=args.out)
    for emb, keys in norm_stats.norm_stats.items():
        for key, stats in keys.items():
            shape = getattr(stats.get("quantile_1"), "shape", None)
            logger.info("stats[%s][%s] quantile_1 shape=%s", emb, key, shape)


if __name__ == "__main__":
    main()
