"""Precompute normalization stats on a CPU node, so GPU-node time isn't spent
computing them at training startup.

Replicates trainHydra's norm loop EXACTLY (same hydra config + data config,
same keymap/transform, same sample_frac/seed) but builds NO model and NO
trainer: it only instantiates the train datasets (which s5cmd-syncs any
missing episodes from S3 as a side effect), infers shapes, computes norm
stats, and stores them twice:

  1. an explicit file, ``<out>/norm_stats/norm_stats.json``, for
     ``norm_stats.precomputed_norm_path=<out>/norm_stats``;
  2. the content-keyed cache under ``norm_stats.cache_dir`` (default
     ``paths.cache_dir``), keyed exactly as trainHydra keys it, so a plain
     training run on the same episodes + recipe hits the cache without any
     override.

The per-episode samples behind it also land in the cache (``episodes/``), so a
later split of the same recipe (a subset, a superset, an operator hold-out)
only samples the episodes it adds.

The norm-mode keymap strips camera + annotation keys, so the stats pass reads
only the numeric proprio/action arrays: pure CPU work, no GPU, no JPEG decode
beyond one shape-inference sample.

Usage (CPU node, repo root, emimic venv):
    python egomimic/scripts/precompute_norm_stats.py \\
        --data mecka_all_6d --model pi0.5_bc_mecka_6d \\
        --sample-frac 0.1 --num-workers 30 --out /path/to/norm_stats/mecka_all_6d
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import egomimic
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr import episode_norm_samples, norm_cache
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.env import load_env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="train_zarr_cartesian_pi")
    ap.add_argument(
        "--data",
        required=True,
        help="data config group; MUST match the training run's data config",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="only needed so the config composes; the model is never built",
    )
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional hard cap on collected samples: the (N, 100, D) float32 "
        "action stack plus np.percentile's sort copy is ~2 x N x 100 x D x 4 "
        "bytes of RAM, so an uncapped 0.1 frac of the full mecka set (~8.5M "
        "frames) needs >100GB at D=18. Part of the cache key: a training run "
        "hits this entry only with the same norm_stats.max_samples.",
    )
    ap.add_argument("--num-workers", type=int, default=30)
    ap.add_argument(
        "--out",
        required=True,
        help="save_cache_dir; writes <out>/norm_stats/norm_stats.json",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="only write the explicit file; skip the content-keyed cache entry",
    )
    ap.add_argument(
        "overrides",
        nargs="*",
        help="extra hydra overrides appended verbatim (e.g. "
        "paths.dataset_dir=/path/to/zarr/mirror)",
    )
    args = ap.parse_args()

    cfg_dir = os.path.join(os.path.dirname(egomimic.__file__), "hydra_configs")
    GlobalHydra.instance().clear()
    overrides = [
        f"data={args.data}",
        f"model={args.model}",
        f"norm_stats.sample_frac={args.sample_frac}",
        f"norm_stats.max_samples={'null' if args.max_samples is None else args.max_samples}",
        f"norm_stats.num_workers={args.num_workers}",
        f"norm_stats.save_cache_dir={args.out}",
        "norm_stats.precomputed_norm_path=null",
        "seed=42",
        *args.overrides,
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    import lightning as L

    L.seed_everything(cfg.seed, workers=True)
    load_env()

    norm_mode = OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile")
    pool_horizon = bool(OmegaConf.select(cfg, "norm_stats.pool_horizon", default=False))
    cache_dir = None if args.no_cache else OmegaConf.select(cfg, "norm_stats.cache_dir")

    # Mirrors trainHydra: instantiate train datasets (resolver syncs from S3
    # as needed), then a stats-only MultiDataset computes the norm stats from
    # a norm-mode (numerics-only) copy of each dataset.
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        if cfg.data.train_datasets[dataset_name] is None:
            continue
        print(f"[precompute] dataset={dataset_name}: instantiating (syncs S3) ...")
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name], dataset_name=dataset_name
        )

    norm_stats = MultiDataset(state={}, norm_mode=norm_mode)
    norm_stats.populate_from_datasets(train_datasets)

    for dataset_name, dataset in train_datasets.items():
        print(f"[precompute] dataset={dataset_name}: inferring shapes ...")
        norm_stats.infer_shapes_from_batch(dataset[0])

        inst = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        km = OmegaConf.to_container(inst.resolver.key_map, resolve=False)
        km["norm_mode"] = True  # strips image + annotation keys
        if "proprio_history" in km:
            # Current-step proprio stats even on a history (K > 1) config: every
            # history step is normalized with the CURRENT-step stats, so a K > 1
            # run and the K = 1 baseline share one norm-stat file. Reading one
            # frame is exactly `[..., -1, :]` of the window, minus the K-fold
            # read.
            km["proprio_history"] = 1
        inst.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(inst, dataset_name=dataset_name)

        t0 = time.perf_counter()
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=args.sample_frac,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            precomputed_norm_path=None,  # force compute
            pool_horizon=pool_horizon,
            episode_cache=None
            if not cache_dir
            else episode_norm_samples.cache_root(
                cache_dir, dataset_name, cfg.data.train_datasets[dataset_name]
            ),
        )
        print(
            f"[precompute] {dataset_name}: norm computed in "
            f"{time.perf_counter() - t0:.1f}s"
        )

        if cache_dir:
            # Same key trainHydra derives, so its next run is a cache hit.
            episodes = {
                h: norm_cache.episode_fingerprint(getattr(ds, "episode_path", None))
                for h, ds in dataset.datasets.items()
            }
            inputs = norm_cache.cache_inputs(
                dataset_name,
                episodes,
                cfg.data.train_datasets[dataset_name],
                args.sample_frac,
                pool_horizon,
                args.max_samples,
            )
            key = norm_cache.norm_cache_key(inputs)
            emb = get_embodiment_id(dataset_name)
            written = norm_cache.write_cached(
                cache_dir,
                dataset_name,
                key,
                inputs,
                emb,
                norm_stats.norm_stats[emb],
                norm_stats._norm_run_metadata,
            )
            print(f"[precompute] {dataset_name}: cache entry {written}")

    norm_stats.cache_stats(save_cache_dir=args.out)
    out_dir = os.path.join(args.out, "norm_stats")
    print("\nDONE. Use this in training:")
    print(f"  norm_stats.precomputed_norm_path={out_dir}")
    if cache_dir:
        print(
            f"(or nothing: the content-keyed cache under {cache_dir} now holds "
            "these stats for the same episodes + recipe)"
        )


if __name__ == "__main__":
    main()
