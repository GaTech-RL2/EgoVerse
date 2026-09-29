"""Precompute normalization stats on a CPU node, so GPU-node time isn't spent
computing them at training startup.

Runs trainHydra's own norm-stats step (``trainHydra.compute_norm_stats``, same
hydra config + data config) but builds NO model and NO trainer: it only
instantiates the train datasets (which s5cmd-syncs any missing episodes from
S3 as a side effect), infers shapes, computes norm stats, and stores them
twice:

  1. an explicit file, ``<out>/norm_stats/norm_stats.json``, for
     ``norm_stats.precomputed_norm_path=<out>/norm_stats``;
  2. the content-keyed cache under ``norm_stats.cache_dir`` (default
     ``paths.cache_dir``), so a plain training run on the same episodes +
     recipe hits the cache without any override. An entry already there is
     reused, not recomputed.

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
import os

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import egomimic
from egomimic.trainHydra import compute_norm_stats
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
        help="extra hydra overrides; the flags above win over them (e.g. "
        "paths.dataset_dir=/path/to/zarr/mirror)",
    )
    args = ap.parse_args()

    cfg_dir = os.path.join(os.path.dirname(egomimic.__file__), "hydra_configs")
    GlobalHydra.instance().clear()
    # The extra overrides go before the script's own flags, which stay
    # authoritative: a trailing norm_stats.save_cache_dir=null would otherwise
    # silently skip the explicit file this script promises.
    overrides = [
        f"data={args.data}",
        f"model={args.model}",
        "seed=42",
        *args.overrides,
        f"norm_stats.sample_frac={args.sample_frac}",
        f"norm_stats.max_samples={'null' if args.max_samples is None else args.max_samples}",
        f"norm_stats.num_workers={args.num_workers}",
        f"norm_stats.save_cache_dir={args.out}",
        "norm_stats.precomputed_norm_path=null",
        f"norm_stats.use_cache={str(not args.no_cache).lower()}",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    import lightning as L

    L.seed_everything(cfg.seed, workers=True)
    load_env()

    # Instantiating syncs any missing episodes from S3.
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        if cfg.data.train_datasets[dataset_name] is None:
            continue
        print(f"[precompute] dataset={dataset_name}: instantiating (syncs S3) ...")
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name], dataset_name=dataset_name
        )

    # Writes <out>/norm_stats/norm_stats.json (norm_stats.save_cache_dir) and,
    # unless --no-cache, the content-keyed cache entry.
    compute_norm_stats(cfg, train_datasets)

    out_dir = os.path.join(args.out, "norm_stats")
    print("\nDONE. Use this in training:")
    print(f"  norm_stats.precomputed_norm_path={out_dir}")
    if not args.no_cache:
        print(
            f"(or nothing: the content-keyed cache under {cfg.norm_stats.cache_dir} "
            "now holds these stats for the same episodes + recipe)"
        )


if __name__ == "__main__":
    main()
