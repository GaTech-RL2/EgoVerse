"""Precompute normalization stats on a CPU node, so GPU-node time isn't spent
computing them at training startup.

Replicates trainHydra's norm loop EXACTLY (same hydra config + data config,
same keymap/transform, same sample_frac/seed) but builds NO model and NO
trainer — it only instantiates the train datasets (which s5cmd-syncs any
missing episodes from S3 as a side effect), infers shapes, computes norm
stats, and caches them. The norm-mode keymap strips camera + annotation keys,
so the stats pass reads only the numeric proprio/action arrays — pure CPU
work, no GPU, no JPEG decode beyond one shape-inference sample.

Output: <out>/norm_stats/norm_stats.json — point training at it with:
    norm_stats.precomputed_norm_path=<out>/norm_stats

Usage (CPU node, repo root, emimic venv):
    python egomimic/scripts/precompute_norm_stats.py \
        --data mecka_all_pi_6d --model pi0.5_bc_mecka_6d \
        --sample-frac 0.1 --num-workers 30 \
        --out /storage/project/r-dxu345-0/agao81/norm_stats/mecka_all_6d
"""

import argparse
import copy
import os
import time

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import egomimic
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="train_zarr_cartesian_pi")
    ap.add_argument(
        "--data",
        default="mecka_all_pi_6d",
        help="data config group — MUST match the training run's data config",
    )
    ap.add_argument(
        "--model",
        default="pi0.5_bc_mecka_6d",
        help="only needed so the config composes; the model is never built",
    )
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=2_000_000,
        help="hard cap on collected samples: the (N, 100, 18) float32 action "
        "stack plus np.percentile's sort copy is ~2 x N x 7.2KB of RAM, so an "
        "uncapped 0.1 frac of the full mecka set (~8.5M) needs ~125GB. 2M "
        "samples (~15GB stack) still gives 2M draws per (step, dim) cell.",
    )
    ap.add_argument("--num-workers", type=int, default=30)
    ap.add_argument(
        "--out",
        required=True,
        help="save_cache_dir; writes <out>/norm_stats/norm_stats.json",
    )
    args = ap.parse_args()

    cfg_dir = os.path.join(os.path.dirname(egomimic.__file__), "hydra_configs")
    GlobalHydra.instance().clear()
    overrides = [
        f"data={args.data}",
        f"model={args.model}",
        f"norm_stats.sample_frac={args.sample_frac}",
        f"norm_stats.num_workers={args.num_workers}",
        f"norm_stats.save_cache_dir={args.out}",
        "norm_stats.precomputed_norm_path=null",
        "seed=42",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    import lightning as L

    L.seed_everything(cfg.seed, workers=True)

    # Mirrors trainHydra: instantiate train datasets (resolver syncs from S3
    # as needed), then a stats-only MultiDataset computes the norm stats from
    # a norm-mode (numerics-only) copy of each dataset.
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        print(f"[precompute] dataset={dataset_name}: instantiating (syncs S3) ...")
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats.populate_from_datasets(train_datasets)

    for dataset_name, dataset in train_datasets.items():
        print(f"[precompute] dataset={dataset_name}: inferring shapes ...")
        norm_stats.infer_shapes_from_batch(dataset[0])

        inst = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        km = OmegaConf.to_container(inst.resolver.key_map, resolve=False)
        km["norm_mode"] = True  # strips image + annotation keys
        inst.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(inst)

        t0 = time.perf_counter()
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=args.sample_frac,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            precomputed_norm_path=None,  # force compute
        )
        print(
            f"[precompute] {dataset_name}: norm computed in "
            f"{time.perf_counter() - t0:.1f}s"
        )

    norm_stats.cache_stats(save_cache_dir=args.out)
    out_dir = os.path.join(args.out, "norm_stats")
    print("\nDONE. Use this in training:")
    print(f"  norm_stats.precomputed_norm_path={out_dir}")


if __name__ == "__main__":
    main()
