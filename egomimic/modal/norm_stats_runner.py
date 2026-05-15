"""Subprocess entry-point for offline norm-stats computation.

Called by offline_norm_stats.py after _prepare_repo(), mirroring how
run.py calls trainHydra.py. Runs in a fresh process so the editable
egomimic install is visible from the start.

Usage (internal — not meant to be called directly):
    python3 norm_stats_runner.py <data_config> \\
        --num_workers <N> --sample_frac <F> --output_path <path>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.rldb.zarr.utils import DataSchematic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_config")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    load_env()
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    repo_root = Path(__file__).resolve().parent.parent.parent
    config_dir = str(repo_root / "egomimic" / "hydra_configs")

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train_zarr_cartesian.yaml",
            overrides=[f"data={args.data_config}"],
        )

    # Disable debug limits — norm stats must cover the full dataset
    with open_dict(cfg):
        for ds_name in list(cfg.data.train_datasets):
            resolver = OmegaConf.select(
                cfg.data.train_datasets[ds_name], "resolver", default=None
            )
            if resolver is not None:
                cfg.data.train_datasets[ds_name].resolver.debug = False

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    for dataset_name in cfg.data.train_datasets:
        print(f"[NormStats] Instantiating dataset <{dataset_name}>")
        dataset = hydra.utils.instantiate(cfg.data.train_datasets[dataset_name])
        data_schematic.infer_shapes_from_batch(dataset[0])

        norm_cfg = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        km = OmegaConf.to_container(norm_cfg.resolver.key_map, resolve=False)
        km["norm_mode"] = True
        norm_cfg.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(norm_cfg)

        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=args.sample_frac,
            num_workers=args.num_workers,
        )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats_out: dict = {}
    for emb, keys_dict in data_schematic.norm_stats.items():
        stats_out[str(emb)] = {
            key_name: {
                stat_name: np.asarray(arr).tolist()
                for stat_name, arr in stat_dict.items()
            }
            for key_name, stat_dict in keys_dict.items()
        }

    payload: dict = {
        "stats": stats_out,
        "loading_time": None,
        "computing_time": None,
        "frames": None,
    }
    if data_schematic._norm_run_metadata is not None:
        for k in ("loading_time", "computing_time", "frames"):
            if k in data_schematic._norm_run_metadata:
                payload[k] = data_schematic._norm_run_metadata[k]

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)

    print(f"[NormStats] Saved to {out_path}")


if __name__ == "__main__":
    main()
