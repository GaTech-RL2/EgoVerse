import copy
import json
import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras

log = RankedLogger(__name__, rank_zero_only=True)


def norm_stats(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    # log.info(f"Instantiating data schematic <{cfg.data_schematic._target_}>")

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    # Modify dataset configs to include `data_schematic` dynamically at runtime
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    percent_list = [0.05, 0.1, 0.2, 0.5, 1.0]
    for dataset_name, dataset in train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        data_schematic.infer_shapes_from_batch(dataset[0])
        # instantiate norm datasets which is same as dataset but with keymap without the image keys
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

        km = {
            k: v
            for k, v in km.items()
            if not (isinstance(v, Mapping) and v.get("key_type") == "camera_keys")
        }

        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        for percent in percent_list:
            data_schematic.infer_norm_from_dataset(
                norm_dataset,
                dataset_name,
                sample_frac=percent,
                benchmark_dir=os.path.join(
                    cfg.trainer.default_root_dir, f"stats_{percent}"
                ),
            )

    df_list = []
    for i, percent in enumerate(percent_list):
        with open(
            os.path.join(
                cfg.trainer.default_root_dir, f"stats_{percent}", "benchmark.json"
            ),
            "r",
        ) as f:
            benchmark_stats = json.load(f)

        frames = benchmark_stats["frames"]
        loading_time = benchmark_stats["loading_time"]
        computing_time = benchmark_stats["computing_time"]
        total_time = loading_time + computing_time
        for embodiment in benchmark_stats["stats"].keys():
            for k in benchmark_stats["stats"][embodiment].keys():
                mean_path = benchmark_stats["stats"][str(embodiment)][k]["mean"]
                std_path = benchmark_stats["stats"][str(embodiment)][k]["std"]
                min_path = benchmark_stats["stats"][str(embodiment)][k]["min"]
                max_path = benchmark_stats["stats"][str(embodiment)][k]["max"]
                median_path = benchmark_stats["stats"][str(embodiment)][k]["median"]
                quantile_1_path = benchmark_stats["stats"][str(embodiment)][k][
                    "quantile_1"
                ]
                quantile_99_path = benchmark_stats["stats"][str(embodiment)][k][
                    "quantile_99"
                ]
                df = pd.DataFrame(
                    [
                        {
                            "percent": percent,
                            "frames": frames,
                            "loading_time": loading_time,
                            "computing_time": computing_time,
                            "total_time": total_time,
                            "embodiment": embodiment,
                            "key": k,
                            "mean_path": mean_path,
                            "std_path": std_path,
                            "min_path": min_path,
                            "max_path": max_path,
                            "median_path": median_path,
                            "quantile_1_path": quantile_1_path,
                            "quantile_99_path": quantile_99_path,
                        }
                    ]
                )
                df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(
        os.path.join(cfg.trainer.default_root_dir, "benchmark_stats.csv"), index=False
    )
    # create dataframe


@hydra.main(
    version_base="1.3", config_path="../../hydra_configs", config_name="train_zarr.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)

    print(OmegaConf.to_yaml(cfg))
    norm_stats(cfg)


if __name__ == "__main__":
    main()
