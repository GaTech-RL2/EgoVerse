"""Canonical Hydra entry point for offline policy evaluation."""

from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from egomimic.trainHydra import train
from egomimic.utils.utils import extras


@hydra.main(
    version_base="1.3",
    config_path="./hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Evaluate a checkpoint through its configured evaluator."""
    with open_dict(cfg):
        cfg.mode = "eval"
    extras(cfg)
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
