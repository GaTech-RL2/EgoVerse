"""Shared Hydra compose helper for tests/unit."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

import egomimic
import egomimic.trainHydra  # noqa: F401  -- registers the ``eval`` and custom resolvers

CONFIG_DIR = Path(egomimic.__file__).parent / "hydra_configs"


def compose_and_resolve(
    config_name: str, overrides: list[str], keep_hydra: bool = False
) -> DictConfig:
    """Compose like ``trainHydra.py`` does, then force every interpolation.

    Returns the task config (without the ``hydra`` node) fully resolved, or with
    ``keep_hydra`` the unresolved full tree (``OmegaConf.select`` resolves on read).
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=config_name, overrides=overrides, return_hydra_config=True
        )
    # ``${hydra:runtime.output_dir}`` is only filled in by a real Hydra run.
    OmegaConf.set_readonly(cfg.hydra, False)
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = "/nonexistent/output_dir"
    HydraConfig.instance().set_config(cfg)  # makes ``${hydra:...}`` resolvable
    if keep_hydra:
        return cfg
    # ``set_config`` freezes ``cfg.hydra``; resolve the task config on its own root.
    task_cfg = OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"])
    # Same flag ``hydra.utils.instantiate`` sets before it resolves: lets resolvers
    # such as ``${oc.select:..., []}`` yield plain Python containers.
    task_cfg._set_flag("allow_objects", True)
    OmegaConf.resolve(task_cfg)
    return task_cfg


@pytest.fixture
def compose_resolve():
    yield compose_and_resolve
    HydraConfig.instance().cfg = None  # do not leak the singleton between tests
