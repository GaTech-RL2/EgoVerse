"""Every shipped Hydra config must compose and fully resolve.

No data, network, GPU or cluster path is touched: this composes the config tree
the way ``trainHydra.py`` does, then forces every interpolation. It catches
absolute ``${train_datasets.x}`` references that break once a data config is
mounted under ``data.``, missing config-group files, and dead ``${paths.*}`` keys.

A second check instantiates only the ``filters`` nodes of every data config:
``DatasetFilter`` needs no DB or S3, and the zarr resolver rejects anything else.
"""

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

import egomimic
import egomimic.trainHydra  # noqa: F401  -- registers the ``eval`` and custom resolvers
from egomimic.rldb.filters import DatasetFilter

CONFIG_DIR = Path(egomimic.__file__).parent / "hydra_configs"


def _options(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


TOP_LEVEL = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
TRAIN_TOP_LEVEL = [c for c in TOP_LEVEL if c.startswith("train_")]


def _compose_and_resolve(config_name: str, overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True,
        )
    # ``${hydra:runtime.output_dir}`` is only filled in by a real Hydra run.
    OmegaConf.set_readonly(cfg.hydra, False)
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = "/nonexistent/output_dir"
    HydraConfig.instance().set_config(cfg)  # makes ``${hydra:...}`` resolvable
    # ``set_config`` freezes ``cfg.hydra``; resolve the task config on its own root.
    task_cfg = OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"])
    # Same flag ``hydra.utils.instantiate`` sets before it resolves: lets resolvers
    # such as ``${oc.select:..., []}`` yield plain Python containers.
    task_cfg._set_flag("allow_objects", True)
    OmegaConf.resolve(task_cfg)
    return task_cfg


@pytest.mark.parametrize("config_name", TOP_LEVEL)
def test_top_level_config_resolves(config_name):
    _compose_and_resolve(config_name, [])


@pytest.mark.parametrize("data", _options("data"))
@pytest.mark.parametrize("top", TRAIN_TOP_LEVEL)
def test_data_option_resolves_under_every_train_config(top, data):
    _compose_and_resolve(top, [f"data={data}"])


@pytest.mark.parametrize("model", _options("model"))
def test_model_option_resolves(model):
    _compose_and_resolve("train_zarr_cartesian", [f"model={model}"])


@pytest.mark.parametrize("evaluator", _options("evaluator"))
def test_evaluator_option_resolves(evaluator):
    _compose_and_resolve("train_zarr_cartesian", [f"evaluator={evaluator}"])


@pytest.mark.parametrize("viz", _options("evaluator/viz"))
def test_evaluator_viz_option_resolves(viz):
    _compose_and_resolve(
        "train_zarr_cartesian", [f"evaluator/viz@evaluator.viz_func={viz}"]
    )


@pytest.mark.parametrize("data", _options("data"))
def test_data_filters_instantiate_to_dataset_filter(data):
    cfg = _compose_and_resolve("train_zarr_cartesian", [f"data={data}"])
    for split in ("train_datasets", "valid_datasets"):
        for name, ds in (cfg.data.get(split) or {}).items():
            if ds is None or ds.get("filters") is None:
                continue
            target = ds.filters.get("_target_")
            assert target is not None, (
                f"{data}: {split}.{name}.filters is a plain dict; the zarr resolver "
                "only accepts a DatasetFilter (add _target_ and filter_lambdas)"
            )
            cls = hydra.utils.get_class(target)
            assert issubclass(
                cls, DatasetFilter
            ), f"{data}: {target} is not a DatasetFilter"
            if cls is DatasetFilter:  # subclasses may need network / API keys
                assert isinstance(hydra.utils.instantiate(ds.filters), DatasetFilter)
