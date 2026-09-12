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
from omegaconf import OmegaConf

import egomimic
from egomimic.rldb.filters import DatasetFilter

# Needed at import time for parametrize; not imported from conftest because
# that only resolves under pytest's default (prepend) import mode.
CONFIG_DIR = Path(egomimic.__file__).parent / "hydra_configs"


def _options(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


TOP_LEVEL = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
TRAIN_TOP_LEVEL = [c for c in TOP_LEVEL if c.startswith("train_")]


@pytest.mark.parametrize("config_name", TOP_LEVEL)
def test_top_level_config_resolves(config_name, compose_resolve):
    compose_resolve(config_name, [])


@pytest.mark.parametrize("data", _options("data"))
@pytest.mark.parametrize("top", TRAIN_TOP_LEVEL)
def test_data_option_resolves_under_every_train_config(top, data, compose_resolve):
    compose_resolve(top, [f"data={data}"])


@pytest.mark.parametrize("model", _options("model"))
def test_model_option_resolves(model, compose_resolve):
    compose_resolve("train_zarr_cartesian", [f"model={model}"])


@pytest.mark.parametrize("evaluator", _options("evaluator"))
def test_evaluator_option_resolves(evaluator, compose_resolve):
    compose_resolve("train_zarr_cartesian", [f"evaluator={evaluator}"])


@pytest.mark.parametrize("viz", _options("evaluator/viz"))
def test_evaluator_viz_option_resolves(viz, compose_resolve):
    compose_resolve("train_zarr_cartesian", [f"evaluator/viz@evaluator.viz_func={viz}"])


@pytest.mark.parametrize("data", _options("data"))
def test_data_filters_instantiate_to_dataset_filter(data, compose_resolve):
    cfg = compose_resolve("train_zarr_cartesian", [f"data={data}"])
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


@pytest.mark.parametrize("launcher", _options("hydra/launcher"))
def test_submitit_launcher_config_resolves(launcher, compose_resolve):
    """The launcher node is resolved by hydra -m before the task runs; it holds
    the ${eval:} gres strings and signal_delay_s."""
    cfg = compose_resolve(
        "train_zarr_cartesian", [f"hydra/launcher={launcher}"], keep_hydra=True
    )
    launcher_cfg = OmegaConf.to_container(cfg.hydra.launcher, resolve=True)
    if launcher.startswith("submitit"):
        assert launcher_cfg["signal_delay_s"] >= 300, launcher_cfg
        assert launcher_cfg["timeout_min"] > 0
