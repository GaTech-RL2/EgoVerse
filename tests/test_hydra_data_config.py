from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egomimic.utils.hydra_utils import (
    instantiate_dataset_splits_from_path,
    load_data_config_from_path,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _config_tree(tmp_path: Path) -> Path:
    config_dir = tmp_path / "hydra_configs"
    _write(
        config_dir / "root.yaml",
        """\
defaults:
  - paths: default
  - data: null
  - _self_
""",
    )
    _write(
        config_dir / "paths" / "default.yaml",
        "dataset_dir: /composed/datasets\n",
    )
    _write(
        config_dir / "data" / "base.yaml",
        """\
train_datasets:
  active_train:
    _target_: builtins.dict
    location: ${paths.dataset_dir}/train
  null_train: null
valid_datasets:
  active_valid:
    _target_: builtins.dict
    location: ${paths.dataset_dir}/valid
  null_valid: null
""",
    )
    config_path = config_dir / "data" / "example.yaml"
    _write(
        config_path,
        """\
defaults:
  - base
  - _self_
""",
    )
    return config_path


def test_data_config_loader_resolves_defaults_and_root_path_context(tmp_path):
    config_path = _config_tree(tmp_path)

    data = load_data_config_from_path(config_path, root_config_name="root")

    assert data.train_datasets.active_train.location == "/composed/datasets/train"
    assert data.valid_datasets.active_valid.location == "/composed/datasets/valid"


def test_dataset_split_instantiation_skips_null_entries_and_accepts_overrides(
    tmp_path,
):
    config_path = _config_tree(tmp_path)

    train, valid = instantiate_dataset_splits_from_path(
        config_path,
        overrides=["paths.dataset_dir=/override"],
        root_config_name="root",
    )

    assert train == {"active_train": {"location": "/override/train"}}
    assert valid == {"active_valid": {"location": "/override/valid"}}


def test_data_config_loader_rejects_non_data_group_path(tmp_path):
    config_dir = tmp_path / "hydra_configs"
    config_path = config_dir / "model" / "example.yaml"
    _write(config_path, "value: 1\n")

    with pytest.raises(ValueError, match="Expected a config below"):
        load_data_config_from_path(config_path, root_config_name="root")


def test_repository_data_configs_resolve_in_root_context():
    config_dir = Path(__file__).parents[1] / "egomimic" / "hydra_configs" / "data"

    for config_path in sorted(config_dir.glob("*.yaml")):
        data = load_data_config_from_path(config_path)
        OmegaConf.to_container(data, resolve=True)
