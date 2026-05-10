from pathlib import Path
from types import SimpleNamespace

import lightning
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


class DummyAlgo:
    def __init__(self, norm_stats, value, echo_value=None):
        self.norm_stats = norm_stats
        self.value = value
        self.echo_value = echo_value
        self.nets = nn.ModuleDict({"policy": nn.Linear(1, 1)})


def _build_norm_stats_state():
    """
    Build a synthetic stats-only MultiDataset by direct field assignment.
    Bypasses populate_from_datasets() since we don't construct a real dataset
    graph in unit tests; this mirrors the post-populate state.
    """
    stats = MultiDataset(state={}, norm_mode="quantile")
    emb_id = 8  # eva_bimanual
    stats.embodiments.add(emb_id)
    stats.key_types[emb_id] = {
        "observations.state.ee_pose": "proprio_keys",
        "actions_cartesian": "action_keys",
    }
    stats.zarr_keys[emb_id] = {
        "observations.state.ee_pose": "observations.state.ee_pose",
        "actions_cartesian": "actions_cartesian",
    }
    stats.infer_shapes_from_batch(
        {
            "observations.state.ee_pose": torch.zeros(2, 14),
            "actions_cartesian": torch.zeros(2, 100, 14),
        }
    )
    stats.norm_stats[emb_id] = {
        "observations.state.ee_pose": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
        },
        "actions_cartesian": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
        },
    }
    return stats.to_state()


def _build_config_tree():
    return OmegaConf.create(
        {
            "model": {
                "robomimic_model": {
                    "_target_": "egomimic.pl_utils.test_model_wrapper.DummyAlgo",
                    "norm_stats": None,
                    "value": 7,
                    "echo_value": "${model.robomimic_model.value}",
                },
                "optimizer": {
                    "_target_": "torch.optim.SGD",
                    "_partial_": True,
                    "lr": 0.1,
                },
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "_partial_": True,
                    "step_size": 1,
                    "gamma": 0.5,
                },
            }
        }
    )


def test_model_wrapper_reconstructs_model_from_config_tree():
    wrapper = ModelWrapper(
        config_tree=_build_config_tree(),
        norm_stats_state=_build_norm_stats_state(),
    )

    assert wrapper.model.__class__.__name__ == "DummyAlgo"
    assert wrapper.model.__class__.__module__ == "egomimic.pl_utils.test_model_wrapper"
    assert wrapper.model.echo_value == 7
    assert wrapper.model.norm_stats.key_shape("actions_cartesian", 8) == (
        2,
        100,
        14,
    )


def test_model_wrapper_config_tree_builds_optimizer_and_scheduler():
    wrapper = ModelWrapper(
        config_tree=_build_config_tree(),
        norm_stats_state=_build_norm_stats_state(),
    )
    wrapper._trainer = SimpleNamespace(model=wrapper)

    optimizers = wrapper.configure_optimizers()

    assert isinstance(optimizers["optimizer"], torch.optim.SGD)
    assert isinstance(optimizers["lr_scheduler"], torch.optim.lr_scheduler.StepLR)


def test_model_wrapper_load_from_checkpoint_reconstructs_from_hparams(tmp_path: Path):
    wrapper = ModelWrapper(
        config_tree=_build_config_tree(),
        norm_stats_state=_build_norm_stats_state(),
    )
    ckpt_path = tmp_path / "dummy_wrapper.ckpt"
    torch.save(
        {
            "state_dict": wrapper.state_dict(),
            "hyper_parameters": dict(wrapper.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        ckpt_path,
    )

    loaded = ModelWrapper.load_from_checkpoint(str(ckpt_path), weights_only=False)

    assert loaded.model.__class__.__name__ == "DummyAlgo"
    assert loaded.model.__class__.__module__ == "egomimic.pl_utils.test_model_wrapper"
    assert loaded.model.echo_value == 7
    torch.testing.assert_close(
        loaded.model.norm_stats.norm_stats[8]["actions_cartesian"]["mean"],
        torch.zeros(14),
    )
