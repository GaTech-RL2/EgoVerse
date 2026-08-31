"""Compose the FULL trainHydra config exactly as the launcher invokes it.

Every other preflight in this repo loads a single YAML by path and instantiates
it. That is strictly weaker than what the run does, and it misses the entire
class of "hydra cannot even find/compose this":

  MissingConfigException: In 'train_zarr_cartesian.yaml':
      Could not find 'data/control_modes_gripper_arc_D10_M16_append_r0'

`data/pusht` and `model/bf` are config GROUPS, so the overrides need the group
prefix (`data=pusht/<name>`, `model=bf/<name>`). Loading
`hydra_configs/data/pusht/<name>.yaml` by path succeeds regardless, which is
exactly why the path-based tests passed while the run could not start.

This composes with the real config_name and the real override list, so a
missing group prefix, an unknown key, or a struct-mode violation fails here in
seconds instead of on a node after the image pull, uv sync, R2 pull and
staging.
"""

from __future__ import annotations

import pathlib

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "egomimic/hydra_configs"
CONFIG_NAME = "train_zarr_cartesian.yaml"

DATA_CFG = "pusht/control_modes_gripper_arc_D10_M16_append_r0"
EVAL_CFG = "eval_sim_control_modes"
MODEL_CFGS = [
    f"bf/bf_ctrlmode_{arm}_{cap}"
    for arm in ("arm1_dp_flow", "arm2_causal_bidir", "arm3_state_action_ar",
                "arm4_state_idm")
    for cap in ("small", "large")
]


def _compose(overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


@pytest.mark.parametrize("model_cfg", MODEL_CFGS)
def test_phase1_norm_stats_composition(model_cfg):
    """Phase 1: 1 GPU, no evaluator, one batch. Exactly the launcher's call."""
    cfg = _compose([
        f"data={DATA_CFG}",
        f"model={model_cfg}",
        "~evaluator",
        "paths.dataset_dir=/workspace/pushshapes",
        "launch_params.nodes=1",
        "launch_params.gpus_per_node=1",
        "norm_stats.save_cache_dir=/workspace/norm_stats/x",
        "trainer.max_epochs=1",
        "trainer.min_epochs=1",
        "trainer.limit_train_batches=1",
        "trainer.limit_val_batches=0",
        "~logger",
        "~callbacks",
        "name=ctrlmode_x",
        "description=ARM x | small capacity | train a+b | holdout c+d_normstats",
    ])
    assert cfg.data is not None and cfg.model is not None
    assert "evaluator" not in cfg or cfg.evaluator is None


@pytest.mark.parametrize("model_cfg", MODEL_CFGS)
def test_phase2_training_composition(model_cfg):
    """Phase 2: N GPUs WITH the evaluator.

    The evaluator is only composed here, which is why an evaluator mistake
    survives all of phase 1 on a real node.
    """
    cfg = _compose([
        f"data={DATA_CFG}",
        f"model={model_cfg}",
        f"evaluator={EVAL_CFG}",
        "trainer.strategy=ddp_find_unused_parameters_true",
        "norm_stats.precomputed_norm_path=/workspace/norm_stats/x/norm_stats.json",
        "paths.dataset_dir=/workspace/pushshapes",
        "launch_params.nodes=1",
        "launch_params.gpus_per_node=8",
        "data.train_dataloader_params.pushshapes_sim_gripper.batch_size=16",
        "data.train_dataloader_params.pushshapes_sim_gripper.num_workers=4",
        "data.valid_dataloader_params.pushshapes_sim_gripper.batch_size=16",
        "data.valid_dataloader_params.pushshapes_sim_gripper.num_workers=4",
        "logger.wandb.entity=rl2-group",
        "trainer.limit_train_batches=2500",
        "trainer.check_val_every_n_epoch=1",
        "trainer.limit_val_batches=1",
        "callbacks.model_checkpoint.every_n_epochs=1",
        "trainer.max_epochs=100",
        "trainer.min_epochs=100",
        "name=ctrlmode_x",
        "description=ARM x | small capacity | train a+b | holdout c+d",
    ])
    assert cfg.evaluator is not None
    gaps = [e.control_gap for e in cfg.evaluator.evals]
    assert gaps == ["ideal", "tight", "laggy", "loose", "sticky", "jittery"]

    # The evaluator must actually be REACHED. trainer/ddp.yaml ships
    # check_val_every_n_epoch=200 with num_sanity_val_steps=0, so at
    # max_epochs=100 the first validation is scheduled for an epoch that never
    # arrives: the run trains fully, exits COMPLETED, and logs no success rate.
    assert int(cfg.trainer.check_val_every_n_epoch) <= int(cfg.trainer.max_epochs)
    # on_validation_step runs the WHOLE evaluator per batch, so >1 batch
    # multiplies every rollout (the shipped default of 80 means 4,800).
    assert int(cfg.trainer.limit_val_batches) == 1
    # Otherwise a checkpoint only lands at the very end (default every_n_epochs=100).
    assert int(cfg.callbacks.model_checkpoint.every_n_epochs) <= 1


def test_group_prefix_is_actually_required():
    """Guards the gate: without `pusht/` the composition must still fail.

    If this ever passes, the group layout changed and the launcher's override
    form should be revisited rather than trusted.
    """
    from hydra.errors import MissingConfigException

    with pytest.raises(MissingConfigException):
        _compose([
            "data=control_modes_gripper_arc_D10_M16_append_r0",
            "model=bf/bf_ctrlmode_arm2_causal_bidir_small",
            "~evaluator", "~logger", "~callbacks",
        ])
