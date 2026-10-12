"""The robot rollout rebuilds a checkpoint's Eva input pipeline from the
transform mode trainHydra records in the model config tree, and refuses to
guess for an unrecorded non-legacy action width."""

from types import SimpleNamespace

import pytest

import egomimic.trainHydra as train_hydra
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.robot.rollout import EMBODIMENT_MAP, PolicyRollout


@pytest.fixture(autouse=True)
def no_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)


def test_config_tree_records_the_eva_transform_mode(compose_resolve, tmp_path):
    cfg = compose_resolve(
        "train_zarr_cartesian", ["data=eva", f"paths.output_dir={tmp_path}"]
    )
    tree = train_hydra._build_model_config_tree(cfg)
    assert tree.data_transform_modes.eva_bimanual == (
        cfg.data.train_datasets.eva_bimanual.resolver.transform_list.mode
    )


def _rollout(config_tree=None, action_width=None):
    emb = EMBODIMENT_MAP["both"]
    shapes = (
        {}
        if action_width is None
        else {emb: {"actions_cartesian": (100, action_width)}}
    )
    policy = SimpleNamespace(
        hparams=SimpleNamespace(
            config_tree=config_tree, norm_stats_state={"shapes": shapes}
        ),
        _as_config=ModelWrapper._as_config,
    )
    r = PolicyRollout.__new__(PolicyRollout)
    r.arm, r.embodiment_id, r.policy = "both", emb, policy
    return r


def test_recorded_mode_is_used():
    tree = {
        "model": {},
        "data_transform_modes": {"eva_bimanual": "cartesian_wristframe_6d"},
    }
    assert _rollout(tree, 20)._checkpoint_transform_mode() == "cartesian_wristframe_6d"


def test_override_wins_over_the_checkpoint():
    tree = {"model": {}, "data_transform_modes": {"eva_bimanual": "cartesian_6d"}}
    r = _rollout(tree, 20)
    assert r._checkpoint_transform_mode("cartesian_wristframe_6d") == (
        "cartesian_wristframe_6d"
    )


def test_unrecorded_legacy_width_uses_the_legacy_pipeline():
    assert _rollout({"model": {}}, 14)._checkpoint_transform_mode() is None


@pytest.mark.parametrize("width", [20, None])
def test_unrecorded_other_width_refuses_to_guess(width):
    with pytest.raises(ValueError, match="--transform-mode"):
        _rollout({"model": {}}, width)._checkpoint_transform_mode()
