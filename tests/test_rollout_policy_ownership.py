import ast
from pathlib import Path

import pytest
import torch

from egomimic.algo.algo import Algo
from egomimic.rollout.policy import (
    RolloutPolicyConfig,
    _checkpoint_algo_class,
    load_rollout_policy,
)

ROOT = Path(__file__).parents[1]


def _classes(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text())
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


@pytest.mark.parametrize(
    "module_path",
    [
        "egomimic/algo/pi.py",
        "egomimic/algo/hpt.py",
        "egomimic/algo/act.py",
        "egomimic/pipeline/algo.py",
    ],
)
def test_each_algorithm_module_owns_a_policy(module_path):
    assert "Policy" in _classes(module_path)


def test_robot_entrypoint_has_no_concrete_algorithm_dependency():
    source = (ROOT / "egomimic/robot/rollout.py").read_text()
    assert "class PolicyRollout" not in source
    assert "DenoisingPolicy" not in source
    assert "PipelineAlgo" not in source
    assert "egomimic.algo." not in source
    assert "load_rollout_policy" in source


def test_robot_launcher_uses_the_validated_python311_image():
    source = (ROOT / "run_eva_docker.sh").read_text()
    assert "egomimic-eva:py311" in source
    assert "robot-env:latest" not in source


def test_checkpoint_target_dispatches_to_algorithm_class():
    checkpoint = {
        "hyper_parameters": {
            "config_tree": {
                "model": {"robomimic_model": {"_target_": "egomimic.algo.algo.Algo"}}
            }
        }
    }
    assert _checkpoint_algo_class(checkpoint) is Algo


def test_rollout_config_validates_the_hardware_contract():
    config = RolloutPolicyConfig(
        arm="both", query_frequency=30, cartesian=True, action_frame="base"
    )
    assert config.embodiment_name == "eva_bimanual"
    assert config.embodiment_id == 6
    with pytest.raises(ValueError, match="query_frequency"):
        RolloutPolicyConfig(arm="both", query_frequency=0, cartesian=True)


def test_live_policy_fails_closed_without_cuda(tmp_path, monkeypatch):
    checkpoint = {
        "hyper_parameters": {
            "config_tree": {
                "model": {"robomimic_model": {"_target_": "egomimic.algo.algo.Algo"}}
            }
        }
    }
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = RolloutPolicyConfig(
        arm="both", query_frequency=1, cartesian=True, require_cuda=True
    )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        load_rollout_policy(str(checkpoint_path), config)
