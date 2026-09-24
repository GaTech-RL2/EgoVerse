import json

import pytest

from astra_reversal.checkpoint import inspect_checkpoint
from astra_reversal.policy_adapter import FrozenOpenPI, load_policy


def test_lerobot_horizon_is_not_the_number_of_executed_steps(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "chunk_size": 50,
                "n_action_steps": 10,
                "max_action_dim": 32,
                "output_features": {"action": {"shape": [7]}},
            }
        )
    )
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "pi05_prepare_state_tokenizer_processor_step",
                        "config": {},
                    },
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"features": {}, "norm_map": {"ACTION": "MEAN_STD"}},
                    },
                ]
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"fixture; not real weights")
    report = inspect_checkpoint(tmp_path)
    assert report["horizon"] == 50 and report["default_execute_steps"] == 10
    assert report["model_action_dim"] == 32 and report["state_in_language_tokens"]
    assert report["normalization"][0]["features"] == {}
    assert not report["openpi_loader_compatible"]
    assert report["loader"] == "lerobot"
    # Auto-routing never imports OpenPI or substitutes its model config.
    from astra_reversal.lerobot_policy import FrozenLeRobotPI05

    calls = []
    monkeypatch.setattr(
        FrozenLeRobotPI05, "load", lambda *a, **k: calls.append((a, k)) or "native"
    )
    assert (
        load_policy(str(tmp_path), config_name="irrelevant", device="cpu") == "native"
    )
    assert calls[0][0][:2] == (str(tmp_path), "cpu")
    # Explicitly selecting the OpenPI-only loader still gives a useful error.
    with pytest.raises(ValueError, match="LeRobot"):
        FrozenOpenPI.load(str(tmp_path))


def test_openpi_weights_require_normalization_assets(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fixture; not real weights")
    assert not inspect_checkpoint(tmp_path)["openpi_loader_compatible"]
    stats = tmp_path / "assets" / "libero" / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}")
    assert inspect_checkpoint(tmp_path)["openpi_loader_compatible"]


def test_lerobot_images_reach_native_resize_unchanged():
    import numpy as np

    from astra_reversal.libero_runner import policy_observation

    frame = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    observation = {
        "agentview_image": frame,
        "robot0_eye_in_hand_image": frame,
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    mapped = policy_observation(observation, image_size=None)
    np.testing.assert_array_equal(mapped["observation/image"], frame[::-1, ::-1])
    assert mapped["observation/image"].shape == (256, 256, 3)
    assert mapped["observation/state"].shape == (8,)
