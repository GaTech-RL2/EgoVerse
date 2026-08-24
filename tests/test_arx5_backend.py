from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import egomimic.robot.backends.arx5 as arx_backend

ROOT = Path(__file__).parents[1]


class _Controller:
    pass


class _JointState:
    pass


def _binding():
    return SimpleNamespace(
        Arx5JointController=_Controller,
        JointState=_JointState,
    )


def test_prefers_upstream_python311_package(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        if name == "arx5_interface":
            return _binding()
        raise AssertionError(name)

    monkeypatch.setattr(arx_backend, "import_module", fake_import)
    api = arx_backend.load_arx5_api()
    assert api.import_name == "arx5_interface"
    assert api.joint_controller is _Controller
    assert calls == ["arx5_interface"]


def test_falls_back_to_legacy_package(monkeypatch):
    def fake_import(name):
        if name == "arx5_interface":
            raise ImportError("not installed")
        return _binding()

    monkeypatch.setattr(arx_backend, "import_module", fake_import)
    assert arx_backend.load_arx5_api().import_name == "arx5.arx5_interface"


def test_missing_binding_is_optional_until_live_control(monkeypatch):
    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(arx_backend, "import_module", unavailable)
    assert arx_backend.optional_arx5_api() is None
    with pytest.raises(arx_backend.Arx5Unavailable, match="Offline rollout"):
        arx_backend.load_arx5_api()


def test_python311_wheel_applies_bounded_x5_gripper_patch():
    patch_name = "0002-allow-bounded-x5-negative-gripper-close.patch"
    build_script = (ROOT / "scripts/build_arx5_py311_wheel.sh").read_text()
    patch = (ROOT / "egomimic/robot/eva/arx5_patches" / patch_name).read_text()

    assert patch_name in build_script
    assert "kX5GripperPositionMinM = -0.012" in patch
    assert "kGripperPositionToleranceM = 0.005" in patch
    assert 'robot_config.robot_model == "X5"' in patch
    assert "? kX5GripperPositionMinM : 0.0" in patch
    assert (
        "+    if (joint_state_.gripper_pos < gripper_position_min - "
        "kGripperPositionToleranceM ||" in patch
    )
    assert "+    if (output_joint_cmd_.gripper_pos < gripper_position_min)" in patch
    assert "+        output_joint_cmd_.gripper_pos = gripper_position_min;" in patch
    assert "gripper_torque_max" not in patch


def test_x5a_gripper_close_endpoints_match_native_floor():
    config_path = ROOT / "egomimic/robot/eva/eva_ws/src/config/configs.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["gripper"]["left"]["close"] == pytest.approx(-0.012)
    assert config["gripper"]["right"]["close"] == pytest.approx(-0.012)
