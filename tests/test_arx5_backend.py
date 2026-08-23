from types import SimpleNamespace

import pytest

import egomimic.robot.backends.arx5 as arx_backend


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
