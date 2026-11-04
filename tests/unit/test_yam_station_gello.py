"""The committed GELLO profile through both CLIs' --check-config; no device opens."""

from pathlib import Path

import pytest

import egomimic
from egomimic.robot.yam_station import calibrate_gello, collect_gello

PROFILE = (
    Path(egomimic.__file__).parent / "hydra_configs/robot/yam_rl2_gello_collect.yaml"
)


def _forbid(*args, **kwargs):
    raise AssertionError("--check-config touched a device")


@pytest.mark.parametrize(
    "module, hardware",
    [
        (
            collect_gello,
            ("validate_serial_ports", "BimanualGelloReader", "create_robot"),
        ),
        (calibrate_gello, ("validate_serial_ports", "DynamixelPositionBus")),
    ],
)
def test_check_config_validates_profile_without_hardware(
    monkeypatch, capsys, module, hardware
):
    for name in hardware:
        monkeypatch.setattr(module, name, _forbid)
    assert module.main(["--config", str(PROFILE), "--check-config"]) == 0
    assert "no devices were opened" in capsys.readouterr().out
