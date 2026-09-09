import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "ice" / "ice_gpu_probe.py"
SPEC = importlib.util.spec_from_file_location("ice_gpu_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_ecc_health_accepts_zero_counters_and_clear_flags():
    report = """
        Volatile Uncorrectable ECC : 0
        Aggregate Uncorrectable ECC : 0
        Pending Page Blacklist : No
        Reset Required : No
        Remapping Failure Occurred : No
    """

    health = MODULE.parse_ecc_health(report)

    assert len(health["uncorrectable_ecc_counters"]) == 2
    assert health["unhealthy_flags"] == []


@pytest.mark.parametrize(
    "report",
    (
        "Volatile Uncorrectable ECC : 1\nReset Required : No\n",
        "Volatile Uncorrectable ECC : 0\nReset Required : Yes\n",
        "Reset Required : No\n",
        "Volatile Uncorrectable ECC : unknown\nReset Required : No\n",
    ),
)
def test_parse_ecc_health_fails_closed(report):
    with pytest.raises(RuntimeError):
        MODULE.parse_ecc_health(report)


def test_probe_exercises_backward_and_one_rank_nccl_lifecycle():
    text = MODULE_PATH.read_text()

    assert "loss.backward()" in text
    assert "dist.init_process_group(" in text
    assert 'backend="nccl"' in text
    assert "dist.all_reduce(collective)" in text
    assert "dist.destroy_process_group()" in text
    assert '"world_size": world_size' in text
