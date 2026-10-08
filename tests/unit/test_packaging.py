"""A non-editable install must ship the training stack and nothing else.

Builds the wheel with the configured build backend (no pip / uv needed) and
inspects its contents: Hydra configs and robot resources must be inside,
``external/`` and ``tests/`` must not be.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def wheel_files(tmp_path_factory) -> list[str]:
    out = tmp_path_factory.mktemp("wheel")
    # setuptools reuses a stale build/lib (deleted modules would linger in the
    # wheel); build from a clean slate and leave the checkout as we found it.
    shutil.rmtree(REPO / "build", ignore_errors=True)
    code = (
        "import sys; from setuptools import build_meta; "
        f"print(build_meta.build_wheel({str(out)!r}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    shutil.rmtree(REPO / "build", ignore_errors=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    wheel = next(out.glob("*.whl"))
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def test_wheel_does_not_ship_external_or_tests(wheel_files):
    stray = sorted(
        {n.split("/")[0] for n in wheel_files}
        - {"egomimic"}
        - {
            n.split("/")[0]
            for n in wheel_files
            if n.split("/")[0].endswith(".dist-info")
        }
    )
    assert stray == [], f"top-level entries other than egomimic in the wheel: {stray}"


def test_wheel_ships_hydra_configs(wheel_files):
    packaged = {
        n[len("egomimic/hydra_configs/") :]
        for n in wheel_files
        if n.startswith("egomimic/hydra_configs/") and n.endswith(".yaml")
    }
    on_disk = {
        str(p.relative_to(REPO / "egomimic/hydra_configs"))
        for p in (REPO / "egomimic/hydra_configs").rglob("*.yaml")
    }
    assert "train_zarr_cartesian.yaml" in packaged
    assert "paths/default.yaml" in packaged
    assert (
        on_disk - packaged == set()
    ), f"hydra yamls missing from wheel: {sorted(on_disk - packaged)}"


def test_wheel_ships_robot_resources(wheel_files):
    # model_x5.xml's meshdir points at egomimic/robot/eva/stanford_repo, which
    # is not packaged, so the MJCF only loads from a checkout. Shipped anyway:
    # the URDFs beside it are self-contained and the robot code resolves the
    # XML relative to egomimic.__file__.
    assert "egomimic/resources/model_x5.xml" in wheel_files
    assert "egomimic/resources/model_arx.urdf" in wheel_files
