"""A non-editable install must ship the training stack and nothing else.

Builds the wheel with the configured build backend (no pip / uv needed) and
inspects its contents: Hydra configs and robot resources must be inside,
``external/`` and ``tests/`` must not be. The SharpaDex kinematic models must
ship and load from the unpacked wheel.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
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
    return next(out.glob("*.whl"))


@pytest.fixture(scope="module")
def wheel_files(wheel) -> list[str]:
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


def test_wheel_ships_mcap_proto(wheel_files):
    # The .proto handed to vendors; egomimic.rldb.mcap.schemas renders the same text.
    assert "egomimic/rldb/mcap/proto/egoverse.proto" in wheel_files


SHARPA_KINEMATICS = "egomimic/scripts/sharpa_process/kinematics/"


def test_wheel_ships_sharpa_kinematics(wheel_files):
    # The SharpaDex converter's models: package data keyed on the kinematics
    # package, so both sharpa_process/ and kinematics/ need an __init__.py.
    shipped = {
        n[len(SHARPA_KINEMATICS) :]
        for n in wheel_files
        if n.startswith(SHARPA_KINEMATICS)
    }
    assert {
        "__init__.py",
        "urdf.py",
        "sharpa_dexmate.yaml",
        "NTH_POC1_0_URDF_250905v4.urdf",
        "left_sharpa_wave_with_flange.urdf",
        "right_sharpa_wave_with_flange.urdf",
        "LICENSE.txt",
        "NOTICE.txt",
    } <= shipped, sorted(shipped)
    assert "egomimic/scripts/sharpa_process/sharpa_to_zarr.py" in wheel_files


def test_sharpa_kinematics_load_from_the_installed_wheel(wheel, tmp_path):
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(site)
    code = f"""
from pathlib import Path
from egomimic.scripts.sharpa_process import kinematics
from egomimic.scripts.sharpa_process.sharpa_tracks import SharpaKinematics
here = Path(kinematics.__file__).resolve()
assert here.is_relative_to(Path({str(site)!r}).resolve()), here
spec = kinematics.load_spec()
for entry in (spec["platform"], *spec["hands"].values()):
    kinematics.verified_urdf(entry["urdf"], entry["urdf_sha256"])
kin = SharpaKinematics()
assert len(kin.column_names) == 65 and "head_base_link" in kin.links
for name in ("LICENSE.txt", "NOTICE.txt"):
    assert kinematics.asset_path(name).read_text().strip()
print("ok")
"""
    # cwd is the extracted wheel, never the checkout: ``python -c`` puts the
    # cwd first on sys.path, and PYTHONPATH comes before the editable install.
    env = {**os.environ, "PYTHONPATH": str(site), "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=site,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().endswith("ok")
