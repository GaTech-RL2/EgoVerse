"""Cluster identity in configs is overridable by environment variables.

Lab defaults stay in the yamls so existing launches are unchanged; an outsider
points the same configs elsewhere without editing them.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import egomimic

CONFIG_DIR = Path(egomimic.__file__).parent / "hydra_configs"

ENV_VARS = [
    "EGOVERSE_DATASET_DIR",
    "EGOVERSE_LOG_DIR",
    "EGOVERSE_PI05_WEIGHTS",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "EGOVERSE_ROOT",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_lab_defaults_when_env_unset(compose_resolve):
    cfg = compose_resolve("train_zarr_cartesian", ["model=pi0.5_base"])
    assert cfg.paths.dataset_dir == "/coc/flash7/scratch/egoverseS3ZarrDataset"
    assert cfg.paths.log_dir.endswith("/logs")
    assert cfg.logger.wandb.entity == "rl2-group"
    assert cfg.logger.wandb.project == "zarr_test"
    assert cfg.model.robomimic_model.config.pytorch_weight_path.endswith(
        "pi05_base_pytorch"
    )


def test_env_vars_override_cluster_identity(monkeypatch, compose_resolve):
    monkeypatch.setenv("EGOVERSE_DATASET_DIR", "/data/zarr")
    monkeypatch.setenv("EGOVERSE_LOG_DIR", "/scratch/runs")
    monkeypatch.setenv("EGOVERSE_PI05_WEIGHTS", "/weights/pi05")
    monkeypatch.setenv("WANDB_ENTITY", "other-team")
    monkeypatch.setenv("WANDB_PROJECT", "other-project")

    cfg = compose_resolve("train_zarr_cartesian", ["model=pi0.5_base"])

    assert cfg.paths.dataset_dir == "/data/zarr"
    assert cfg.paths.log_dir == "/scratch/runs"
    assert cfg.model.robomimic_model.config.pytorch_weight_path == "/weights/pi05"
    assert cfg.logger.wandb.entity == "other-team"
    assert cfg.logger.wandb.project == "other-project"


def test_hydra_run_dir_follows_log_dir(monkeypatch, compose_resolve):
    monkeypatch.setenv("EGOVERSE_LOG_DIR", "/scratch/runs")
    cfg = compose_resolve(
        "train_zarr_cartesian", ["name=exp", "description=d"], keep_hydra=True
    )
    run_dir = OmegaConf.select(cfg, "hydra.run.dir")
    sweep_dir = OmegaConf.select(cfg, "hydra.sweep.dir")
    assert run_dir.startswith("/scratch/runs/exp/d_"), run_dir
    assert sweep_dir.startswith("/scratch/runs/exp/d_"), sweep_dir


def test_multirun_launch_resolves_sweep_dir(tmp_path, monkeypatch):
    """`hydra -m` (the submitit launch path) reads hydra.sweep.dir before any
    HydraConfig exists, so paths.* must not depend on ``${hydra:...}``."""
    app = tmp_path / "app.py"
    app.write_text(
        "import hydra\n"
        "from hydra.core.hydra_config import HydraConfig\n"
        "import egomimic.utils.hydra_resolvers  # noqa: F401\n"
        f"@hydra.main(version_base=None, config_path={str(CONFIG_DIR)!r}, config_name='train_zarr_cartesian')\n"
        "def app(cfg):\n"
        "    print('OUTPUT_DIR=' + HydraConfig.get().runtime.output_dir)\n"
        "app()\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(app),
            "-m",
            "hydra/launcher=basic",
            "name=exp",
            "description=d",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = [line for line in proc.stdout.splitlines() if line.startswith("OUTPUT_DIR=")]
    assert out, proc.stdout[-1000:]
    output_dir = Path(out[0].removeprefix("OUTPUT_DIR=")).resolve()
    assert output_dir.is_relative_to(tmp_path.resolve()), output_dir
    assert output_dir.parent.parent.name == "exp" and output_dir.name == "0", output_dir
    assert output_dir.parent.parent.parent.name == "logs", output_dir
