import hashlib
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import OmegaConf

import egomimic.trainHydra as train_hydra
from egomimic.utils.slurm_requeue import SaveOnlySignalCheckpoint

_OWNERSHIP_VARIABLES = ("ICE_REQUEUE_OWNER", "ICE_CHILD_REQUEUE_DISABLED")
_RESUME_VARIABLES = (
    "ICE_RESUME_CHECKPOINT",
    "ICE_RESUME_CHECKPOINT_SHA256",
    "ICE_RESUME_GLOBAL_STEP",
)


def _clear_ownership(monkeypatch):
    for name in _OWNERSHIP_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _set_runner_environment(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "runner")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "1")


def _write_checkpoint(path, *, global_step):
    torch.save({"global_step": global_step, "state_dict": {}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cfg(owner="lightning"):
    return OmegaConf.create(
        {
            "mmap_checkpoint": False,
            "runtime": {"slurm_requeue_owner": owner},
        }
    )


def test_base_config_records_lightning_as_default_requeue_owner():
    config_path = (
        Path(__file__).parents[1]
        / "egomimic"
        / "hydra_configs"
        / "train_zarr_cartesian.yaml"
    )
    cfg = OmegaConf.load(config_path)
    cfg.paths = {"output_dir": "/tmp/egomimic-test-output"}

    assert cfg.runtime.slurm_requeue_owner == "lightning"
    assert cfg.runtime.slurm_save_signal == "SIGUSR2"
    assert (
        cfg.runtime.slurm_signal_checkpoint_dir
        == "/tmp/egomimic-test-output/checkpoints"
    )


def test_skynet_default_keeps_lightning_auto_requeue(monkeypatch):
    _clear_ownership(monkeypatch)
    plugin = train_hydra._slurm_environment(_cfg())

    assert isinstance(plugin, SLURMEnvironment)
    assert plugin.auto_requeue is True
    assert plugin.requeue_signal == signal.SIGUSR1


def test_runner_owned_job_builds_one_non_requeueing_slurm_environment(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "runner")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "1")

    plugins = train_hydra._instantiate_trainer_plugins(_cfg("runner"))

    slurm_plugins = [
        plugin for plugin in plugins if isinstance(plugin, SLURMEnvironment)
    ]
    assert len(slurm_plugins) == 1
    assert slurm_plugins[0].auto_requeue is False
    assert slurm_plugins[0].requeue_signal == signal.SIGUSR1


def test_runner_owned_job_adds_one_save_only_sigusr2_callback(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "runner")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "1")
    cfg = _cfg("runner")
    cfg.runtime.slurm_save_signal = "SIGUSR2"
    cfg.runtime.slurm_signal_checkpoint_dir = str(tmp_path / "checkpoints")

    callbacks = train_hydra._instantiate_slurm_callbacks(cfg)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], SaveOnlySignalCheckpoint)
    assert callbacks[0].save_signal == signal.SIGUSR2
    assert callbacks[0].checkpoint_dir == (tmp_path / "checkpoints").resolve()


def test_child_owned_job_keeps_lightning_auto_requeue(monkeypatch):
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "child")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "0")

    plugin = train_hydra._slurm_environment(_cfg())

    assert plugin.auto_requeue is True


@pytest.mark.parametrize(
    ("configured_owner", "owner", "disabled"),
    [
        ("runner", "runner", None),
        ("runner", "runner", "0"),
        ("runner", None, None),
        ("lightning", "runner", "1"),
        ("lightning", "child", None),
        ("lightning", "child", "1"),
        ("lightning", None, "1"),
        ("unknown", "runner", "1"),
    ],
)
def test_inconsistent_requeue_ownership_fails_closed(
    monkeypatch, configured_owner, owner, disabled
):
    _clear_ownership(monkeypatch)
    if owner is not None:
        monkeypatch.setenv("ICE_REQUEUE_OWNER", owner)
    if disabled is not None:
        monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", disabled)

    with pytest.raises(RuntimeError, match="inconsistent Slurm requeue ownership"):
        train_hydra._slurm_environment(_cfg(configured_owner))


def test_runner_resume_uses_and_validates_exact_selected_checkpoint(
    monkeypatch, tmp_path
):
    _set_runner_environment(monkeypatch)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "4")
    checkpoint_path = tmp_path / "runner-selected.ckpt"
    checkpoint_sha256 = _write_checkpoint(checkpoint_path, global_step=321)
    monkeypatch.setenv("ICE_RESUME_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("ICE_RESUME_CHECKPOINT_SHA256", checkpoint_sha256)
    monkeypatch.setenv("ICE_RESUME_GLOBAL_STEP", "321")
    cfg = _cfg("runner")
    cfg.ckpt_path = str(checkpoint_path)
    trainer = SimpleNamespace(default_root_dir=str(tmp_path / "lightning-root"))

    selected = train_hydra._resolve_training_checkpoint(cfg, trainer)

    assert selected == str(checkpoint_path.resolve())
    assert cfg.ckpt_path == selected
    assert selected != str(tmp_path / "lightning-root" / "checkpoints" / "last.ckpt")


@pytest.mark.parametrize(
    ("sha_override", "step_override", "message"),
    [
        ("0" * 64, "321", "ICE_RESUME_CHECKPOINT_SHA256 mismatch"),
        (None, "320", "ICE_RESUME_GLOBAL_STEP mismatch"),
    ],
)
def test_runner_resume_rejects_checkpoint_identity_mismatch(
    monkeypatch, tmp_path, sha_override, step_override, message
):
    _set_runner_environment(monkeypatch)
    checkpoint_path = tmp_path / "runner-selected.ckpt"
    checkpoint_sha256 = _write_checkpoint(checkpoint_path, global_step=321)
    monkeypatch.setenv("ICE_RESUME_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv(
        "ICE_RESUME_CHECKPOINT_SHA256", sha_override or checkpoint_sha256
    )
    monkeypatch.setenv("ICE_RESUME_GLOBAL_STEP", step_override)

    with pytest.raises(RuntimeError, match=message):
        train_hydra._resolve_training_checkpoint(
            _cfg("runner"),
            SimpleNamespace(default_root_dir=str(tmp_path)),
        )


def test_runner_resume_requires_complete_environment(monkeypatch, tmp_path):
    _set_runner_environment(monkeypatch)
    checkpoint_path = tmp_path / "runner-selected.ckpt"
    monkeypatch.setenv("ICE_RESUME_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("ICE_RESUME_CHECKPOINT_SHA256", "")
    monkeypatch.setenv("ICE_RESUME_GLOBAL_STEP", "")

    with pytest.raises(RuntimeError, match="must be supplied together"):
        train_hydra._resolve_training_checkpoint(
            _cfg("runner"),
            SimpleNamespace(default_root_dir=str(tmp_path)),
        )


def test_runner_restart_refuses_last_checkpoint_fallback(monkeypatch, tmp_path):
    _set_runner_environment(monkeypatch)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    for name in _RESUME_VARIABLES:
        monkeypatch.setenv(name, "")

    with pytest.raises(RuntimeError, match="has no ICE_RESUME_CHECKPOINT"):
        train_hydra._resolve_training_checkpoint(
            _cfg("runner"),
            SimpleNamespace(default_root_dir=str(tmp_path)),
        )


def test_runner_clean_first_attempt_accepts_explicit_empty_selection(
    monkeypatch, tmp_path
):
    _set_runner_environment(monkeypatch)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    for name in _RESUME_VARIABLES:
        monkeypatch.setenv(name, "")
    cfg = _cfg("runner")

    selected = train_hydra._resolve_training_checkpoint(
        cfg,
        SimpleNamespace(default_root_dir=str(tmp_path)),
    )

    assert selected is None
    assert cfg.ckpt_path is None


def test_lightning_owned_restart_retains_last_checkpoint_fallback(
    monkeypatch, tmp_path
):
    _clear_ownership(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    cfg = _cfg("lightning")
    trainer = SimpleNamespace(default_root_dir=str(tmp_path / "lightning-root"))

    selected = train_hydra._resolve_training_checkpoint(cfg, trainer)

    expected = str(tmp_path / "lightning-root" / "checkpoints" / "last.ckpt")
    assert selected == expected
    assert cfg.ckpt_path == expected
