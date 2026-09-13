"""A resumed or evaluated PI run does not need the base weights on disk: when
cfg.ckpt_path is an existing checkpoint file, the model's config tree gets
pytorch_weight_path=null (the checkpoint's state_dict holds every weight) while
cfg keeps the real path. Slurm requeues resolve their ckpt_path before that
decision, and only when last.ckpt exists."""

import inspect
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import egomimic.trainHydra as train_hydra
from egomimic.utils.checkpoint_utils import load_checkpoint_weights


@pytest.fixture(autouse=True)
def no_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)


def _requeue(monkeypatch, restart_count="1"):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", restart_count)


def _pi_cfg(compose_resolve, run_dir: Path, *overrides):
    # paths.output_dir feeds both trainer.default_root_dir and the checkpoint
    # callback's dirpath, as in a real run.
    return compose_resolve(
        "train_zarr_cartesian",
        ["model=pi0.5_base", f"paths.output_dir={run_dir}", *overrides],
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _model_weight_path(cfg):
    tree = train_hydra._build_model_config_tree(cfg)
    return tree.model.robomimic_model.config.pytorch_weight_path


def test_fresh_run_keeps_base_weights(compose_resolve, tmp_path):
    cfg = _pi_cfg(compose_resolve, tmp_path)
    train_hydra._prepare_checkpoint_resume(cfg)
    assert cfg.ckpt_path is None
    assert _model_weight_path(cfg) is not None


def test_explicit_ckpt_path_nulls_base_weights_for_model_only(
    compose_resolve, tmp_path
):
    ckpt = _touch(tmp_path / "runA" / "last.ckpt")
    cfg = _pi_cfg(compose_resolve, tmp_path, f"ckpt_path={ckpt}")
    before = OmegaConf.to_container(cfg)
    train_hydra._prepare_checkpoint_resume(cfg)
    assert _model_weight_path(cfg) is None
    # cfg (logged to W&B, used for everything but the model) is untouched.
    assert OmegaConf.to_container(cfg) == before
    assert cfg.model.robomimic_model.config.pytorch_weight_path is not None


@pytest.mark.parametrize("ckpt_path", ["last", "best", "hpc", "missing.ckpt"])
def test_ckpt_path_that_is_not_a_file_keeps_base_weights(
    compose_resolve, tmp_path, ckpt_path
):
    """Lightning's ``last`` with no last checkpoint yet loads nothing, so the
    base weights are the only weights the run gets."""
    cfg = _pi_cfg(compose_resolve, tmp_path, f"ckpt_path={ckpt_path}")
    train_hydra._prepare_checkpoint_resume(cfg)
    assert cfg.ckpt_path == ckpt_path
    assert _model_weight_path(cfg) is not None


def test_requeue_resumes_from_last_ckpt(monkeypatch, compose_resolve, tmp_path):
    _requeue(monkeypatch)
    last = _touch(tmp_path / "checkpoints" / "last.ckpt")
    cfg = _pi_cfg(compose_resolve, tmp_path)
    train_hydra._prepare_checkpoint_resume(cfg)
    assert cfg.ckpt_path == str(last)
    assert _model_weight_path(cfg) is None


def test_requeue_reads_checkpoint_callback_dirpath(
    monkeypatch, compose_resolve, tmp_path
):
    _requeue(monkeypatch)
    _touch(tmp_path / "checkpoints" / "last.ckpt")  # the default dir: not used
    last = _touch(tmp_path / "elsewhere" / "last.ckpt")
    cfg = _pi_cfg(
        compose_resolve,
        tmp_path,
        f"callbacks.model_checkpoint.dirpath={last.parent}",
    )
    train_hydra._prepare_checkpoint_resume(cfg)
    assert cfg.ckpt_path == str(last)


def test_requeue_without_checkpoint_dirpath_uses_run_dir(
    monkeypatch, compose_resolve, tmp_path
):
    _requeue(monkeypatch)
    cfg = _pi_cfg(compose_resolve, tmp_path, "callbacks.model_checkpoint.dirpath=null")
    assert train_hydra._requeue_resume_path(cfg) == str(
        tmp_path / "checkpoints" / "last.ckpt"
    )


def test_requeue_before_first_checkpoint_starts_over(
    monkeypatch, compose_resolve, tmp_path, caplog
):
    _requeue(monkeypatch, "2")
    cfg = _pi_cfg(compose_resolve, tmp_path)
    with caplog.at_level("WARNING"):
        train_hydra._prepare_checkpoint_resume(cfg)
    assert cfg.ckpt_path is None
    assert _model_weight_path(cfg) is not None
    assert "last.ckpt does not exist" in caplog.text
    assert "ckpt_path=None (training starts over)" in caplog.text


def test_requeue_before_first_checkpoint_keeps_launch_ckpt_path(
    monkeypatch, compose_resolve, tmp_path, caplog
):
    _requeue(monkeypatch)
    ckpt = _touch(tmp_path / "runA" / "last.ckpt")
    cfg = _pi_cfg(compose_resolve, tmp_path / "runB", f"ckpt_path={ckpt}")
    with caplog.at_level("WARNING"):
        train_hydra._prepare_checkpoint_resume(cfg)
    assert cfg.ckpt_path == str(ckpt)
    assert f"falling back to ckpt_path={ckpt}" in caplog.text
    assert "starts over" not in caplog.text


def test_first_slurm_attempt_is_not_a_requeue(monkeypatch, compose_resolve, tmp_path):
    _requeue(monkeypatch, "0")
    cfg = _pi_cfg(compose_resolve, tmp_path)
    assert train_hydra._requeue_resume_path(cfg) is None


def test_hpt_config_without_weight_key_is_left_alone(compose_resolve, tmp_path):
    ckpt = _touch(tmp_path / "last.ckpt")
    cfg = compose_resolve(
        "train_zarr_cartesian",
        [
            "model=hpt_bc_flow_eva",
            f"paths.output_dir={tmp_path}",
            f"ckpt_path={ckpt}",
        ],
    )
    before = OmegaConf.to_container(cfg)
    train_hydra._prepare_checkpoint_resume(cfg)
    tree = train_hydra._build_model_config_tree(cfg)  # no KeyError
    assert OmegaConf.to_container(cfg) == before
    assert "config" not in tree.model.robomimic_model


def test_pretrained_eval_keeps_base_weights(compose_resolve, tmp_path):
    """eval_latent's pretrained=true evaluates the base weights and uses
    ckpt_path only to route its output; the weights must not be nulled."""
    ckpt = _touch(tmp_path / "last.ckpt")
    cfg = _pi_cfg(compose_resolve, tmp_path, f"ckpt_path={ckpt}", "+pretrained=true")
    train_hydra._prepare_checkpoint_resume(cfg)
    assert _model_weight_path(cfg) is not None


def test_ckpt_path_settled_before_model_config_tree():
    """A requeue's ckpt_path must be known when the model's config tree is
    built, or the resumed PI run reads its base weights again."""
    module_src = inspect.getsource(train_hydra)  # train() is wrapped, no __wrapped__
    src = module_src[module_src.index("\ndef train(") :]
    assert src.index("_prepare_checkpoint_resume(cfg)") < src.index(
        "_build_model_config_tree(cfg)"
    )


def test_unresolved_struct_cfg_like_a_real_run(monkeypatch, compose_resolve, tmp_path):
    """train() gets an unresolved, struct-flagged cfg where the PI key is still
    the ${oc.env:...} interpolation; select must resolve it and update must be
    allowed on the model tree."""
    _requeue(monkeypatch)
    last = _touch(tmp_path / "checkpoints" / "last.ckpt")
    cfg = compose_resolve(
        "train_zarr_cartesian",
        ["model=pi0.5_base", f"paths.output_dir={tmp_path}"],
        keep_hydra=True,
    )
    assert OmegaConf.is_struct(cfg)
    raw = OmegaConf.to_container(cfg.model.robomimic_model.config, resolve=False)
    assert str(raw["pytorch_weight_path"]).startswith("${oc.env:")
    train_hydra._prepare_checkpoint_resume(cfg)
    assert cfg.ckpt_path == str(last)
    assert _model_weight_path(cfg) is None
    assert cfg.model.robomimic_model.config.pytorch_weight_path is not None


def _save_ckpt(path: Path, state_dict) -> Path:
    torch.save({"state_dict": state_dict, "optimizer_states": []}, path)
    return path


def test_load_checkpoint_weights_fails_on_missing_keys(tmp_path):
    model = torch.nn.Linear(2, 2)
    ckpt = _save_ckpt(tmp_path / "eval.ckpt", {"weight": torch.zeros(2, 2)})
    with pytest.raises(RuntimeError, match=r"eval.ckpt.*missing 1 key.*bias"):
        load_checkpoint_weights(model, ckpt)


def test_load_checkpoint_weights_warns_on_unexpected_keys(tmp_path, caplog):
    model = torch.nn.Linear(2, 2)
    sd = dict(model.state_dict())
    sd["weight"] = torch.ones(2, 2)
    sd["extra.buffer"] = torch.zeros(1)
    ckpt = _save_ckpt(tmp_path / "x.ckpt", sd)
    with caplog.at_level("WARNING"):
        load_checkpoint_weights(model, ckpt)
    assert "extra.buffer" in caplog.text
    assert torch.equal(model.weight, torch.ones(2, 2))
