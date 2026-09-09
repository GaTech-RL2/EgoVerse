from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from egomimic.pipeline.core import Stage

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "ice"
    / "validate_lightning_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_lightning_checkpoint", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SOURCE = "a" * 40
CONFIG = "b" * 64
SPLIT = "c" * 64
NORMALIZATION = "d" * 64
RUN = "planar-bc-smoke"


def payload(*, scheduler: bool = True) -> dict:
    scheduler_config = (
        {"_target_": "example.scheduler", "max_steps": 100}
        if scheduler
        else None
    )
    return {
        "global_step": 7,
        "epoch": 2,
        "state_dict": {"model.weight": torch.tensor([1.0, 2.0])},
        "optimizer_states": [
            {
                "state": {0: {"momentum": torch.tensor([0.25])}},
                "param_groups": [{"params": [0], "lr": 1.0e-4}],
            }
        ],
        "lr_schedulers": (
            [{"last_epoch": 7, "_last_lr": [1.0e-4]}] if scheduler else []
        ),
        "loops": {"fit_loop": {"completed": 7}},
        "hyper_parameters": {
            "config_tree": {
                "model": {
                    "_target_": "example.ModelWrapper",
                    "scheduler": scheduler_config,
                }
            }
        },
        "run_provenance": {
            "source_commit": SOURCE,
            "config_sha256": CONFIG,
            "run_id": RUN,
            "split_manifest_sha256": SPLIT,
            "normalization_sha256": NORMALIZATION,
        },
    }


def add_unranked_model_checkpoint_state(
    candidate: dict,
    *,
    include_config: bool = False,
    save_top_k: int = -1,
) -> tuple[str, dict]:
    state_key = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', "
        "'every_n_train_steps': 2, 'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )
    state = {
        "monitor": None,
        "best_model_score": None,
        "best_model_path": "/output/checkpoints/epoch-0-step-2.ckpt",
        "current_score": None,
        "dirpath": "/output/checkpoints",
        "best_k_models": {},
        "kth_best_model_path": "",
        "kth_value": torch.tensor(float("inf")),
        "last_model_path": "/output/checkpoints/last.ckpt",
    }
    candidate["callbacks"] = {state_key: state}
    if include_config:
        candidate["hyper_parameters"]["callbacks"] = {
            "model_checkpoint": {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "monitor": None,
                "mode": "min",
                "save_top_k": save_top_k,
                "every_n_train_steps": 2,
                "every_n_epochs": None,
                "train_time_interval": None,
            }
        }
    return state_key, state


def save(path: Path, value: object) -> Path:
    torch.save(value, path)
    return path


def invoke(
    checkpoint: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_environment = os.environ.copy()
    if environment:
        merged_environment.update(environment)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(checkpoint), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=merged_environment,
        check=False,
    )


def success_json(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines
    return json.loads(lines[-1])


def test_structural_cli_emits_runner_compatible_json(tmp_path: Path):
    checkpoint = save(tmp_path / "model.ckpt", payload())
    result = invoke(
        checkpoint,
        "--structural-only",
        environment={
            "ICE_EXPECTED_SOURCE_COMMIT": SOURCE,
            "ICE_EXPECTED_CONFIG_SHA256": CONFIG,
            "ICE_EXPECTED_RUN_ID": RUN,
            "ICE_EXPECTED_SPLIT_SHA256": SPLIT,
            "ICE_EXPECTED_NORMALIZATION_SHA256": NORMALIZATION,
            "ICE_REQUIRE_LR_SCHEDULERS": "1",
        },
    )

    metadata = success_json(result)

    assert metadata["global_step"] == 7
    assert type(metadata["global_step"]) is int
    assert metadata["valid"] is True
    assert metadata["strict_model_reload"] is False
    assert metadata["optimizer_state_count"] == 1
    assert metadata["lr_scheduler_state_count"] == 1
    assert metadata["source_commit"] == SOURCE
    assert metadata["config_sha256"] == CONFIG
    assert metadata["run_id"] == RUN
    assert metadata["split_sha256"] == SPLIT
    assert metadata["normalization_sha256"] == NORMALIZATION


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("state_dict", {}, "state_dict"),
        ("optimizer_states", [], "optimizer_states"),
        ("loops", {}, "loops"),
        ("lr_schedulers", [], "missing lr_schedulers"),
    ],
)
def test_required_full_state_is_enforced(
    tmp_path: Path, key: str, value: object, message: str
):
    candidate = payload()
    candidate[key] = value
    result = invoke(save(tmp_path / f"{key}.ckpt", candidate), "--structural-only")

    assert result.returncode == 1
    assert message in result.stderr
    assert result.stdout == ""


def test_checkpoint_without_configured_scheduler_may_omit_scheduler_state(
    tmp_path: Path,
):
    checkpoint = save(tmp_path / "no-scheduler.ckpt", payload(scheduler=False))

    metadata = success_json(invoke(checkpoint, "--structural-only"))

    assert metadata["scheduler_configured"] is False
    required = invoke(
        checkpoint,
        "--structural-only",
        "--require-lr-schedulers",
    )
    assert required.returncode == 1
    assert "missing lr_schedulers" in required.stderr


@pytest.mark.parametrize("step", [-1, 1.5, True])
def test_global_step_must_be_an_exact_nonnegative_integer(
    tmp_path: Path, step: object
):
    candidate = payload()
    candidate["global_step"] = step

    result = invoke(save(tmp_path / "bad-step.ckpt", candidate), "--structural-only")

    assert result.returncode == 1
    assert "global_step must be a non-negative integer" in result.stderr


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["state_dict"].__setitem__(
                "bad", torch.tensor(float("nan"))
            ),
            "non-finite tensor",
        ),
        (
            lambda value: value["optimizer_states"][0]["param_groups"][0].__setitem__(
                "lr", float("inf")
            ),
            "non-finite scalar",
        ),
        (
            lambda value: value["lr_schedulers"][0].__setitem__(
                "last_epoch", float("inf")
            ),
            "non-finite scalar",
        ),
        (
            lambda value: value["loops"]["fit_loop"].__setitem__(
                "completed", float("inf")
            ),
            "non-finite scalar",
        ),
        (
            lambda value: value.__setitem__("numeric_content", float("inf")),
            "non-finite scalar",
        ),
    ],
)
def test_nonfinite_checkpoint_content_is_rejected(
    tmp_path: Path, mutate, message: str
):
    candidate = payload()
    mutate(candidate)

    result = invoke(save(tmp_path / "nonfinite.ckpt", candidate), "--structural-only")

    assert result.returncode == 1
    assert message in result.stderr


@pytest.mark.parametrize("include_config", [False, True])
def test_unranked_model_checkpoint_kth_value_sentinel_is_allowed(
    tmp_path: Path, include_config: bool
):
    candidate = payload()
    add_unranked_model_checkpoint_state(candidate, include_config=include_config)

    metadata = success_json(
        invoke(save(tmp_path / "model-checkpoint.ckpt", candidate), "--structural-only")
    )

    assert metadata["allowed_callback_sentinel_count"] == 1


def test_model_checkpoint_sentinel_requires_save_all_config_when_available(
    tmp_path: Path,
):
    candidate = payload()
    add_unranked_model_checkpoint_state(
        candidate,
        include_config=True,
        save_top_k=1,
    )

    result = invoke(
        save(tmp_path / "top-one-checkpoint.ckpt", candidate),
        "--structural-only",
    )

    assert result.returncode == 1
    assert "non-finite tensor" in result.stderr
    assert ".kth_value" in result.stderr


def test_model_checkpoint_exception_does_not_cover_other_callback_state(
    tmp_path: Path,
):
    candidate = payload()
    _, callback_state = add_unranked_model_checkpoint_state(candidate)
    callback_state["current_score"] = torch.tensor(float("inf"))

    result = invoke(
        save(tmp_path / "bad-callback-state.ckpt", candidate),
        "--structural-only",
    )

    assert result.returncode == 1
    assert "non-finite tensor" in result.stderr
    assert ".current_score" in result.stderr


def test_model_checkpoint_exception_requires_unmonitored_callback_identity(
    tmp_path: Path,
):
    candidate = payload()
    state_key, callback_state = add_unranked_model_checkpoint_state(candidate)
    monitored_key = state_key.replace("'monitor': None", "'monitor': 'Valid/MSE'")
    callback_state["monitor"] = "Valid/MSE"
    candidate["callbacks"] = {monitored_key: callback_state}

    result = invoke(
        save(tmp_path / "monitored-checkpoint.ckpt", candidate),
        "--structural-only",
    )

    assert result.returncode == 1
    assert "non-finite tensor" in result.stderr
    assert ".kth_value" in result.stderr


def test_model_checkpoint_exception_requires_tensor_ranking_sentinel(
    tmp_path: Path,
):
    candidate = payload()
    _, callback_state = add_unranked_model_checkpoint_state(candidate)
    callback_state["kth_value"] = float("inf")

    result = invoke(
        save(tmp_path / "scalar-sentinel.ckpt", candidate),
        "--structural-only",
    )

    assert result.returncode == 1
    assert "non-finite scalar" in result.stderr
    assert ".kth_value" in result.stderr


def test_expected_identity_is_fail_closed_on_missing_or_mismatch(tmp_path: Path):
    missing = payload()
    del missing["run_provenance"]["split_manifest_sha256"]
    missing_result = invoke(
        save(tmp_path / "missing.ckpt", missing),
        "--structural-only",
        "--expected-split-sha256",
        SPLIT,
    )
    mismatch_result = invoke(
        save(tmp_path / "mismatch.ckpt", payload()),
        "--structural-only",
        "--expected-run-id",
        "another-run",
    )

    assert missing_result.returncode == 1
    assert "not present" in missing_result.stderr
    assert mismatch_result.returncode == 1
    assert "run checkpoint identity mismatch" in mismatch_result.stderr


def test_cli_identity_overrides_environment(tmp_path: Path):
    checkpoint = save(tmp_path / "override.ckpt", payload())

    metadata = success_json(
        invoke(
            checkpoint,
            "--structural-only",
            "--expected-run-id",
            RUN,
            environment={"ICE_EXPECTED_RUN_ID": "wrong-from-environment"},
        )
    )

    assert metadata["run_id"] == RUN


def test_conflicting_embedded_identity_aliases_are_rejected(tmp_path: Path):
    candidate = payload()
    candidate["metadata"] = {"wandb_run_id": "different-run"}

    result = invoke(save(tmp_path / "conflict.ckpt", candidate), "--structural-only")

    assert result.returncode == 1
    assert "conflicting run checkpoint identities" in result.stderr


def test_default_requires_strict_model_reload(tmp_path: Path):
    result = invoke(save(tmp_path / "synthetic.ckpt", payload()))

    assert result.returncode == 1
    assert "ModelWrapper" in result.stderr


def test_invalid_scheduler_environment_is_rejected(tmp_path: Path):
    result = invoke(
        save(tmp_path / "environment.ckpt", payload()),
        "--structural-only",
        environment={"ICE_REQUIRE_LR_SCHEDULERS": "yes"},
    )

    assert result.returncode == 1
    assert "must be 0 or 1" in result.stderr


def test_strict_reload_reconstructs_configured_pipeline_wrapper(tmp_path: Path):
    from egomimic.pl_utils.pl_model import ModelWrapper

    class_path = f"{__name__}.TinyParameterizedStage"
    config_tree = {
        "model": {
            "_target_": "egomimic.pl_utils.pl_model.ModelWrapper",
            "pipeline": {
                "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                "stages": [{"_target_": class_path}],
            },
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "_partial_": True,
                "lr": 1.0e-4,
            },
            "scheduler": None,
        }
    }
    wrapper = ModelWrapper(config_tree=config_tree)
    candidate = {
        "global_step": 0,
        "epoch": 0,
        "state_dict": wrapper.state_dict(),
        "optimizer_states": [{"state": {}, "param_groups": [{"params": [0]}]}],
        "lr_schedulers": [],
        "loops": {"fit_loop": {"completed": 0}},
        "hyper_parameters": {
            "config_tree": config_tree,
            "scheduler_interval": "step",
            "scheduler_frequency": 1,
            "enable_grad_norm": True,
        },
    }
    checkpoint = save(tmp_path / "strict.ckpt", candidate)

    metadata = MODULE.validate_checkpoint(checkpoint)

    assert metadata["strict_model_reload"] is True
    assert metadata["strict_model_parameter_count"] == 1
    assert metadata["strict_model_tensor_count"] >= 1

    candidate["state_dict"]["unexpected.weight"] = torch.tensor(0.0)
    broken = save(tmp_path / "strict-broken.ckpt", candidate)
    with pytest.raises(RuntimeError, match="strict reload failed"):
        MODULE.validate_checkpoint(broken)


class TinyParameterizedStage(Stage):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict) -> dict:
        return batch
