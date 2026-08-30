import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.scripts import verify_training_smoke as verifier


def _wandb_record(**items):
    record = verifier.wandb_internal_pb2.Record()
    for key, value in items.items():
        item = record.history.item.add()
        item.key = key
        item.value_json = json.dumps(value)
    return record.SerializeToString()


def _wandb_exit(exit_code: int = 0):
    record = verifier.wandb_internal_pb2.Record()
    record.exit.exit_code = exit_code
    return record.SerializeToString()


def test_read_wandb_history_collects_all_supported_metric_namespaces(
    monkeypatch,
) -> None:
    payloads = [
        _wandb_record(
            **{
                "trainer/global_step": 1,
                "Train/Loss": 1.0,
                "Train/MSE/pushshapes_sim_u_socket": 0.9,
                "Timing/Custom_Sec": 0.2,
                "Optimizer/custom_lr": 3.0e-6,
                "Valid/MSE": 0.8,
                "Valid/Native_MSE/pushshapes_sim_u_socket": 0.7,
                "Ignored/Metric": 123.0,
            }
        ),
        _wandb_exit(),
        None,
    ]

    class FakeDataStore:
        def __init__(self) -> None:
            self.closed = False

        def open_for_scan(self, path: str) -> None:
            assert path.endswith("run.wandb")

        def scan_data(self):
            return payloads.pop(0)

        def close(self) -> None:
            self.closed = True

    store = FakeDataStore()
    monkeypatch.setattr(verifier, "DataStore", lambda: store)
    training, validation, exit_code = verifier.read_wandb_history(
        Path("/tmp/run.wandb")
    )

    assert exit_code == 0
    assert store.closed is True
    assert training == [
        {
            "trainer_global_step": 1,
            "train_metrics": {
                "Train/Loss": 1.0,
                "Train/MSE/pushshapes_sim_u_socket": 0.9,
            },
            "timing_metrics": {"Timing/Custom_Sec": 0.2},
            "optimizer_metrics": {"Optimizer/custom_lr": 3.0e-6},
        }
    ]
    assert validation == [
        {
            "trainer_global_step": 1,
            "validation_metrics": {
                "Valid/MSE": 0.8,
                "Valid/Native_MSE/pushshapes_sim_u_socket": 0.7,
            },
        }
    ]


def test_configured_metrics_are_required_on_finite_appropriate_rows() -> None:
    configured_metrics = [
        "Train/MSE",
        "Train/MSE/pushshapes_sim_u_socket",
        "Timing/Custom_Sec",
        "Optimizer/custom_lr",
        "Valid/MSE",
        "Valid/Native_MSE/pushshapes_sim_u_socket",
    ]
    cfg = OmegaConf.create(
        {
            "run_provenance": {
                "required_wandb_metrics": configured_metrics,
            }
        }
    )
    configured, required_step, required_validation = verifier._required_metric_contract(
        cfg
    )

    assert configured == configured_metrics
    assert required_step == {
        "train_metrics": {
            "Train/Loss",
            "Train/MSE",
            "Train/MSE/pushshapes_sim_u_socket",
        },
        "timing_metrics": {
            "Timing/Process_Batch_Sec",
            "Timing/Forward_Pass_Sec",
            "Timing/Compute_Losses_Sec",
            "Timing/Custom_Sec",
        },
        "optimizer_metrics": {
            "Optimizer/param_group_0_lr",
            "Optimizer/custom_lr",
        },
    }
    assert required_validation == {
        "Valid/MSE",
        "Valid/Native_MSE/pushshapes_sim_u_socket",
    }

    row = {
        "trainer_global_step": 0,
        "train_metrics": {metric: 1.0 for metric in required_step["train_metrics"]},
        "timing_metrics": {metric: 1.0 for metric in required_step["timing_metrics"]},
        "optimizer_metrics": {
            metric: 3.0e-6 for metric in required_step["optimizer_metrics"]
        },
    }
    history = [row, {**row, "trainer_global_step": 1}]
    _, steps = verifier._select_dense_training_history(history, required_step)
    assert steps == [0, 1]

    missing_train_metric = [
        row,
        {
            **history[1],
            "train_metrics": {
                "Train/Loss": 1.0,
                "Train/MSE": 1.0,
            },
        },
    ]
    with pytest.raises(AssertionError):
        verifier._select_dense_training_history(missing_train_metric, required_step)

    nonfinite_training = [
        row,
        {
            **history[1],
            "train_metrics": {
                **history[1]["train_metrics"],
                "Train/MSE": float("nan"),
            },
        },
    ]
    with pytest.raises(AssertionError):
        verifier._select_dense_training_history(nonfinite_training, required_step)

    validation_history = [
        {
            "trainer_global_step": 1,
            "validation_metrics": {metric: 0.5 for metric in required_validation},
        }
    ]
    selected = verifier._select_scheduled_validation(
        validation_history,
        [19, 20],
        required_validation,
    )
    assert selected == validation_history[0]

    with pytest.raises(AssertionError):
        verifier._select_scheduled_validation(
            [
                {
                    "trainer_global_step": 1,
                    "validation_metrics": {"Valid/MSE": 0.5},
                }
            ],
            [19, 20],
            required_validation,
        )
    with pytest.raises(AssertionError):
        verifier._select_scheduled_validation(
            [
                {
                    "trainer_global_step": 1,
                    "validation_metrics": {
                        "Valid/MSE": 0.5,
                        "Valid/Native_MSE/pushshapes_sim_u_socket": float("inf"),
                    },
                }
            ],
            [19, 20],
            required_validation,
        )


def test_legacy_validation_contract_still_requires_finite_action_mse() -> None:
    history = [
        {
            "trainer_global_step": 1,
            "validation_metrics": {
                "Valid/emb19_actions_action_mse": 0.1,
                "Valid/emb20_actions_action_mse": 0.2,
            },
        }
    ]
    assert verifier._select_scheduled_validation(history, [19, 20], set()) == history[0]

    history[0]["validation_metrics"]["Valid/emb20_actions_action_mse"] = float("nan")
    with pytest.raises(AssertionError):
        verifier._select_scheduled_validation(history, [19, 20], set())


def test_checkpoint_model_wrapper_is_reconstructed_strictly_on_cpu(
    monkeypatch,
) -> None:
    captured = {}

    class FakeModelWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 3)
            self.register_buffer("scale", torch.ones(1))

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, **kwargs):
            captured["checkpoint_path"] = checkpoint_path
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(verifier, "ModelWrapper", FakeModelWrapper)
    result = verifier._strict_load_model_wrapper(Path("/tmp/smoke.ckpt"))

    assert captured == {
        "checkpoint_path": "/tmp/smoke.ckpt",
        "map_location": "cpu",
        "weights_only": False,
        "strict": True,
    }
    assert result == {
        "status": "passed",
        "model_class": (
            f"{FakeModelWrapper.__module__}.{FakeModelWrapper.__qualname__}"
        ),
        "map_location": "cpu",
        "strict": True,
        "state_dict_key_count": 3,
        "parameter_count": 9,
        "buffer_count": 1,
    }
