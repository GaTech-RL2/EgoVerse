from collections import OrderedDict

import pytest
import torch

from egomimic.pipeline.stages_planar import PlanarActionMSELoss
from egomimic.pl_utils.pl_model import ModelWrapper


class _MetricPipeline:
    def __init__(self):
        self.nets = torch.nn.ModuleDict({"anchor": torch.nn.Linear(1, 1)})

    @staticmethod
    def process_batch_for_training(batch):
        return batch

    @staticmethod
    def forward_training(batch):
        return OrderedDict(
            (
                source,
                {
                    "loss/test": torch.tensor(1.0, requires_grad=True),
                    "log/MSE": values["metric"],
                    "prediction": values.get("prediction", torch.ones(2)),
                },
            )
            for source, values in batch.items()
        )

    @staticmethod
    def compute_losses(predictions, _batch):
        return {
            "loss": torch.stack(
                [result["loss/test"] for result in predictions.values()]
            ).mean()
        }

    @staticmethod
    def log_info(_info):
        return {}


def _wrapper_with_log_capture(monkeypatch):
    wrapper = ModelWrapper(pipeline=_MetricPipeline())
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **kwargs: logged.setdefault(name, (value, kwargs)),
    )
    return wrapper, logged


def test_training_logs_each_opaque_source_and_equal_source_macro(monkeypatch):
    wrapper, logged = _wrapper_with_log_capture(monkeypatch)
    batch = OrderedDict(
        (
            ("pushshapes_sim_u_socket", {"metric": torch.tensor(2.0)}),
            ("another_source", {"metric": 6.0}),
        )
    )

    wrapper.training_step(batch, batch_idx=0)

    expected = {
        "Train/MSE/pushshapes_sim_u_socket": 2.0,
        "Train/MSE/another_source": 6.0,
        "Train/MSE": 4.0,
    }
    assert expected.keys() <= logged.keys()
    for name, expected_value in expected.items():
        value, kwargs = logged[name]
        assert float(value) == pytest.approx(expected_value)
        assert kwargs == {
            "sync_dist": True,
            "on_step": False,
            "on_epoch": True,
        }


def test_training_rejects_non_scalar_log_metric_but_ignores_other_predictions(
    monkeypatch,
):
    wrapper, _logged = _wrapper_with_log_capture(monkeypatch)
    batch = {"source": {"metric": torch.tensor([1.0, 2.0])}}

    with pytest.raises(TypeError, match="must be scalar"):
        wrapper.training_step(batch, batch_idx=0)


def test_training_rejects_non_finite_log_metric(monkeypatch):
    wrapper, _logged = _wrapper_with_log_capture(monkeypatch)
    batch = {"source": {"metric": torch.tensor(float("nan"))}}

    with pytest.raises(RuntimeError, match="Non-finite pipeline metric"):
        wrapper.training_step(batch, batch_idx=0)


def test_planar_action_loss_exports_canonical_mse_metric():
    stage = PlanarActionMSELoss()
    result = stage(
        {
            "pred_action": torch.zeros(2, 3, 5),
            "target": torch.ones(2, 3, 5),
        }
    )

    assert stage.writes == ("loss/action", "log/MSE")
    assert float(result["loss/action"]) == pytest.approx(1.0)
    assert float(result["log/MSE"]) == pytest.approx(1.0)
    assert "log/action_mse" not in result
