import inspect
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_io import ActionTargetBuilder
from egomimic.pipeline.stages_sampler import FusedObsEncoder
from egomimic.pl_utils.pl_model import ModelWrapper


class _PackedState(torch.nn.Module):
    def forward_packed(self, *, obs_packed, **_context):
        return obs_packed["state"]


class _TinyHead(Stage):
    reads = ["condition", "target", "selector"]
    writes = ["loss/fit", "log/fit"]
    reads_by_mode = {"inference": ["condition", "selector"]}
    writes_by_mode = {"inference": ["prediction"]}

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.seen_keys = None

    def execute(self, batch, *, mode):
        self.seen_keys = set(batch)
        if mode == "train":
            loss = (batch["target"] * self.scale).square().mean()
            batch["loss/fit"] = loss
            batch["log/fit"] = loss.detach()
        elif mode == "inference":
            batch_size = batch["condition"].shape[0]
            batch["prediction"] = self.scale.expand(batch_size, 3, 2)
        else:  # pragma: no cover - Pipeline validates modes first
            raise ValueError(mode)
        return batch

    def forward(self, batch):
        return self.execute(batch, mode="train")


class _ImageHead(Stage):
    reads = ["pixels", "reference"]
    writes = ["loss/reconstruction"]
    reads_by_mode = {"inference": ["pixels"]}
    writes_by_mode = {"inference": ["reconstruction"]}

    def execute(self, batch, *, mode):
        if mode == "train":
            batch["loss/reconstruction"] = (
                (batch["pixels"] - batch["reference"]).square().mean()
            )
        elif mode == "inference":
            batch["reconstruction"] = batch["pixels"]
        else:  # pragma: no cover - Pipeline validates modes first
            raise ValueError(mode)
        return batch

    def forward(self, batch):
        return self.execute(batch, mode="train")


def _algo():
    return PipelineAlgo(
        stages=[
            FusedObsEncoder(
                encoder=_PackedState(),
                inputs={"state": "state"},
                n_obs_steps=1,
            ),
            ActionTargetBuilder(),
            _TinyHead(),
        ],
        device="cpu",
    )


def _raw_batch(scale=1.0):
    return OrderedDict(
        source_a={
            "state": torch.randn(2, 4),
            "actions": torch.full((2, 3, 2), float(scale)),
            "selector": "arm_a",
            "ignored": torch.ones(2),
        }
    )


def test_pipeline_algo_preserves_opaque_sources_and_reduces_losses():
    algo = _algo()
    processed = algo.process_batch_for_training(_raw_batch())

    assert list(processed) == ["source_a"]
    assert set(processed["source_a"]) == {
        "state",
        "actions",
        "selector",
        "ignored",
    }
    predictions = algo.forward_training(processed)
    losses = algo.compute_losses(predictions, processed)

    assert isinstance(predictions, OrderedDict)
    assert list(predictions) == ["source_a"]
    assert losses["loss"].ndim == 0
    assert torch.equal(losses["loss"], losses["source_0_loss"])
    losses["loss"].backward()
    assert algo.pipeline.stages[2].scale.grad is not None
    logged = algo.log_info({"losses": losses})
    assert logged["Loss"] == pytest.approx(losses["loss"].item())


def test_pipeline_algo_equal_weights_multiple_sources():
    algo = _algo()
    raw = _raw_batch(scale=1.0)
    raw["another_source"] = _raw_batch(scale=3.0)["source_a"]
    processed = algo.process_batch_for_training(raw)

    predictions = algo.forward_training(processed)
    losses = algo.compute_losses(predictions, processed)

    expected = torch.stack(
        [predictions[source]["loss/fit"] for source in predictions]
    ).mean()
    torch.testing.assert_close(losses["loss"], expected)


def test_pipeline_algo_inference_excludes_training_nodes_without_filtering_metadata():
    algo = _algo()
    processed = algo.process_batch_for_training(_raw_batch())
    algo.pipeline.train()

    results = algo.forward_eval(processed)

    expected = torch.full((2, 3, 2), 0.5)
    assert torch.equal(results["source_a"]["prediction"], expected)
    assert algo.pipeline.stages[2].seen_keys == {
        "state",
        "actions",
        "selector",
        "ignored",
        "condition",
    }
    assert results["source_a"]["actions"] is processed["source_a"]["actions"]
    assert "target" not in results["source_a"]
    assert "target" not in processed["source_a"]


def test_pipeline_algo_is_not_specific_to_control_batches():
    algo = PipelineAlgo(stages=[_ImageHead()], device="cpu")
    pixels = torch.randn(2, 3, 8, 8)
    batch = {"images": {"pixels": pixels, "reference": pixels + 1}}
    processed = algo.process_batch_for_training(batch)

    train_results = algo.forward_training(processed)
    losses = algo.compute_losses(train_results, processed)
    inference_results = algo.forward_eval(processed)

    assert losses["loss"].item() == pytest.approx(1.0)
    assert torch.equal(inference_results["images"]["reconstruction"], pixels)
    assert inference_results["images"]["reference"] is processed["images"]["reference"]


def test_pipeline_algo_constructor_and_source_are_route_agnostic():
    parameters = inspect.signature(PipelineAlgo).parameters
    assert tuple(parameters) == ("stages", "device")
    with pytest.raises(TypeError):
        PipelineAlgo(stages=[], domains=["anything"])

    root = Path(__file__).parents[1]
    source = "\n".join(
        (root / relative).read_text().lower()
        for relative in ("egomimic/pipeline/core.py", "egomimic/pipeline/algo.py")
    )
    for forbidden in (
        "domain",
        "embodiment",
        "ac_key",
        "action_key",
        "norm_stats",
        "rollout_adapter",
        "action_horizon",
        "rollout",
    ):
        assert forbidden not in source


def test_pipeline_algo_rejects_malformed_grouped_input():
    algo = PipelineAlgo(stages=[], device="cpu")
    with pytest.raises(TypeError, match="mapping of flat batches"):
        algo.process_batch_for_training([{"pixels": torch.zeros(1)}])
    with pytest.raises(TypeError, match="source values"):
        algo.process_batch_for_training({"source": torch.zeros(1)})


def test_pipeline_algo_moves_nested_tensors_without_changing_dtype():
    algo = PipelineAlgo(stages=[], device="cpu")
    processed = algo.process_batch_for_training(
        {
            "source": {
                "value": torch.ones(2, dtype=torch.float64),
                "metadata": {"index": torch.ones(2, dtype=torch.int16)},
            }
        }
    )

    assert processed["source"]["value"].dtype == torch.float64
    assert processed["source"]["metadata"]["index"].dtype == torch.int16


def test_model_wrapper_keeps_data_provenance_out_of_pipeline():
    config_tree = {
        "model": {
            "pipeline": {
                "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                "stages": [],
                "device": "cpu",
            }
        }
    }
    wrapper = ModelWrapper(config_tree=config_tree)

    assert isinstance(wrapper.model, PipelineAlgo)
    with pytest.raises(TypeError, match="norm_stats_state"):
        ModelWrapper(config_tree=config_tree, norm_stats_state={"legacy": True})


def test_model_wrapper_accepts_generic_pipeline_loss(monkeypatch):
    wrapper = ModelWrapper(pipeline=_algo())
    monkeypatch.setattr(wrapper, "log", lambda *args, **kwargs: None)

    loss = wrapper.training_step(_raw_batch(), batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_wrapper_logs_stable_diffusion_mse_aliases(monkeypatch):
    wrapper = ModelWrapper(pipeline=_algo())
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **_kwargs: logged.__setitem__(name, value),
    )
    original_compute_losses = wrapper.model.compute_losses

    def compute_losses(predictions, batch):
        losses = original_compute_losses(predictions, batch)
        losses["source_0_log_diffusion_noise"] = losses["source_0_loss"]
        return losses

    monkeypatch.setattr(wrapper.model, "compute_losses", compute_losses)
    loss = wrapper.training_step(_raw_batch(), batch_idx=0)

    assert torch.equal(logged["Train/MSE/source_a"], loss)
    assert torch.equal(logged["Train/MSE"], loss)
