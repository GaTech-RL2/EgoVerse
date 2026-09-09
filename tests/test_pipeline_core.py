import numpy as np
import pytest
import torch

from egomimic.pipeline.core import (
    Pipeline,
    Stage,
    resolve_homogeneous_scalar,
    sum_losses,
)


class _FeatureStage(Stage):
    reads = ["obs/*"]
    writes = ["feature/*"]

    def forward(self, batch):
        batch["feature/x"] = batch["obs/x"] + 1
        return batch


class _LossStage(Stage):
    train_only = True
    reads = ["feature/x"]
    writes = ["loss/value"]

    def forward(self, batch):
        batch["loss/value"] = batch["feature/x"].square().mean()
        return batch


class _ModeStage(Stage):
    reads = ["target"]
    writes = ["loss/mode"]
    reads_by_mode = {"inference": ["condition"]}
    writes_by_mode = {"inference": ["prediction"]}


def test_pipeline_registers_and_executes_stages():
    pipeline = Pipeline([_FeatureStage(), _LossStage()])
    source = torch.tensor([2.0], requires_grad=True)

    output = pipeline({"obs/x": source})
    total = sum_losses(output)
    total.backward()

    assert total.item() == 9.0
    assert source.grad.item() == 6.0


def test_pipeline_plan_resolves_wildcards_and_train_only_stages():
    feature = _FeatureStage()
    loss = _LossStage()
    pipeline = Pipeline([feature, loss])

    runnable, excluded = pipeline.plan(["obs/x"], mode="train")
    assert runnable == [feature, loss]
    assert excluded == []

    runnable, excluded = pipeline.plan(["obs/x"], mode="inference")
    assert runnable == [feature]
    assert excluded == [(loss, ["<train-only>"])]


def test_stage_mode_contract_and_explain_are_exact():
    stage = _ModeStage()
    pipeline = Pipeline([stage])

    assert stage.contract("train") == (("target",), ("loss/mode",))
    assert stage.contract("inference") == (("condition",), ("prediction",))
    assert "EXCLUDED: missing target" in pipeline.explain([], mode="train")
    assert "prediction" in pipeline.explain(["condition"], mode="inference")

    with pytest.raises(ValueError, match="train\\|inference"):
        stage.contract("validation")


def test_sum_losses_rejects_graph_without_an_objective():
    with pytest.raises(RuntimeError, match=r"no loss/\* keys"):
        sum_losses({"log/value": torch.tensor(1.0)})


@pytest.mark.parametrize("value", [1.0, torch.ones(2)])
def test_sum_losses_requires_scalar_tensors(value):
    with pytest.raises(TypeError, match="must be a scalar tensor"):
        sum_losses({"loss/value": value})


@pytest.mark.parametrize(
    "value,expected",
    [
        ("image", "image"),
        (torch.tensor(3), 3),
        (torch.tensor([3, 3]), 3),
        (np.asarray("image"), "image"),
        (np.asarray([4, 4]), 4),
        (["image", "image"], "image"),
        ((torch.tensor(5), torch.tensor(5)), 5),
    ],
)
def test_resolve_homogeneous_scalar(value, expected):
    assert resolve_homogeneous_scalar(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        [],
        torch.tensor([]),
        torch.tensor([1, 2]),
        torch.zeros(1, 1),
        np.asarray(["left", "right"]),
        np.zeros((1, 1)),
    ],
)
def test_resolve_homogeneous_scalar_rejects_ambiguous_values(value):
    with pytest.raises(ValueError):
        resolve_homogeneous_scalar(value)


def test_resolve_homogeneous_scalar_rejects_non_scalar_objects():
    with pytest.raises(TypeError, match="must be a scalar"):
        resolve_homogeneous_scalar({"selector": "image"})
