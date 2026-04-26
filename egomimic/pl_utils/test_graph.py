import pytest
import torch
import torch.nn as nn

from egomimic.pl_utils.graph import Graph
from egomimic.pl_utils.nodes import EvalNode, FreezeWrapper, Node


class AddOne(nn.Module):
    def forward(self, x):
        return x + 1


class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat([a, b], dim=-1)


def test_default_dispatch_calls_module():
    n = Node(AddOne(), inputs=["x"], outputs=["y"])
    out = Graph([n])({"x": torch.zeros(2)}, global_mode="train")
    assert torch.equal(out["y"], torch.ones(2))


def test_kwarg_routing_by_input_name():
    n = Node(Concat(), inputs=["a", "b"], outputs=["c"])
    out = Graph([n])(
        {"a": torch.zeros(1, 3), "b": torch.ones(1, 2)}, global_mode="train"
    )
    assert out["c"].shape == (1, 5)


def test_topo_order_chains_nodes():
    a = Node(AddOne(), inputs=["x"], outputs=["y"], name="a")
    b = Node(AddOne(), inputs=["y"], outputs=["z"], name="b")
    # Pass in reverse declaration order to confirm sorting works.
    out = Graph([b, a])({"x": torch.zeros(1)}, global_mode="train")
    assert torch.equal(out["z"], torch.full((1,), 2.0))


def test_duplicate_output_names_raise():
    a = Node(AddOne(), inputs=["x"], outputs=["y"], name="a")
    b = Node(AddOne(), inputs=["x"], outputs=["y"], name="b")
    with pytest.raises(ValueError, match="produced by both"):
        Graph([a, b])


def test_cycle_detection():
    a = Node(AddOne(), inputs=["b_out"], outputs=["a_out"], name="a")
    b = Node(AddOne(), inputs=["a_out"], outputs=["b_out"], name="b")
    with pytest.raises(ValueError, match="cycle"):
        Graph([a, b])


def test_missing_input_raises_at_runtime():
    n = Node(AddOne(), inputs=["x"], outputs=["y"])
    with pytest.raises(KeyError):
        Graph([n])({}, global_mode="train")


def test_mode_fn_overrides_default():
    def double(node, inputs):
        return {"y": inputs["x"] * 2}

    n = Node(AddOne(), inputs=["x"], outputs=["y"], mode_fns={"train": double})
    out = Graph([n])({"x": torch.full((1,), 3.0)}, global_mode="train")
    assert torch.equal(out["y"], torch.full((1,), 6.0))


def test_empty_mode_fn_skips_node():
    consumer_called = []

    def record(node, inputs):
        consumer_called.append(True)
        return {"z": inputs["y"]}

    a = Node(AddOne(), inputs=["x"], outputs=["y"], mode_fns={"train": None}, name="a")
    b = Node(
        AddOne(), inputs=["y"], outputs=["z"], mode_fns={"train": record}, name="b"
    )
    out = Graph([a, b])({"x": torch.zeros(1)}, global_mode="train")
    assert "y" not in out
    assert "z" not in out
    assert consumer_called == []


def test_eval_node_only_runs_in_eval_mode():
    ran = []

    def record(node, inputs):
        ran.append(True)
        return {"y": inputs["x"]}

    n = EvalNode(AddOne(), inputs=["x"], outputs=["y"], mode_fns={"eval": record})
    g = Graph([n])
    g({"x": torch.zeros(1)}, global_mode="train")
    assert ran == []
    g({"x": torch.zeros(1)}, global_mode="eval")
    assert ran == [True]


def test_freeze_wrapper_disables_grad_and_holds_eval():
    inner = nn.Linear(3, 3)
    w = FreezeWrapper(inner, freeze=True)
    assert all(not p.requires_grad for p in w.parameters())
    assert not inner.training
    w.train()
    assert not inner.training


def test_tuple_output_packs_in_order():
    class TwoOut(nn.Module):
        def forward(self, x):
            return x, x * 2

    n = Node(TwoOut(), inputs=["x"], outputs=["a", "b"])
    out = Graph([n])({"x": torch.full((1,), 3.0)}, global_mode="train")
    assert torch.equal(out["a"], torch.full((1,), 3.0))
    assert torch.equal(out["b"], torch.full((1,), 6.0))


def test_dict_output_passes_through():
    def fn(node, inputs):
        return {"a": inputs["x"], "b": inputs["x"] + 1}

    n = Node(AddOne(), inputs=["x"], outputs=["a", "b"], mode_fns={"train": fn})
    out = Graph([n])({"x": torch.zeros(1)}, global_mode="train")
    assert torch.equal(out["a"], torch.zeros(1))
    assert torch.equal(out["b"], torch.ones(1))
