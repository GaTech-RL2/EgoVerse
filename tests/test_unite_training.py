import copy
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.pl_utils.pl_model_unite_released import ReleasedUniteModelWrapper
from egomimic.trainHydra import _instantiate_model_wrapper, _resolve_model_wrapper_class
from egomimic.utils.unite_optim import (
    ReleasedUniteCompositeOptimizer,
    VendoredTorchMuon,
    partition_released_unite_parameters,
    released_unite_two_stage_scheduler,
)


class _PartitionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        visual = nn.Module()
        visual.encoder = nn.Module()
        visual.encoder.img_encoders = nn.ModuleDict(
            {"front_img_1": nn.Sequential(nn.Conv2d(3, 2, 1), nn.Flatten())}
        )
        generative = nn.Module()
        generative.blocks = nn.ModuleList([nn.Linear(2, 2)])
        generative.content_projection = nn.Linear(2, 2)
        generative.condition_projection = nn.Linear(2, 2)
        generative.final_layer = nn.Linear(2, 2)
        generative.action_projection = nn.Linear(2, 2)
        self.stages = nn.ModuleList([visual, generative])


class _PartitionAlgo:
    def __init__(self):
        self.nets = nn.ModuleDict({"pipeline": _PartitionPolicy()})


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(1, 1, bias=False)
        self.content_projection = nn.Linear(1, 1, bias=False)


class _SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.domains = ("demo",)
        self.denoising_module = _Backbone()
        self.output_norm = nn.Linear(1, 1, bias=False)
        self.condition_projection = nn.Linear(1, 1, bias=False)
        self.null_condition_inputs = nn.ParameterDict(
            {"demo": nn.Parameter(torch.full((1,), 0.5))}
        )
        self.domain_embeddings = nn.ParameterDict(
            {"demo": nn.Parameter(torch.full((1,), 0.5))}
        )
        self.action_context_projections = nn.ModuleDict(
            {"demo": nn.Linear(1, 1, bias=False)}
        )

    @staticmethod
    def _resolve_domain(_value=None):
        return "demo"


class _SeparateEncoder(_SharedEncoder):
    def __init__(self):
        super().__init__()
        self.tokenization_module = self.denoising_module
        self.denoising_module = _Backbone()
        self.denoising_output_norm = nn.Linear(1, 1, bias=False)
        self.tokenization_condition_projection = nn.Linear(1, 1, bias=False)
        self.tokenization_null_condition_inputs = nn.ParameterDict(
            {"demo": nn.Parameter(torch.full((1,), 0.5))}
        )
        self.denoising_domain_embeddings = nn.ParameterDict(
            {"demo": nn.Parameter(torch.full((1,), 0.5))}
        )


class _UniteStage(nn.Module):
    def __init__(self, shared: bool, dropout_probability: float = 0.0):
        super().__init__()
        self.generative_encoder = _SharedEncoder() if shared else _SeparateEncoder()
        self.condition_dropout_probability = float(dropout_probability)


class _MetricPipeline(nn.Module):
    def __init__(self, shared: bool, dropout_probability: float):
        super().__init__()
        self.stages = nn.ModuleList(
            [_UniteStage(shared, dropout_probability=dropout_probability)]
        )
        self.anchor = nn.Parameter(torch.ones(()))


class _MetricAlgo:
    def __init__(self, *, shared: bool = True, dropout_probability: float = 0.0):
        self.nets = nn.ModuleDict(
            {
                "pipeline": _MetricPipeline(
                    shared, dropout_probability=dropout_probability
                )
            }
        )
        self.device = torch.device("cpu")

    @property
    def pipeline(self):
        return self.nets["pipeline"]

    def process_batch_for_training(self, batch):
        return batch

    def forward_training(self, batch):
        anchor = self.pipeline.anchor
        results = OrderedDict()
        for source, values in batch.items():
            count = int(values["count"])
            results[source] = {
                "target": anchor.new_zeros((count, 1, 1)),
                "embodiment": "demo",
                "loss/unite_reconstruction": anchor
                * anchor.new_tensor(values["reconstruction"]),
                "loss/unite_latent": anchor * anchor.new_tensor(values["flow"]),
                "log/unite_reconstruction_l1": anchor.new_tensor(values["l1"]),
            }
        return results


def _wrapper(*, shared=True, dropout_probability=0.0):
    return ReleasedUniteModelWrapper(
        pipeline=_MetricAlgo(shared=shared, dropout_probability=dropout_probability)
    )


def _batch(*rows):
    return OrderedDict(
        (
            f"source_{index}",
            {
                "count": count,
                "reconstruction": reconstruction,
                "flow": flow,
                "l1": l1,
            },
        )
        for index, (count, reconstruction, flow, l1) in enumerate(rows)
    )


def _optimizer(model) -> ReleasedUniteCompositeOptimizer:
    return ReleasedUniteCompositeOptimizer(
        model.named_parameters(),
        lr=1.0e-3,
        adamw_weight_decay=0.0,
        muon_weight_decay=0.0,
    )


def _quadratic_loss(model):
    return sum(parameter.square().sum() for parameter in model.parameters())


def _parameter_loss(named, offset: float):
    return sum((parameter - offset).square().sum() for _, parameter in named)


def test_parameter_partition_is_complete_disjoint_and_name_stable():
    model = _PartitionAlgo().nets
    named = tuple(model.named_parameters(prefix="nets", remove_duplicate=True))
    adamw, muon = partition_released_unite_parameters(named)
    adamw_names = {name for name, _ in adamw}
    muon_names = {name for name, _ in muon}
    assert adamw_names.isdisjoint(muon_names)
    assert adamw_names | muon_names == {name for name, _ in named}
    assert len(named) == len({id(parameter) for _, parameter in named})
    assert any("img_encoders" in name for name in adamw_names)
    assert any("content_projection" in name for name in adamw_names)
    assert any("condition_projection" in name for name in adamw_names)
    assert any("blocks.0.weight" in name for name in muon_names)
    assert any("action_projection.weight" in name for name in muon_names)


def test_composite_optimizer_step_and_state_round_trip():
    torch.manual_seed(7)
    model_a = _PartitionAlgo().nets
    optimizer_a = _optimizer(model_a)
    _quadratic_loss(model_a).backward()
    optimizer_a.step()
    optimizer_a.zero_grad()
    assert isinstance(optimizer_a.muon, VendoredTorchMuon)
    assert optimizer_a.muon.state
    state = copy.deepcopy(optimizer_a.state_dict())

    model_b = _PartitionAlgo().nets
    model_b.load_state_dict(model_a.state_dict())
    optimizer_b = _optimizer(model_b)
    optimizer_b.load_state_dict(state)
    _quadratic_loss(model_a).backward()
    _quadratic_loss(model_b).backward()
    optimizer_a.step()
    optimizer_b.step()
    for left, right in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_released_scheduler_intentionally_keeps_phase_two_beyond_run():
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0e-4)
    scheduler = released_unite_two_stage_scheduler(
        optimizer,
        warmup_steps=8_000,
        decay_start_1_steps=12_000,
        decay_end_1_steps=20_000,
        decay_start_2_steps=1_200_000,
        decay_end_2_steps=1_200_000,
        base_lr_1=1.0e-4,
        base_lr_2=5.0e-5,
        final_lr=1.0e-5,
    )
    multiplier = scheduler.lr_lambdas[0]
    assert multiplier(0) == 0.0
    assert multiplier(8_000) == 1.0
    assert multiplier(12_000) == 1.0
    assert multiplier(20_000) == 0.5
    assert multiplier(240_000) == 0.5
    assert multiplier(1_200_000) == pytest.approx(0.1)


def test_wrapper_has_no_direct_optimizer_or_topology_compatibility():
    wrapper = _wrapper()
    with pytest.raises(RuntimeError, match="requires config_tree"):
        wrapper.configure_optimizers()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ReleasedUniteModelWrapper(pipeline=_MetricAlgo(), share_encoder_denoiser=True)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ReleasedUniteModelWrapper(pipeline=_MetricAlgo(), optimizer=lambda *_args: None)


def test_training_components_are_sample_weighted_and_finite(monkeypatch):
    wrapper = _wrapper()
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **kwargs: logged.setdefault(name, (value, kwargs)),
    )
    loss = wrapper.training_step(_batch((1, 1.0, 2.0, 3.0), (3, 5.0, 6.0, 7.0)), 0)

    assert float(loss) == pytest.approx(9.0)
    expected = {
        "Train/UNITE/TotalLoss": 9.0,
        "Train/UNITE/ReconstructionLoss": 4.0,
        "Train/UNITE/FlowLoss": 5.0,
        "Train/UNITE/ReconstructionL1": 6.0,
    }
    assert set(logged) == set(expected)
    for name, expected_value in expected.items():
        value, kwargs = logged[name]
        assert float(value) == pytest.approx(expected_value)
        assert kwargs == {
            "on_step": True,
            "on_epoch": True,
            "sync_dist": False,
            "batch_size": 4,
        }

    with pytest.raises(RuntimeError, match="Non-finite UNITE metric"):
        wrapper.training_step(_batch((1, float("nan"), 1.0, 1.0)), 1)


def test_component_reduction_weights_remote_ranks(monkeypatch):
    local = OrderedDict(
        TotalLoss=torch.tensor(3.0),
        ReconstructionLoss=torch.tensor(1.0),
        FlowLoss=torch.tensor(2.0),
        ReconstructionL1=torch.tensor(4.0),
    )
    remote_means = torch.tensor((9.0, 4.0, 5.0, 7.0), dtype=torch.float64)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def add_remote(payload, op):
        assert op is torch.distributed.ReduceOp.SUM
        payload.add_(torch.cat((remote_means * 3, torch.tensor((3.0,)))))

    monkeypatch.setattr(torch.distributed, "all_reduce", add_remote)
    reduced, count = ReleasedUniteModelWrapper._distributed_weighted_components(
        local, 1
    )
    assert count == 4
    assert float(reduced["TotalLoss"]) == pytest.approx(7.5)
    assert float(reduced["ReconstructionLoss"]) == pytest.approx(3.25)
    assert float(reduced["FlowLoss"]) == pytest.approx(4.25)
    assert float(reduced["ReconstructionL1"]) == pytest.approx(6.25)


def test_validation_components_weight_sources_batches_and_ranks(monkeypatch):
    wrapper = _wrapper()
    wrapper.on_validation_start()
    wrapper.validation_step(_batch((1, 1.0, 2.0, 3.0), (3, 5.0, 6.0, 7.0)), 0)
    wrapper.validation_step(_batch((2, 9.0, 10.0, 11.0)), 1)

    remote_means = torch.tensor((6.0, 2.0, 4.0, 5.0), dtype=torch.float64)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def add_remote(payload, op):
        assert op is torch.distributed.ReduceOp.SUM
        payload.add_(torch.cat((remote_means * 4, torch.tensor((4.0,)))))

    monkeypatch.setattr(torch.distributed, "all_reduce", add_remote)
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log_dict",
        lambda values, **kwargs: logged.update({"values": values, "kwargs": kwargs}),
    )
    wrapper.on_validation_epoch_end()

    assert logged["kwargs"] == {
        "on_step": False,
        "on_epoch": True,
        "sync_dist": False,
    }
    values = logged["values"]
    assert set(values) == {
        "Valid/UNITE/TotalLoss",
        "Valid/UNITE/ReconstructionLoss",
        "Valid/UNITE/FlowLoss",
        "Valid/UNITE/ReconstructionL1",
    }
    assert float(values["Valid/UNITE/TotalLoss"]) == pytest.approx(9.8)
    assert float(values["Valid/UNITE/ReconstructionLoss"]) == pytest.approx(4.2)
    assert float(values["Valid/UNITE/FlowLoss"]) == pytest.approx(5.6)
    assert float(values["Valid/UNITE/ReconstructionL1"]) == pytest.approx(6.6)
    assert not any("Native" in name or "Energy" in name for name in values)


def test_gradient_telemetry_runs_on_update_completing_each_hundred_steps(monkeypatch):
    wrapper = _wrapper()
    monkeypatch.setattr(wrapper, "log", lambda *_args, **_kwargs: None)
    completed_steps = {"value": 0}
    monkeypatch.setattr(
        ReleasedUniteModelWrapper,
        "global_step",
        property(lambda _self: completed_steps["value"]),
    )
    measured = []
    monkeypatch.setattr(
        wrapper,
        "_measure_topology_gradients",
        lambda *_args: measured.append(completed_steps["value"] + 1),
    )
    for step in (0, 1, 98, 99, 100, 198, 199, 200):
        completed_steps["value"] = step
        wrapper.training_step(_batch((1, 1.0, 1.0, 1.0)), step)
    assert measured == [100, 200]


def test_shared_telemetry_uses_true_intersection_without_touching_grad(monkeypatch):
    wrapper = _wrapper(shared=True, dropout_probability=0.0)
    _, encoder, shared = wrapper._unite_topology()
    assert shared
    selected = wrapper._shared_named_parameters(
        encoder, ("demo",), include_null_input=False
    )
    selected_names = {name for name, _ in selected}
    assert "domain_embeddings.demo" in selected_names
    assert not any("content_projection" in name for name in selected_names)
    assert not any("action_context_projections" in name for name in selected_names)
    assert "null_condition_inputs.demo" not in selected_names
    with_null = wrapper._shared_named_parameters(
        encoder, ("demo",), include_null_input=True
    )
    assert "null_condition_inputs.demo" in {name for name, _ in with_null}

    for _, parameter in selected:
        parameter.data.fill_(0.5)
        parameter.grad = torch.full_like(parameter, 7.0)
    before = {id(parameter): parameter.grad.clone() for _, parameter in selected}
    reconstruction = _parameter_loss(selected, 0.0)
    flow = _parameter_loss(selected, 2.0)
    predictions = {"source": {"embodiment": "demo"}}
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **_kwargs: logged.setdefault(name, value),
    )
    wrapper._measure_topology_gradients(reconstruction, flow, predictions)

    assert set(logged) == {
        "log/unite_gradient_cosine",
        "log/unite_recon_grad_norm",
        "log/unite_denoise_grad_norm",
        "log/unite_gradient_parameter_count",
        "log/unite_gradient_tensor_count",
    }
    assert float(logged["log/unite_gradient_cosine"]) == pytest.approx(-1.0)
    for _, parameter in selected:
        torch.testing.assert_close(parameter.grad, before[id(parameter)])
    (reconstruction + flow).backward()


def test_separate_telemetry_is_disjoint_and_does_not_touch_grad(monkeypatch):
    wrapper = _wrapper(shared=False, dropout_probability=0.0)
    _, encoder, shared = wrapper._unite_topology()
    assert not shared
    tokenizer, denoiser = wrapper._separate_named_parameters(
        encoder, ("demo",), include_denoiser_null_input=False
    )
    assert {id(parameter) for _, parameter in tokenizer}.isdisjoint(
        {id(parameter) for _, parameter in denoiser}
    )
    assert "null_condition_inputs.demo" not in {name for name, _ in denoiser}
    for _, parameter in (*tokenizer, *denoiser):
        parameter.data.fill_(0.5)
        parameter.grad = torch.full_like(parameter, 7.0)
    before = {
        id(parameter): parameter.grad.clone()
        for _, parameter in (*tokenizer, *denoiser)
    }
    reconstruction = _parameter_loss(tokenizer, 0.0)
    flow = _parameter_loss(denoiser, 2.0)
    logged = {}
    monkeypatch.setattr(
        wrapper,
        "log",
        lambda name, value, **_kwargs: logged.setdefault(name, value),
    )
    wrapper._measure_topology_gradients(
        reconstruction, flow, {"source": {"embodiment": "demo"}}
    )

    assert set(logged) == {
        "log/unite_tokenizer_recon_grad_norm",
        "log/unite_denoiser_flow_grad_norm",
    }
    for _, parameter in (*tokenizer, *denoiser):
        torch.testing.assert_close(parameter.grad, before[id(parameter)])
    (reconstruction + flow).backward()


def test_topology_and_gradient_gates_fail_loudly():
    wrapper = _wrapper(shared=True)
    wrapper._configured_share_encoder_denoiser = False
    with pytest.raises(RuntimeError, match="disagrees"):
        wrapper._unite_topology()

    with pytest.raises(RuntimeError, match="zero or non-finite"):
        wrapper._gradient_norm((torch.zeros(2),))
    with pytest.raises(RuntimeError, match="zero or non-finite"):
        wrapper._gradient_norm((torch.full((2,), float("nan")),))

    wrapper._configured_share_encoder_denoiser = None
    _, encoder, _ = wrapper._unite_topology()
    named = wrapper._shared_named_parameters(
        encoder, ("demo",), include_null_input=False
    )
    disconnected = named[0][1].square().sum()
    with pytest.raises(RuntimeError, match="not have been used in the graph"):
        wrapper._measure_shared_gradients(disconnected, disconnected, named)


def test_training_entry_resolves_only_model_wrapper_subclasses():
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": (
                    "egomimic.pl_utils.pl_model_unite_released."
                    "ReleasedUniteModelWrapper"
                )
            }
        }
    )
    assert _resolve_model_wrapper_class(cfg) is ReleasedUniteModelWrapper
    cfg.model._target_ = "torch.nn.Linear"
    with pytest.raises(TypeError, match="ModelWrapper subclass"):
        _resolve_model_wrapper_class(cfg)


def test_training_entry_forwards_enable_grad_norm(monkeypatch):
    class CaptureWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "unused.for.this.unit.test",
                "enable_grad_norm": False,
            }
        }
    )
    monkeypatch.setattr(
        "egomimic.trainHydra._resolve_model_wrapper_class",
        lambda _cfg: CaptureWrapper,
    )

    wrapper = _instantiate_model_wrapper(cfg)

    assert wrapper.kwargs["enable_grad_norm"] is False
