import copy
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import OmegaConf

import egomimic.trainHydra as train_hydra
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pl_utils.pl_model_unite_released import ReleasedUniteModelWrapper
from egomimic.utils.unite_optim import (
    ReleasedUniteCompositeOptimizer,
    VendoredTorchMuon,
)


class _TinyVisualCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.projection = nn.Linear(2, 2)


class _VisualStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.img_encoders = nn.ModuleDict({"front_img_1": _TinyVisualCore()})


class _GenerativeStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.final_layer = nn.Linear(2, 2)
        self.content_projection = nn.Linear(2, 2)
        self.condition_projection = nn.Linear(2, 2)
        self.action_projection = nn.Linear(2, 2)


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([_VisualStage(), _GenerativeStage()])


class _TinyAlgo(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleDict({"policy": _TinyPolicy()})


class _TinyNormStats:
    def keys_of_type(self, key_type, emb_id):
        return {
            "proprio_keys": [],
            "lang_keys": [],
            "camera_keys": [],
            "action_keys": ["actions"],
        }[key_type]

    def is_key_with_embodiment(self, key, emb_id):
        return key == "actions"


def _tiny_pipeline_algo() -> PipelineAlgo:
    return PipelineAlgo(
        stages=[_VisualStage(), _GenerativeStage()],
        norm_stats=_TinyNormStats(),
        domains=["pushshapes_sim_u_socket"],
        ac_keys={"pushshapes_sim_u_socket": "actions"},
        device=torch.device("cpu"),
    )


def _optimizer(model: nn.Module) -> ReleasedUniteCompositeOptimizer:
    return ReleasedUniteCompositeOptimizer(
        model.named_parameters(),
        lr=1.0e-3,
        adamw_weight_decay=0.0,
        muon_weight_decay=0.0,
    )


def _quadratic_loss(model: nn.Module) -> torch.Tensor:
    return sum(parameter.square().sum() for parameter in model.parameters())


def test_train_hydra_honors_configured_wrapper_without_changing_default(
    monkeypatch,
):
    default_cfg = OmegaConf.create({"model": {"robomimic_model": {}}})
    released_cfg = OmegaConf.create(
        {
            "model": {
                "_target_": (
                    "egomimic.pl_utils.pl_model_unite_released."
                    "ReleasedUniteModelWrapper"
                ),
                "robomimic_model": {},
            }
        }
    )
    assert train_hydra._resolve_model_wrapper_class(default_cfg) is ModelWrapper
    assert (
        train_hydra._resolve_model_wrapper_class(released_cfg)
        is ReleasedUniteModelWrapper
    )

    captured = {}

    class _CapturingWrapper(ModelWrapper):
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        train_hydra.hydra.utils, "get_class", lambda target: _CapturingWrapper
    )
    norm_stats = SimpleNamespace(to_state=lambda: {"sentinel": True})
    wrapper = train_hydra._instantiate_model_wrapper(released_cfg, norm_stats)
    assert isinstance(wrapper, _CapturingWrapper)
    assert captured["norm_stats_state"] == {"sentinel": True}
    assert captured["scheduler_frequency"] == 1


def test_released_wrapper_constructs_optimizer_without_attached_trainer():
    model = _TinyAlgo()
    wrapper = ReleasedUniteModelWrapper(
        robomimic_model=model,
        optimizer=partial(ReleasedUniteCompositeOptimizer, lr=1.0e-3),
        scheduler=None,
    )
    configured = wrapper.configure_optimizers()
    assert isinstance(configured["optimizer"], ReleasedUniteCompositeOptimizer)


def test_released_wrapper_constructs_hydra_config_tree_optimizer():
    pipeline_algo = _tiny_pipeline_algo()
    assert not isinstance(pipeline_algo, nn.Module)
    wrapper = ReleasedUniteModelWrapper(
        robomimic_model=pipeline_algo,
        optimizer=None,
        scheduler=None,
    )
    wrapper.hparams.config_tree = OmegaConf.create(
        {
            "model": {
                "optimizer": {
                    "_target_": (
                        "egomimic.utils.unite_optim.ReleasedUniteCompositeOptimizer"
                    ),
                    "_partial_": True,
                    "lr": 1.0e-3,
                },
                "scheduler": None,
            }
        }
    )
    configured = wrapper.configure_optimizers()
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, ReleasedUniteCompositeOptimizer)
    expected_named = tuple(
        wrapper.nets.named_parameters(prefix="nets", remove_duplicate=True)
    )
    expected_names = {name for name, _ in expected_named}
    expected_ids = {id(parameter) for _, parameter in expected_named}
    adamw_names = set(optimizer.group_manifest["adamw_parameter_names"])
    muon_names = set(optimizer.group_manifest["muon_parameter_names"])
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert expected_names == adamw_names | muon_names
    assert adamw_names.isdisjoint(muon_names)
    assert len(expected_named) == len(expected_names) == len(expected_ids)
    assert len(optimizer_parameters) == len({id(p) for p in optimizer_parameters})
    assert {id(parameter) for parameter in optimizer_parameters} == expected_ids
    assert "nets.policy.stages.1.content_projection.weight" in adamw_names
    assert "nets.policy.stages.1.action_projection.weight" in muon_names


def test_visual_core_paths_and_non_matrix_weights_are_adamw():
    optimizer = _optimizer(_TinyAlgo())
    adamw_names = set(optimizer.group_manifest["adamw_parameter_names"])
    muon_names = set(optimizer.group_manifest["muon_parameter_names"])

    visual_prefix = "nets.policy.stages.0.encoder.img_encoders.front_img_1"
    assert f"{visual_prefix}.backbone.weight" in adamw_names
    # The 2D output projection is the regression guard: dimensionality alone
    # would incorrectly send this real VisualCore path to Muon.
    assert f"{visual_prefix}.projection.weight" in adamw_names
    assert "nets.policy.stages.1.final_layer.weight" in adamw_names
    # Clean-action content is the robot patch stem and follows patch_embed.
    assert "nets.policy.stages.1.content_projection.weight" in adamw_names
    # Pooled observation projection is an explicit modality-adapter exception.
    assert "nets.policy.stages.1.condition_projection.weight" in adamw_names
    # The final action matrix follows released decoder_pred and remains Muon-eligible.
    assert "nets.policy.stages.1.action_projection.weight" in muon_names
    assert "nets.policy.stages.1.blocks.0.weight" in muon_names
    assert adamw_names.isdisjoint(muon_names)


def test_composite_optimizer_step_and_state_roundtrip():
    torch.manual_seed(7)
    model_a = _TinyAlgo()
    optimizer_a = _optimizer(model_a)
    _quadratic_loss(model_a).backward()
    optimizer_a.step()
    optimizer_a.zero_grad()

    assert isinstance(optimizer_a.muon, VendoredTorchMuon)
    assert optimizer_a.muon.state
    state = copy.deepcopy(optimizer_a.state_dict())

    model_b = _TinyAlgo()
    model_b.load_state_dict(model_a.state_dict())
    optimizer_b = _optimizer(model_b)
    optimizer_b.load_state_dict(state)

    _quadratic_loss(model_a).backward()
    _quadratic_loss(model_b).backward()
    optimizer_a.step()
    optimizer_b.step()

    for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(parameter_a, parameter_b, rtol=0.0, atol=0.0)
    assert optimizer_b.state_dict()["group_manifest"] == state["group_manifest"]
