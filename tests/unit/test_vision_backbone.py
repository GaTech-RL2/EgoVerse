"""The frozen-backbone eval-mode fix and the separate backbone LR group.

Covers ``ResNet`` (unchanged forward, BatchNorm in eval when frozen,
``backbone_parameters``), ``ModelWrapper.configure_optimizers`` param groups,
and the flagship model config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torchvision
from omegaconf import OmegaConf
from torch import nn

from egomimic.models.hpt_nets import ResNet
from egomimic.pl_utils.pl_model import ModelWrapper

# --------------------------------------------------------------------------
# ResNet
# --------------------------------------------------------------------------


def _resnet_stem(**kwargs):
    torch.manual_seed(0)
    return ResNet(output_dim=16, **kwargs)


def test_resnet_forward_matches_reference_implementation():
    """The baseline image path is bit-identical: the stem's output equals the
    torchvision backbone followed by the stem's projection."""
    stem = _resnet_stem().eval()
    x = torch.rand(2, 1, 1, 3, 96, 128)
    with torch.no_grad():
        out = stem(x)

    backbone = nn.Sequential(
        *list(torchvision.models.resnet18(weights="DEFAULT").children())[:-2]
    ).eval()
    with torch.no_grad():
        feat = backbone(x.view(-1, 3, 96, 128))
        expected = stem.proj(feat.view(2, feat.shape[1], -1).transpose(1, 2))
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_resnet_frozen_backbone_batchnorm_stays_in_eval():
    """0.15: a frozen ResNet must not update / use batch statistics."""
    stem = _resnet_stem(freeze_backbone=True)
    stem.train()
    assert stem.training
    bns = [
        m for m in stem.net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    assert bns
    assert all(not m.training for m in bns)
    assert all(not p.requires_grad for p in stem.net.parameters())
    # an unfrozen stem keeps the default behaviour
    trainable = _resnet_stem()
    trainable.train()
    assert all(
        m.training
        for m in trainable.net.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    )


def test_resnet_frozen_backbone_running_stats_do_not_drift():
    stem = _resnet_stem(freeze_backbone=True)
    stem.train()
    bn = next(
        m for m in stem.net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    )
    before = bn.running_mean.clone()
    with torch.no_grad():
        stem(torch.rand(2, 1, 1, 3, 96, 128) * 5.0)
    assert torch.equal(bn.running_mean, before)


def test_resnet_backbone_parameters_are_the_net():
    stem = _resnet_stem()
    backbone = {id(p) for p in stem.backbone_parameters()}
    assert backbone == {id(p) for p in stem.net.parameters()}
    assert all(id(p) not in backbone for p in stem.proj.parameters())


def test_resnet_backbone_parameters_with_multiple_copies():
    stem = ResNet(output_dim=16, num_of_copy=2)
    backbone = {id(p) for p in stem.backbone_parameters()}
    assert backbone == {id(p) for p in stem.net.parameters()}


# --------------------------------------------------------------------------
# configure_optimizers
# --------------------------------------------------------------------------


class _StubEncoder(nn.Module):
    """Encoder with a pretrained-style backbone plus a trainable projection."""

    def __init__(self, frozen: bool = False):
        super().__init__()
        self.net = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)
        if frozen:
            for p in self.net.parameters():
                p.requires_grad = False

    def backbone_parameters(self):
        return list(self.net.parameters())


def _stub_wrapper(backbone_lr_scale=None, frozen=False):
    policy = nn.Module()
    policy.encoders = nn.ModuleDict(
        {
            "front_img_1": _StubEncoder(frozen=frozen),
            "other": nn.Linear(4, 4),  # no backbone_parameters -> main group
        }
    )
    policy.trunk = nn.Linear(4, 4)
    root = nn.Module()
    root.nets = nn.ModuleDict({"policy": policy})

    model_cfg = {
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "_partial_": True,
            "lr": 5e-5,
            "weight_decay": 0.0001,
        },
        "scheduler": None,
    }
    if backbone_lr_scale is not None:
        model_cfg["backbone_lr_scale"] = backbone_lr_scale
    config_tree = OmegaConf.create({"model": model_cfg})
    # A LightningModule cannot be built without a trainer / datamodule here, so
    # call the real methods against the attributes they actually read.
    stub = SimpleNamespace(
        hparams=SimpleNamespace(config_tree=config_tree),
        _as_config=ModelWrapper._as_config,
        model=root,
        trainer=SimpleNamespace(model=root),
    )
    stub._backbone_param_groups = ModelWrapper._backbone_param_groups.__get__(stub)
    return stub, root


@pytest.mark.parametrize("scale", [None, 1.0])
def test_configure_optimizers_single_group_by_default(scale):
    stub, _ = _stub_wrapper(backbone_lr_scale=scale)
    out = ModelWrapper.configure_optimizers(stub)
    groups = out["optimizer"].param_groups
    assert len(groups) == 1
    assert groups[0]["lr"] == pytest.approx(5e-5)


def test_configure_optimizers_backbone_group():
    stub, root = _stub_wrapper(backbone_lr_scale=0.1)
    out = ModelWrapper.configure_optimizers(stub)
    groups = out["optimizer"].param_groups
    assert len(groups) == 2
    by_lr = {round(g["lr"], 12): g for g in groups}
    assert set(by_lr) == {round(5e-5, 12), round(5e-6, 12)}

    encoder = root.nets["policy"].encoders["front_img_1"]
    backbone_ids = {id(p) for p in encoder.backbone_parameters()}
    got_backbone = {id(p) for p in by_lr[round(5e-6, 12)]["params"]}
    main = {id(p) for p in by_lr[round(5e-5, 12)]["params"]}
    assert got_backbone == backbone_ids
    assert not (got_backbone & main)
    trainable = {id(p) for p in root.parameters() if p.requires_grad}
    assert got_backbone | main == trainable


def test_configure_optimizers_skips_frozen_backbone_params():
    stub, root = _stub_wrapper(backbone_lr_scale=0.1, frozen=True)
    out = ModelWrapper.configure_optimizers(stub)
    every = [p for g in out["optimizer"].param_groups for p in g["params"]]
    assert all(p.requires_grad for p in every)
    assert {id(p) for p in every} == {
        id(p) for p in root.parameters() if p.requires_grad
    }


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------

FLAGSHIP = "train_zarr_mecka_flagship_6d_hpt"


def test_base_model_config_declares_unit_backbone_lr_scale(compose_resolve):
    cfg = compose_resolve(FLAGSHIP, [])
    assert cfg.model.backbone_lr_scale == 1.0
    encoder = cfg.model.robomimic_model.encoder_specs.front_img_1
    assert encoder._target_ == "egomimic.models.hpt_nets.ResNet"
