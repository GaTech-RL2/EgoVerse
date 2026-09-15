"""SigLIP vision stem, the frozen-backbone eval-mode fix and the separate
backbone LR group.

Covers: ``SigLIPStem`` (shape, pretrained weights untouched at construction,
the duck-typed pretrained-weight protocol, freeze semantics), ``ResNet``
(unchanged forward, BatchNorm in eval when frozen, ``backbone_parameters``),
``ModelWrapper.configure_optimizers`` param groups, and the two model configs.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torchvision
from omegaconf import OmegaConf
from torch import nn

from egomimic.models.hpt_nets import ResNet, _local_snapshot_if_offline
from egomimic.pl_utils.pl_model import ModelWrapper

SIGLIP = "google/siglip2-base-patch16-256"


def _snapshot_dir() -> str:
    try:
        return _local_snapshot_if_offline(SIGLIP)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"{SIGLIP} not in the local HF cache: {exc}")


@pytest.fixture(scope="module")
def siglip_stem():
    from egomimic.models.hpt_nets import SigLIPStem

    _snapshot_dir()
    return SigLIPStem(model_name=SIGLIP, output_dim=32)


def _snapshot_vision_weights(keys):
    """``vision_model.*`` tensors of the snapshot, re-keyed as ``tower.*``."""
    from safetensors import safe_open

    path = os.path.join(_snapshot_dir(), "model.safetensors")
    out = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith("vision_model."):
                continue
            name = "tower." + key[len("vision_model.") :]
            if name in keys:
                out[name] = handle.get_tensor(key)
    return out


# --------------------------------------------------------------------------
# SigLIPStem
# --------------------------------------------------------------------------


def test_siglip_stem_forward_shape(siglip_stem):
    cfg = siglip_stem.tower.config
    expected_tokens = (cfg.image_size // cfg.patch_size) ** 2
    x = torch.rand(2, 1, 1, 3, 360, 640)
    with torch.no_grad():
        out = siglip_stem(x)
    assert out.shape == (2, expected_tokens, 32), out.shape
    assert torch.isfinite(out).all()


def test_siglip_tower_holds_checkpoint_weights(siglip_stem):
    """The tower must carry the snapshot's weights straight after
    construction (nothing re-initialised while loading)."""
    live = siglip_stem.pretrained_state_dict()
    reference = _snapshot_vision_weights(set(live))
    assert set(live) == set(reference), (
        sorted(set(live) - set(reference))[:5],
        sorted(set(reference) - set(live))[:5],
    )
    for name, tensor in live.items():
        assert torch.equal(tensor, reference[name].to(tensor.dtype)), name


def test_siglip_pretrained_protocol(siglip_stem):
    assert siglip_stem._hpt_pretrained is True
    assert siglip_stem._hpt_pretrained_attrs == ("tower",)
    subs = siglip_stem.pretrained_submodules()
    assert len(subs) == 1 and subs[0] is siglip_stem.tower
    ref = siglip_stem.pretrained_reference_state_dict()
    assert ref is None or set(ref) == set(siglip_stem.pretrained_state_dict())


def test_siglip_backbone_parameters_are_the_tower(siglip_stem):
    backbone = {id(p) for p in siglip_stem.backbone_parameters()}
    assert backbone == {id(p) for p in siglip_stem.tower.parameters()}
    assert all(id(p) not in backbone for p in siglip_stem.proj.parameters())


def test_siglip_frozen_tower_stays_in_eval():
    from egomimic.models.hpt_nets import SigLIPStem

    _snapshot_dir()
    stem = SigLIPStem(model_name=SIGLIP, output_dim=32, freeze_backbone=True)
    stem.train()
    assert stem.training
    assert not stem.tower.training
    assert all(not p.requires_grad for p in stem.tower.parameters())
    assert all(p.requires_grad for p in stem.proj.parameters())


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
    assert stem._hpt_pretrained_attrs == ("net",)


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


def test_siglip_model_config(compose_resolve):
    cfg = compose_resolve(FLAGSHIP, ["model=hpt_bc_mecka_6d_300M_siglip"])
    rm = cfg.model.robomimic_model
    encoder = rm.encoder_specs.front_img_1
    assert encoder._target_ == "egomimic.models.hpt_nets.SigLIPStem"
    assert encoder.model_name == SIGLIP
    assert encoder.output_dim == 840
    assert encoder.freeze_backbone is False
    assert cfg.model.backbone_lr_scale == 0.1
    assert rm.eval_image_augs is None
    targets = [t._target_ for t in rm.train_image_augs.transforms]
    assert targets == ["torchvision.transforms.ColorJitter"]
    # the shared image stem and the trunk width still agree
    assert rm.shared_stem_specs.front_img_1.input_dim == rm.trunk.embed_dim


def test_siglip_stem_normalization_matches_checkpoint(siglip_stem):
    """The stem owns the normalization the SigLIP config prescribes, so the
    config's augs must not add a second one."""
    with open(
        os.path.join(_snapshot_dir(), "preprocessor_config.json"), encoding="utf-8"
    ) as handle:
        preproc = json.load(handle)
    assert torch.allclose(
        siglip_stem.image_mean.flatten(), torch.tensor(preproc["image_mean"])
    )
    assert torch.allclose(
        siglip_stem.image_std.flatten(), torch.tensor(preproc["image_std"])
    )
    assert siglip_stem.image_size == siglip_stem.tower.config.image_size
