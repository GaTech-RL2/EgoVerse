import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.pl_utils.ema import WeightEMA
from egomimic.pl_utils.pl_model import ModelWrapper


def _ema(module, **kw):
    return WeightEMA(module.named_parameters(), **kw)


def test_decay_matches_rdt_schedule():
    ema = _ema(nn.Linear(2, 2))
    decays = []
    for _ in range(5):
        decays.append(ema.decay())
        ema.num_updates += 1
    assert decays[:2] == [0.0, 0.0]
    assert decays[2] == 1 - 2**-0.75
    ema.num_updates = 10**7
    assert ema.decay() == 0.9999


def test_update_tracks_weights():
    lin = nn.Linear(2, 2)
    ema = _ema(lin)
    ema.update()
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema.update()
    assert torch.equal(ema.shadow[0], lin.weight.detach())
    with torch.no_grad():
        lin.weight.fill_(3.0)
    decay = ema.update()
    assert torch.allclose(ema.shadow[0], torch.full((2, 2), decay + 3 * (1 - decay)))


def test_swap_round_trip():
    lin = nn.Linear(2, 2)
    raw = lin.weight.detach().clone()
    ema = _ema(lin)
    ema.shadow[0].fill_(7.0)
    ema.swap_in()
    assert torch.equal(lin.weight.detach(), torch.full((2, 2), 7.0))
    assert torch.equal(ema.raw_by_id()[id(lin.weight)], raw)
    ema.swap_out()
    assert torch.equal(lin.weight.detach(), raw)
    assert not ema.swapped


def test_state_round_trip():
    src = _ema(nn.Linear(2, 2))
    src.shadow[0].fill_(5.0)
    src.num_updates = 42
    dst = _ema(nn.Linear(2, 2))
    dst.load_state_dict(src.state_dict())
    assert torch.equal(dst.shadow[0], src.shadow[0])
    assert dst.num_updates == 42


class _Algo:
    def __init__(self):
        self.nets = nn.ModuleDict({"policy": nn.Linear(2, 2)})


def _wrapper(monkeypatch, ema_cfg):
    monkeypatch.setattr(ModelWrapper, "_instantiate_model", lambda *a, **k: _Algo())
    model = {"robomimic_model": {}}
    if ema_cfg is not None:
        model["ema"] = ema_cfg
    return ModelWrapper(
        config_tree=OmegaConf.create({"model": model}), norm_stats_state=None
    )


def test_ema_is_off_unless_asked(monkeypatch):
    assert _wrapper(monkeypatch, None)._ema_cfg is None
    assert _wrapper(monkeypatch, {"enabled": False})._ema_cfg is None


def test_checkpoint_saved_mid_validation_keeps_raw_weights(monkeypatch):
    wrapper = _wrapper(monkeypatch, {"enabled": True})
    wrapper._ensure_ema()
    raw = wrapper.nets["policy"].weight.detach().clone()
    wrapper.ema.shadow[0].fill_(9.0)
    wrapper.ema.swap_in()
    checkpoint = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(checkpoint)
    keys = [k for k in checkpoint["state_dict"] if k.endswith("policy.weight")]
    assert keys
    for key in keys:
        assert torch.equal(checkpoint["state_dict"][key], raw)
    assert torch.equal(
        next(iter(checkpoint["ema"]["shadow"].values())), torch.full((2, 2), 9.0)
    )


def test_loaded_ema_survives_until_first_use(monkeypatch):
    src = _wrapper(monkeypatch, {"enabled": True})
    src._ensure_ema()
    src.ema.shadow[0].fill_(4.0)
    src.ema.num_updates = 11
    checkpoint = {"state_dict": src.state_dict()}
    src.on_save_checkpoint(checkpoint)

    dst = _wrapper(monkeypatch, {"enabled": True})
    dst.on_load_checkpoint(checkpoint)
    resaved = {"state_dict": dst.state_dict()}
    dst.on_save_checkpoint(resaved)
    assert resaved["ema"]["num_updates"] == 11
    dst._ensure_ema()
    assert dst.ema.num_updates == 11
    assert torch.equal(dst.ema.shadow[0], torch.full((2, 2), 4.0))
