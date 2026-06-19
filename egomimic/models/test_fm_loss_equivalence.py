"""Equivalence tests for the Pi and HPT training losses.

Both model families train with the *same* flow-matching velocity-prediction loss,
but the math lives in two separate implementations:

  Pi : external/openpi/src/openpi/models_pytorch/pi0_pytorch.py
       ``PI0Pytorch.forward``                (interpolation/target/MSE, ~lines 326-373)
  HPT: egomimic/models/fm_policy.py
       ``FMPolicy.predict`` + ``DenoisingPolicy.loss_fn``

Given actions ``a``, noise ``e ~ N(0, 1)`` and time ``t ~ Beta(1.5, 1.0)*0.999+0.001``
both must compute:

    x_t  = t * e + (1 - t) * a      # noised sample fed to the velocity model
    u_t  = e - a                    # target (constant) velocity
    loss = MSE(v_t, u_t)            # v_t is the model's velocity prediction

If these conventions ever drift apart (e.g. a flipped sign on ``u_t`` or a swapped
interpolation), train/val losses stop being comparable across the two families. These
tests pin the convention by running the *real* loss code of both paths with identical
inputs and a shared, fixed velocity output ``V`` (the differing neural nets are stubbed
out so only the loss formulation is compared).
"""

import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.fm_policy import FMPolicy

try:
    import openpi.models_pytorch.pi0_pytorch as pim

    _HAS_OPENPI = True
except Exception:  # pragma: no cover - exercised only when openpi is absent
    pim = None
    _HAS_OPENPI = False

requires_openpi = pytest.mark.skipif(not _HAS_OPENPI, reason="openpi not importable")

# Small shapes; H is an arbitrary stubbed hidden width for the Pi network surface.
B, T, D, H = 3, 4, 7, 5
EMB = "eva_bimanual"


def _rand(shape, seed):
    """Deterministic float32 tensor that does not depend on torch's RNG.

    The HPT path is driven by monkeypatching ``torch.randn`` (see below), so inputs
    must be built with a torch-RNG-independent source.
    """
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal(size=shape).astype(np.float32))


def _fixed_inputs():
    """Shared (actions, noise, time, V) ground truth for an equivalence comparison."""
    actions = _rand((B, T, D), seed=1)
    noise = _rand((B, T, D), seed=2)
    rng = np.random.default_rng(3)
    time = torch.from_numpy(rng.uniform(0.05, 0.95, size=(B,)).astype(np.float32))
    velocity = _rand((B, T, D), seed=4)  # the shared, fixed model output v_t
    return actions, noise, time, velocity


def _fm_reference(actions, noise, time):
    """The agreed flow-matching convention, written once as the spec."""
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions
    u_t = noise - actions
    return x_t, u_t


class _StubVelocityModel(nn.Module):
    """Stands in for the HPT velocity net: returns a fixed ``V`` and captures inputs."""

    def __init__(self, velocity):
        super().__init__()
        self.velocity = velocity
        self.captured = {}

    def forward(self, x_t, time, global_cond):
        self.captured["x_t"] = x_t.detach().clone()
        self.captured["time"] = time.detach().clone()
        return self.velocity


def _run_hpt(actions, noise, time, velocity, monkeypatch):
    """Run the real ``FMPolicy.predict`` + ``loss_fn`` with injected noise/time.

    ``predict`` samples noise/time internally, so we monkeypatch the two RNG sources
    to inject the shared ground-truth values, then read back what the model received.
    """
    model = _StubVelocityModel(velocity)
    policy = FMPolicy(model=model, action_horizon=T, infer_ac_dims={EMB: D})
    policy.eval()

    # Inject the shared noise: predict calls ``torch.randn(actions.shape, ...)``.
    monkeypatch.setattr(torch, "randn", lambda *a, **k: noise)
    # Inject the shared time: predict draws ``Beta(a, b).sample(...)`` then applies
    # ``t = raw*0.999 + 0.001``. Hand back ``raw`` so the transform yields ``time``.
    raw = (time - 0.001) / 0.999
    monkeypatch.setattr(
        torch.distributions,
        "Beta",
        lambda a, b: types.SimpleNamespace(sample=lambda shape: raw),
    )

    pred, target = policy.predict(actions, global_cond=torch.zeros(B, H))
    loss = policy.loss_fn(pred, target)
    return {
        "x_t": model.captured["x_t"],
        "time": model.captured["time"],
        "pred": pred,
        "target": target,
        "loss": loss,
    }


def _run_pi(actions, noise, time, velocity):
    """Run the real ``PI0Pytorch.forward`` loss math with the transformer stubbed out.

    We bypass ``__init__`` (which builds PaliGemma) and supply only the network surface
    ``forward`` touches, so the genuine interpolation/target/MSE lines execute while the
    expensive model is replaced by a fixed ``V``. ``forward`` returns the per-element MSE.
    """
    pi = pim.PI0Pytorch.__new__(pim.PI0Pytorch)
    nn.Module.__init__(pi)  # set up nn.Module's attribute machinery
    captured = {}

    pi.config = types.SimpleNamespace(action_horizon=T)
    pi._preprocess_observation = lambda observation, train=True: (None,) * 5
    pi.embed_prefix = lambda *a: (
        torch.zeros(B, 1, H),
        torch.ones(B, 1, dtype=torch.bool),
        torch.zeros(B, 1),
    )

    def _embed_suffix(state, noisy_actions, timestep):
        captured["x_t"] = noisy_actions.detach().clone()  # x_t computed by forward
        return (
            torch.zeros(B, T, H),
            torch.ones(B, T, dtype=torch.bool),
            torch.zeros(B, T),
            None,
        )

    pi.embed_suffix = _embed_suffix
    # forward checks q_proj.weight.dtype against bfloat16; float32 skips the cast branch.
    pi.paligemma_with_expert = types.SimpleNamespace(
        paligemma=types.SimpleNamespace(
            language_model=types.SimpleNamespace(
                layers=[
                    types.SimpleNamespace(
                        self_attn=types.SimpleNamespace(
                            q_proj=types.SimpleNamespace(weight=torch.zeros(1))
                        )
                    )
                ]
            )
        ),
        forward=lambda **kw: ((None, torch.cat(kw["inputs_embeds"], dim=1)), None),
    )
    pi._prepare_attention_masks_4d = lambda att_2d_masks: att_2d_masks
    pi._apply_checkpoint = lambda func, *a, **k: func(*a, **k)
    pi.action_out_proj = lambda suffix_out: velocity  # fixed v_t

    loss_elementwise = pi.forward(None, actions, noise=noise, time=time)
    return {"x_t": captured["x_t"], "loss_elementwise": loss_elementwise}


def test_hpt_predict_matches_fm_reference(monkeypatch):
    actions, noise, time, velocity = _fixed_inputs()
    out = _run_hpt(actions, noise, time, velocity, monkeypatch)
    x_ref, u_ref = _fm_reference(actions, noise, time)

    torch.testing.assert_close(out["time"], time)
    torch.testing.assert_close(out["x_t"], x_ref)
    torch.testing.assert_close(out["target"], u_ref)  # u_t = noise - actions
    torch.testing.assert_close(out["pred"], velocity)
    torch.testing.assert_close(out["loss"], F.mse_loss(velocity, u_ref))


@requires_openpi
def test_pi_forward_matches_fm_reference():
    actions, noise, time, velocity = _fixed_inputs()
    out = _run_pi(actions, noise, time, velocity)
    x_ref, u_ref = _fm_reference(actions, noise, time)

    torch.testing.assert_close(out["x_t"], x_ref)
    # PI returns per-element MSE(u_t, v_t) with reduction="none".
    torch.testing.assert_close(out["loss_elementwise"], (u_ref - velocity) ** 2)
    torch.testing.assert_close(
        out["loss_elementwise"].mean(), F.mse_loss(velocity, u_ref)
    )


@requires_openpi
def test_pi_and_hpt_losses_match(monkeypatch):
    """The headline check: identical inputs + identical v_t => identical loss."""
    actions, noise, time, velocity = _fixed_inputs()
    hpt = _run_hpt(actions, noise, time, velocity, monkeypatch)
    pi = _run_pi(actions, noise, time, velocity)

    # Same noised sample (interpolation convention) ...
    torch.testing.assert_close(pi["x_t"], hpt["x_t"])
    # ... and same scalar loss (target + MSE convention).
    torch.testing.assert_close(pi["loss_elementwise"].mean(), hpt["loss"])


@requires_openpi
def test_validation_loss_value_matches(monkeypatch):
    """The per-embodiment validation loss reduces to the same scalar for both.

    ``forward_eval`` turns the flow-matching loss into one scalar ``{emb}_loss``:
    Pi (egomimic/algo/pi.py) does ``forward(obs, act).mean()``; HPT
    (egomimic/algo/hpt.py) does ``compute_loss(...)`` which, for the ``bc_flow``
    FMPolicy head, is ``loss_fn`` == mean MSE. Same inputs => same val loss.
    """
    actions, noise, time, velocity = _fixed_inputs()
    hpt = _run_hpt(actions, noise, time, velocity, monkeypatch)
    pi = _run_pi(actions, noise, time, velocity)

    _, u_ref = _fm_reference(actions, noise, time)
    expected = F.mse_loss(velocity, u_ref)  # mean velocity MSE

    torch.testing.assert_close(hpt["loss"], expected)  # HPT compute_loss (loss_fn mean)
    torch.testing.assert_close(
        pi["loss_elementwise"].mean(), expected
    )  # Pi forward().mean()


@requires_openpi
def test_time_sampler_matches():
    """Pi's ``sample_time`` and HPT's beta branch draw from the same distribution."""
    bare_pi = pim.PI0Pytorch.__new__(pim.PI0Pytorch)
    dev = torch.device("cpu")

    torch.manual_seed(0)
    t_pi = bare_pi.sample_time(B, dev)

    # HPT (fm_policy.predict, beta branch): Beta(1.5, 1.0).sample()*0.999 + 0.001.
    torch.manual_seed(0)
    t_hpt = torch.distributions.Beta(1.5, 1.0).sample((B,)) * 0.999 + 0.001

    torch.testing.assert_close(t_pi, t_hpt)
