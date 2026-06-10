"""Regression: the v-step helpers must dispatch on diffusion.objective.

2026-06-10 bug: pred_x0-trained parity model + pred_v-hardcoded conversion
negates x0 at zero-terminal SNR (mirror-through-center overlay). Oracle test:
feeding the TRUE x0 as the model output through one full-noise step must
return ~x0 under pred_x0, for all three call sites' shared math.
"""
import torch

from egomimic.algo.diffusion.algo import DFoT
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion


def _step(objective, level=999):
    diff = DiscreteDiffusion(action_dim=2, timesteps=1000, objective=objective,
                             max_beta=0.999, loss_weighting_strategy="constant")
    algo = object.__new__(DFoT)
    torch.manual_seed(0)
    x0 = torch.rand(1, 4, 5) * 2 - 1
    T = x0.shape[1]
    k = torch.full((T,), level, dtype=torch.long)
    kBT = k.unsqueeze(0)
    x_t = diff.q_sample(x0, kBT, noise=torch.randn_like(x0).clamp_(-20, 20))
    if objective == "pred_x0":
        v = x0  # oracle: model outputs the true x0
    elif objective == "pred_noise":
        v = diff.predict_noise_from_start(x_t, kBT, x0)
    else:
        v = diff.predict_v(x0, kBT, torch.randn_like(x0))
    nxt = torch.full((T,), -1, dtype=torch.long)  # one step to clean
    out = DFoT._struct_ddim_step(algo, diff, x_t, v, k, nxt)
    return x0, out


def test_pred_x0_oracle_recovers_x0():
    x0, out = _step("pred_x0")
    assert torch.allclose(out, x0, atol=1e-4), (out - x0).abs().max()


def test_pred_noise_oracle_recovers_x0():
    x0, out = _step("pred_noise", level=500)  # terminal level amplifies fp32 roundoff via 1/sqrt(abar); dispatch is what is under test
    assert torch.allclose(out, x0, atol=1e-3), (out - x0).abs().max()
