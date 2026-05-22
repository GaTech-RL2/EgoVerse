"""Smoke test BCRNNOuterStage:
  * padded forward + backward
  * packed forward (cu_seqlens) + backward
  * single-step rollout (init_step_state -> step)

No norm_stats / data pipeline — just the OuterStage in isolation.
"""

import torch

from egomimic.algo.bcrnn import BCRNNOuterStage
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.models.hnet_nets.image_encoders import SimpleConv


def _make_stage(d_cond=64, hidden=128, layers=2):
    cond = CondEncoderModule(
        d_cond=d_cond,
        obs_specs={
            "state_agent_obj": {"input_dim": 5, "embed_dim": 32, "widths": [64]},
        },
        img_encoders={
            "front_img_1": SimpleConv(
                in_channels=3,
                channels=[16, 32],
                kernel_size=3,
                stride=2,
                embed_dim=32,
                norm_groups=8,
            ),
        },
        cond_proj_widths=[d_cond],
    )
    stage = BCRNNOuterStage(
        action_dim=2,
        cond_encoder=cond,
        rnn_hidden_dim=hidden,
        rnn_num_layers=layers,
        rnn_type="LSTM",
    )
    return stage


class _Ctx:
    def __init__(self, cu=None):
        self.cu_seqlens = cu
        self.rnn_state = None


def test_padded():
    torch.manual_seed(0)
    stage = _make_stage()
    B, T = 3, 7
    obs = {
        "state_agent_obj": torch.randn(B, T, 5),
        "front_img_1": torch.rand(B, T, 3, 16, 16),
    }
    actions = torch.randn(B, T, 2)
    batch = {"__obs": obs, "actions": actions}
    ctx = _Ctx()
    h = stage(batch, ctx)
    pred = batch["pred_action"]
    assert pred.shape == (B, T, 2), pred.shape
    loss = (pred - actions).pow(2).mean()
    loss.backward()
    # all params should have a grad
    missing = [n for n, p in stage.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad: {missing[:5]}"
    print(f"[OK] padded forward+backward: loss={loss.item():.4f}, h={tuple(h.shape)}")


def test_packed():
    torch.manual_seed(1)
    stage = _make_stage()
    seq_lens = [4, 6, 3]
    T_total = sum(seq_lens)
    cu = torch.tensor([0, 4, 10, 13], dtype=torch.long)
    obs = {
        "state_agent_obj": torch.randn(T_total, 5),
        "front_img_1": torch.rand(T_total, 3, 16, 16),
    }
    actions = torch.randn(T_total, 2)
    batch = {"__obs": obs, "actions": actions}
    ctx = _Ctx(cu=cu)
    h = stage(batch, ctx)
    pred = batch["pred_action"]
    assert pred.shape == (T_total, 2), pred.shape
    loss = (pred - actions).pow(2).mean()
    loss.backward()
    missing = [n for n, p in stage.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad: {missing[:5]}"
    print(f"[OK] packed forward+backward: loss={loss.item():.4f}")


def test_step():
    torch.manual_seed(2)
    stage = _make_stage()
    stage.eval()
    state = stage.init_step_state(batch_size=1, device=torch.device("cpu"))
    actions = []
    for t in range(5):
        obs = {
            "state_agent_obj": torch.randn(1, 1, 5),
            "front_img_1": torch.rand(1, 1, 3, 16, 16),
        }
        a = stage.step(state, obs, t)
        actions.append(a)
    a_seq = torch.cat(actions, dim=1)
    assert a_seq.shape == (1, 5, 2)
    # Hidden state should have evolved (non-zero after a few steps).
    h, c = state["rnn_state"]
    assert h.abs().sum().item() > 0
    print(f"[OK] step rollout 5 ticks: a_seq={tuple(a_seq.shape)}")


if __name__ == "__main__":
    test_padded()
    test_packed()
    test_step()
    print("\nAll smoke tests passed.")
