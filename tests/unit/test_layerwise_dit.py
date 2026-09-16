"""Layer-wise cross-attention DiT (GR00T N1.5 / starVLA QwenPI port)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from egomimic.models.layerwise_dit import DiTBlock, LayerwiseDiT

B, N, S, H, HEADS, L = 2, 5, 7, 16, 2, 4


def _dit(**kwargs) -> LayerwiseDiT:
    torch.manual_seed(0)
    defaults = dict(num_layers=L, hidden=H, num_heads=HEADS, dropout=0.0)
    defaults.update(kwargs)
    return LayerwiseDiT(**defaults).eval()


def _inputs(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(B, N, H, generator=g)
    contexts = [torch.randn(B, S, H, generator=g) for _ in range(L)]
    mask = torch.ones(B, S, dtype=torch.bool)
    t = torch.rand(B, generator=g)
    return tokens, contexts, mask, t


def test_output_shape():
    tokens, contexts, mask, t = _inputs()
    assert _dit()(tokens, contexts, mask, t).shape == (B, N, H)


def test_interleave_alternates_cross_and_self_blocks():
    blocks = _dit(interleave_self_attention=True).blocks
    assert [b.is_cross for b in blocks] == [True, False, True, False]
    assert all(b.is_cross for b in _dit(interleave_self_attention=False).blocks)


def test_wrong_number_of_contexts_raises():
    tokens, contexts, mask, t = _inputs()
    with pytest.raises(ValueError):
        _dit()(tokens, contexts[:-1], mask, t)


def test_masked_context_positions_do_not_affect_the_output():
    dit = _dit()
    tokens, contexts, mask, t = _inputs()
    mask[:, -2:] = False
    ref = dit(tokens, contexts, mask, t)
    changed = [c.clone() for c in contexts]
    for c in changed:
        c[:, -2:] += 100.0
    assert torch.allclose(dit(tokens, changed, mask, t), ref, atol=1e-5)
    # sanity: an unmasked change does move the output
    changed[0][:, 0] += 100.0
    assert not torch.allclose(dit(tokens, changed, mask, t), ref, atol=1e-3)


def test_each_cross_block_reads_its_own_layer_only():
    dit = _dit(interleave_self_attention=True)
    tokens, contexts, mask, t = _inputs()
    ref = dit(tokens, contexts, mask, t)
    # layer 1 feeds a SELF block: changing it changes nothing
    c = [x.clone() for x in contexts]
    c[1] += 100.0
    assert torch.allclose(dit(tokens, c, mask, t), ref, atol=1e-5)
    # layer 2 feeds a cross block: changing it changes the output
    c = [x.clone() for x in contexts]
    c[2] += 100.0
    assert not torch.allclose(dit(tokens, c, mask, t), ref, atol=1e-3)


def test_adaln_zero_makes_the_stack_the_identity_up_to_the_output_norm():
    dit = _dit(adaln_zero=True)
    tokens, contexts, mask, t = _inputs()
    out = dit(tokens, contexts, mask, t)
    expected = nn.functional.layer_norm(tokens, (H,))
    assert torch.allclose(out, expected, atol=1e-5)


def test_time_changes_the_output():
    dit = _dit()
    tokens, contexts, mask, t = _inputs()
    a = dit(tokens, contexts, mask, torch.zeros(B))
    b = dit(tokens, contexts, mask, torch.ones(B))
    assert not torch.allclose(a, b, atol=1e-3)


def test_block_gradients_flow_to_context():
    block = DiTBlock(H, HEADS, cross_dim=H, dropout=0.0)
    x = torch.randn(B, N, H)
    ctx = torch.randn(B, S, H, requires_grad=True)
    temb = torch.randn(B, H)
    block(x, temb, ctx, torch.ones(B, S, dtype=torch.bool)).sum().backward()
    assert ctx.grad is not None and ctx.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# the head
# ---------------------------------------------------------------------------
from egomimic.models.layerwise_dit import LayerwiseFMHead, StateEncoder  # noqa: E402

W, T, VLM_H = 6, 10, 24
STATE_DIMS = {"human_bimanual": 3, "eva_bimanual": 5}


def _head(**kwargs) -> LayerwiseFMHead:
    torch.manual_seed(0)
    defaults = dict(
        action_width=W,
        action_horizon=T,
        vlm_hidden=VLM_H,
        num_layers=L,
        state_dims=STATE_DIMS,
        dit_hidden=H,
        head_dim=8,
        num_register_tokens=3,
        dropout=0.0,
        num_inference_steps=3,
    )
    defaults.update(kwargs)
    return LayerwiseFMHead(**defaults)


def _head_inputs(seed: int = 2, K: int = 1):
    g = torch.Generator().manual_seed(seed)
    contexts = [torch.randn(B, S, VLM_H, generator=g) for _ in range(L)]
    mask = torch.ones(B, S, dtype=torch.bool)
    actions = torch.randn(B, T, W, generator=g)
    loss_mask = torch.ones(B, T, W)
    state = torch.randn(B, K, STATE_DIMS["human_bimanual"], generator=g)
    return contexts, mask, actions, loss_mask, state


def test_head_loss_is_finite_and_backpropagates():
    head = _head().train()
    contexts, mask, actions, loss_mask, state = _head_inputs()
    loss = head.compute_loss(
        contexts, mask, actions, loss_mask, state, "human_bimanual"
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert head.projectors[0][1].weight.grad is not None
    assert head.state_encoder.mlps["human_bimanual"][0].weight.grad is not None


def test_projectors_are_identity_when_widths_match():
    same = _head(vlm_hidden=H, dit_hidden=H)
    assert all(isinstance(p, nn.Identity) for p in same.projectors)
    assert all(isinstance(p, nn.Sequential) for p in _head().projectors)


def test_loss_is_the_masked_mean_of_the_velocity_error():
    """The mask weights the LOSS only (the noised input still carries every
    dim), so check the reduction against a seeded replica of compute_loss."""
    head = _head().eval()
    contexts, mask, actions, loss_mask, state = _head_inputs()
    loss_mask[..., -2:] = 0.0
    torch.manual_seed(5)
    loss = head.compute_loss(
        contexts, mask, actions, loss_mask, state, "human_bimanual"
    )
    torch.manual_seed(5)  # same draw order: noise first, then t
    noise = torch.randn_like(actions)
    t = head.sample_time(B, actions.device).to(actions.dtype)
    te = t[:, None, None]
    v = head.velocity(
        te * noise + (1 - te) * actions,
        t,
        head.project(contexts),
        mask,
        state,
        "human_bimanual",
    )
    expected = ((v - (noise - actions)) ** 2)[..., :-2].mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_sample_shape_and_determinism_under_a_generator():
    head = _head().eval()
    contexts, mask, _, _, state = _head_inputs()
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    a = head.sample(contexts, mask, state, "human_bimanual", generator=g1)
    b = head.sample(contexts, mask, state, "human_bimanual", generator=g2)
    assert a.shape == (B, T, W)
    assert torch.equal(a, b)
    g3 = torch.Generator().manual_seed(12)
    assert not torch.equal(
        a, head.sample(contexts, mask, state, "human_bimanual", generator=g3)
    )


def test_state_selects_the_embodiment_mlp_and_checks_its_width():
    head = _head().eval()
    contexts, mask, _, _, state = _head_inputs()
    g = torch.Generator().manual_seed(1)
    with_state = head.sample(contexts, mask, state, "human_bimanual", generator=g)
    g = torch.Generator().manual_seed(1)
    without = head.sample(contexts, mask, None, "human_bimanual", generator=g)
    assert not torch.allclose(with_state, without)
    with pytest.raises(ValueError):
        head.sample(contexts, mask, state, "eva_bimanual")  # 3-wide state, 5-wide MLP
    with pytest.raises(KeyError):
        head.sample(contexts, mask, state, "aria_bimanual")


def test_history_time_embed_is_zero_init_and_history_length_is_checked():
    head = _head(history_len=2)
    assert head.state_encoder.time_embed.shape == (2, H)
    assert torch.all(head.state_encoder.time_embed == 0)
    assert _head(history_len=1).state_encoder.time_embed is None
    contexts, mask, actions, loss_mask, state = _head_inputs(K=1)
    with pytest.raises(ValueError):
        head.compute_loss(contexts, mask, actions, loss_mask, state, "human_bimanual")


def test_history_dropout_replaces_the_past_with_the_current_step():
    enc = StateEncoder(STATE_DIMS, H, history_len=3, history_dropout=1.0).train()
    state = torch.randn(B, 3, 3)
    current_only = state[:, -1:].expand_as(state)
    assert torch.allclose(
        enc(state, "human_bimanual"), enc(current_only, "human_bimanual")
    )
    enc.eval()
    assert not torch.allclose(
        enc(state, "human_bimanual"), enc(current_only, "human_bimanual")
    )


def test_repeated_diffusion_steps_repeats_the_batch():
    head = _head(repeated_diffusion_steps=2).train()
    contexts, mask, actions, loss_mask, state = _head_inputs()
    loss = head.compute_loss(
        contexts, mask, actions, loss_mask, state, "human_bimanual"
    )
    assert torch.isfinite(loss)


def test_action_horizon_mismatch_raises():
    head = _head()
    contexts, mask, actions, loss_mask, state = _head_inputs()
    with pytest.raises(ValueError):
        head.compute_loss(
            contexts, mask, actions[:, :-1], loss_mask[:, :-1], state, "human_bimanual"
        )
