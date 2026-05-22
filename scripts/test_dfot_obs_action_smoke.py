"""Smoke test ObsActionDFoTOuterStage:
  * padded forward (B, T, bundle) returns pred_v of bundle width, grads flow
  * packed forward (T_total, bundle) + cu_seqlens behaves the same
  * action_slice correctly indexes the action portion of the bundle
"""

import torch

from egomimic.algo.dfot.backbone import DFoTBackbone
from egomimic.algo.dfot.continuous_diffusion import ContinuousDiffusion
from egomimic.algo.dfot.obs_action_outer_stage import ObsActionDFoTOuterStage
from egomimic.algo.dfot.outer_stage import make_dfot_ctx
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.models.hnet_nets.image_encoders import SimpleConv


def _make_stage(obs_dim=5, action_dim=2, d_model=64):
    bundle = obs_dim + action_dim
    cond = CondEncoderModule(
        d_cond=d_model,
        img_encoders={
            "front_img_1": SimpleConv(
                in_channels=3, channels=[8, 16], kernel_size=3, stride=2,
                embed_dim=32, norm_groups=8,
            ),
        },
        cond_proj_widths=[d_model],
    )
    backbone = DFoTBackbone(
        action_dim=bundle,
        action_horizon=8,
        d_model=d_model,
        d_cond=d_model,
        cond_dim=d_model,
        time_embed_dim=d_model,
        cond_emb_dim=d_model,
        use_fourier_time=True,
        stochastic_time_p=0.0,
        cond_dropout_prob=0.0,
        arch_layout="T2",
        num_heads=4,
        d_intermediate=128,
        rotary_emb_dim=16,
        dropout=0.0,
        resid_dropout=0.0,
        causal=True,
    )
    diff = ContinuousDiffusion(
        action_dim=bundle,
        precond_scale=0.25,
        sigmoid_bias=-1.0,
        clip_noise=20.0,
        logsnr_min=-15.0,
        logsnr_max=15.0,
        shift=1.0,
    )
    return ObsActionDFoTOuterStage(
        action_dim=action_dim,
        cond_encoder=cond,
        backbone=backbone,
        diffusion=diff,
        bundle_obs_keys=["state_agent_obj"],
        bundle_obs_dims=[obs_dim],
    )


def test_padded():
    torch.manual_seed(0)
    stage = _make_stage()
    B, T = 2, 6
    batch = {
        "actions": torch.randn(B, T, 2),
        "__obs": {  # not used for bundle, just placeholder
            "state_agent_obj": torch.randn(B, T, 5),
            "front_img_1": torch.rand(B, T, 3, 16, 16),
        },
    }
    ctx = make_dfot_ctx(is_packed=False, action_key="actions", obs=batch["__obs"])
    v_pred = stage(batch, ctx)
    assert v_pred.shape == (B, T, 7), v_pred.shape
    assert batch["pred_v"].shape == (B, T, 7)
    loss = stage.diffusion.compute_loss(v_pred, ctx.q_state).mean()
    loss.backward()
    missing = [n for n, p in stage.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad: {missing[:5]}"
    print(f"[OK] padded: v_pred={tuple(v_pred.shape)} loss={loss.item():.4f}")


def test_packed():
    torch.manual_seed(1)
    stage = _make_stage()
    seq_lens = [4, 6, 3]
    T_total = sum(seq_lens)
    cu = torch.tensor([0, 4, 10, 13], dtype=torch.long)
    batch = {
        "actions": torch.randn(T_total, 2),
        "__obs": {
            "state_agent_obj": torch.randn(T_total, 5),
            "front_img_1": torch.rand(T_total, 3, 16, 16),
        },
    }
    ctx = make_dfot_ctx(
        is_packed=True, action_key="actions", obs=batch["__obs"],
        cu_seqlens=cu, max_seqlen=int(max(seq_lens)),
    )
    v_pred = stage(batch, ctx)
    assert v_pred.shape == (T_total, 7), v_pred.shape
    loss = stage.diffusion.compute_loss(v_pred, ctx.q_state).mean()
    loss.backward()
    missing = [n for n, p in stage.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad: {missing[:5]}"
    print(f"[OK] packed: v_pred={tuple(v_pred.shape)} loss={loss.item():.4f}")


def test_action_slice():
    stage = _make_stage(obs_dim=5, action_dim=2)
    assert stage.bundle_dim == 7
    assert stage.action_slice == slice(5, 7)
    fake = torch.arange(1 * 3 * 7).reshape(1, 3, 7).float()
    sliced = fake[..., stage.action_slice]
    # last two columns of each row
    expected = fake[..., 5:7]
    assert torch.equal(sliced, expected)
    print(f"[OK] action_slice = {stage.action_slice} indexes the trailing action dims")


if __name__ == "__main__":
    test_padded()
    test_packed()
    test_action_slice()
    print("\nAll smoke tests passed.")
