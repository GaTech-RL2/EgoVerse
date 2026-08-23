"""Shared-residual sparse-MoE variant of the Pipeline cross denoiser."""

import torch
from torch import nn

from egomimic.models.denoising_nets import CrossBlock, CrossTransformer
from egomimic.models.moe_ffn import MoEFFN


class DDPSafeMoEFFN(MoEFFN):
    """Keep every expert parameter in the DDP graph on sparse-routing steps."""

    def forward(self, x):
        output = super().forward(x)
        anchor = sum(
            parameter.reshape(-1)[0] * 0.0
            for expert in self.experts
            for parameter in expert.parameters()
        )
        return output + anchor.to(output)


class SharedResidualMoEFFN(nn.Module):
    """Retain the dense shared FFN and add a scaled expert residual."""

    def __init__(
        self,
        shared_ffn,
        d_model,
        moe_experts=8,
        moe_top_k=4,
        moe_d_expert=256,
        moe_aux_weight=1.0e-3,
        expert_scale_init=0.1,
    ):
        super().__init__()
        self.shared_ffn = shared_ffn
        self.expert_ffn = DDPSafeMoEFFN(
            d_model=int(d_model),
            d_intermediate=int(moe_d_expert),
            num_experts=int(moe_experts),
            top_k=int(moe_top_k),
            aux_weight=float(moe_aux_weight),
        )
        self.expert_scale = nn.Parameter(torch.tensor(float(expert_scale_init)))

    def forward(self, x):
        return self.shared_ffn(x) + self.expert_scale.to(x) * self.expert_ffn(x)


class MoECrossBlock(CrossBlock):
    """CrossBlock whose shared FFN is augmented by routed experts."""

    def __init__(
        self,
        cond_dim,
        hidden_dim,
        n_heads,
        dropout,
        mlp_layers,
        mlp_ratio,
        moe_experts=8,
        moe_top_k=4,
        moe_d_expert=256,
        moe_aux_weight=1.0e-3,
        expert_scale_init=0.1,
        **kwargs,
    ):
        super().__init__(
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            mlp_layers=mlp_layers,
            mlp_ratio=mlp_ratio,
            **kwargs,
        )
        self.mlp = SharedResidualMoEFFN(
            shared_ffn=self.mlp,
            d_model=hidden_dim,
            moe_experts=moe_experts,
            moe_top_k=moe_top_k,
            moe_d_expert=moe_d_expert,
            moe_aux_weight=moe_aux_weight,
            expert_scale_init=expert_scale_init,
        )


class MoECrossTransformer(CrossTransformer):
    """CrossTransformer with shared-residual sparse MoE in every block."""

    def __init__(
        self,
        nblocks,
        cond_dim,
        hidden_dim,
        act_dim,
        act_seq,
        n_heads,
        dropout,
        mlp_layers,
        mlp_ratio,
        time_conditioning="concat",
        moe_experts=8,
        moe_top_k=4,
        moe_d_expert=256,
        moe_aux_weight=1.0e-3,
        expert_scale_init=0.1,
        **kwargs,
    ):
        super().__init__(
            nblocks=nblocks,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            act_seq=act_seq,
            n_heads=n_heads,
            dropout=dropout,
            mlp_layers=mlp_layers,
            mlp_ratio=mlp_ratio,
            time_conditioning=time_conditioning,
            **kwargs,
        )
        self.layers = nn.ModuleList(
            MoECrossBlock(
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                mlp_layers=mlp_layers,
                mlp_ratio=mlp_ratio,
                moe_experts=moe_experts,
                moe_top_k=moe_top_k,
                moe_d_expert=moe_d_expert,
                moe_aux_weight=moe_aux_weight,
                expert_scale_init=expert_scale_init,
                **kwargs,
            )
            for _ in range(int(nblocks))
        )
