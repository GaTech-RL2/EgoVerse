"""HPT policy HEAD modules relocated from ``models/hpt_nets.py``.

Role home for the HPT output-side readouts: the :class:`PolicyHead` base, the
:class:`MLPPolicyHead`, and the learnable-query
:class:`MultiBlockTransformerDecoder` (built from
:class:`TransformerDecoderBlock`). Moved here verbatim (class bodies
byte-identical) in the models/ hierarchy pass; the ``Attention`` /
``CrossAttention`` primitives now import from
``egomimic.models.cores.hpt_transformer``.
"""

from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.cores.hpt_transformer import Attention, CrossAttention
from egomimic.utils.tensor_utils import get_sinusoid_encoding_table

INIT_CONST = 0.02

LOSS = partial(F.smooth_l1_loss, beta=0.05)
LOSS_MSE = partial(F.mse_loss)

class PolicyHead(nn.Module):
    """Abstract class for policy head."""

    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_loss(self, x: torch.Tensor, data: dict):
        """
        Compute smooth L1 loss between predicted and target actions,
        slicing as needed if their dimensions differ.

        Args:
            x (torch.Tensor): Transformer outputs used to predict actions.
            data (dict): Contains:
                - 'action': ground-truth action tensor of shape (B, T, D_target)

        Returns:
            torch.Tensor: Scalar loss
        """
        target_action = data["action"]
        B, T = target_action.shape[:2]

        pred_action = self(x).view(B, T, -1)

        D_pred = pred_action.shape[-1]
        D_target = target_action.shape[-1]

        D_common = min(D_pred, D_target)
        pred_action = pred_action[..., :D_common]
        target_action = target_action[..., :D_common]

        return LOSS(pred_action, target_action)


class MLPPolicyHead(PolicyHead):
    """Simple MLP based policy head"""

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        dropout: bool = False,
        tanh_end: bool = False,
        ln: bool = True,
        **kwargs,
    ) -> None:
        """vanilla MLP head on the pooled feature"""
        super().__init__()
        self.input = input
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if dropout:
                modules.append(nn.Dropout(p=0.1))
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass of the policy head module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        y = self.net(x)
        return y


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention = Attention(
            dim=input_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        self.cross_attention = CrossAttention(
            input_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.SiLU(), nn.Linear(input_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, tokens, context):
        query = self.self_attention(self.norm1(tokens))
        query = tokens + query

        out = self.cross_attention(self.norm2(query), context)
        out = query + out

        mlp_out = self.mlp(self.norm3(out))
        tokens = mlp_out + out
        return tokens


class MultiBlockTransformerDecoder(PolicyHead):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 10,
        action_horizon: int = 16,
        latent_token_len: int = 8,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        num_layers: int = 4,
        final_norm: bool = False,
    ):
        super().__init__()
        self.tokens = nn.Parameter(
            torch.randn(1, action_horizon, input_dim) * INIT_CONST
        )
        self.pos_token = nn.Parameter(
            get_sinusoid_encoding_table(0, action_horizon, input_dim)
        )
        self.pos_context = nn.Parameter(
            get_sinusoid_encoding_table(0, latent_token_len, input_dim)
        )

        self.context_norm = nn.LayerNorm(input_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = final_norm
        if self.final_norm:
            self.last_layer_norm = nn.LayerNorm(input_dim)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[MultiBlockTransformerDecoder] Total trainable parameters: {total_params / 1e6:.2f}M"
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = self.tokens.expand(B, -1, -1) + self.pos_token.expand(B, -1, -1)
        context = self.context_norm(x + self.pos_context.expand(B, -1, -1))

        for block in self.blocks:
            tokens = block(tokens, context)

        if self.final_norm:
            tokens = self.last_layer_norm(tokens)

        return self.out_proj(tokens)
