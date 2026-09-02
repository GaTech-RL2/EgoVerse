"""Fixed-horizon action decoder for compact UNITE register latents."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _sincos_positions(length: int, width: int) -> torch.Tensor:
    position = torch.arange(int(length), dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, int(width), 2, dtype=torch.float32)
        * (-math.log(10_000.0) / float(width))
    )
    table = torch.zeros(int(length), int(width), dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * frequency)
    odd_width = table[:, 1::2].shape[1]
    table[:, 1::2] = torch.cos(position * frequency[:odd_width])
    return table.unsqueeze(0)


class UniteActionDecoder(nn.Module):
    """Decode ``N`` compact registers into a fixed ``H`` clean action chunk.

    Latent registers and fixed action-query tokens are concatenated and updated
    by ordinary non-causal Transformer blocks. Returning only the action-query
    suffix mirrors a ViT/MAE decoder while allowing ``N`` and ``H`` to differ.
    The one-dimensional action position table is fixed, as in the released
    decoder's fixed positional geometry.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_latent_tokens: int,
        action_horizon: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.num_latent_tokens = int(num_latent_tokens)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if (
            min(
                self.latent_dim,
                self.action_dim,
                self.num_latent_tokens,
                self.action_horizon,
                self.hidden_dim,
                self.depth,
                self.num_heads,
            )
            <= 0
        ):
            raise ValueError("UNITE action-decoder dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.action_horizon % self.num_latent_tokens:
            raise ValueError(
                "action_horizon must be divisible by num_latent_tokens for the "
                "Pipeline temporal-factor contract"
            )
        self.temporal_factor = self.action_horizon // self.num_latent_tokens

        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.action_query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.token_identity = nn.Parameter(torch.zeros(2, self.hidden_dim))
        nn.init.normal_(self.action_query, std=0.02)
        nn.init.normal_(self.token_identity, std=0.02)
        self.register_buffer(
            "latent_pos_embed",
            _sincos_positions(self.num_latent_tokens, self.hidden_dim),
            persistent=True,
        )
        self.register_buffer(
            "decoder_pos_embed",
            _sincos_positions(self.action_horizon, self.hidden_dim),
            persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.hidden_dim * self.mlp_ratio),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer,
            num_layers=self.depth,
            norm=nn.LayerNorm(self.hidden_dim),
            enable_nested_tensor=False,
        )
        self.action_projection = nn.Linear(self.hidden_dim, self.action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.latent_projection.weight)
        nn.init.zeros_(self.latent_projection.bias)
        nn.init.xavier_uniform_(self.action_projection.weight)
        nn.init.zeros_(self.action_projection.bias)

    def output_num_tokens(self, input_num_tokens: int) -> int:
        input_num_tokens = int(input_num_tokens)
        if input_num_tokens != self.num_latent_tokens:
            raise ValueError(
                f"UNITE decoder received {input_num_tokens} register tokens, "
                f"expected {self.num_latent_tokens}"
            )
        return self.action_horizon

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.num_latent_tokens, self.latent_dim)
        if latent.ndim != 3 or tuple(latent.shape[1:]) != expected:
            raise ValueError(
                "UniteActionDecoder expected latent shape "
                f"(B, {self.num_latent_tokens}, {self.latent_dim}), got "
                f"{tuple(latent.shape)}"
            )
        batch_size = int(latent.shape[0])
        memory = self.latent_projection(latent)
        memory = (
            memory
            + self.latent_pos_embed.to(memory)
            + self.token_identity[0].to(memory).reshape(1, 1, -1)
        )
        actions = (
            self.action_query.to(memory)
            + self.decoder_pos_embed.to(memory)
            + self.token_identity[1].to(memory).reshape(1, 1, -1)
        ).expand(batch_size, -1, -1)
        hidden = torch.cat((memory, actions), dim=1)
        for layer in self.decoder.layers:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        decoded_actions = hidden[:, self.num_latent_tokens :]
        output = self.action_projection(decoded_actions)
        output_shape = (batch_size, self.action_horizon, self.action_dim)
        if tuple(output.shape) != output_shape:
            raise RuntimeError(
                f"UNITE action decoder produced {tuple(output.shape)}, "
                f"expected {output_shape}"
            )
        return output
