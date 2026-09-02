"""UNITE Generative Encoder backbone for action-register tokenization.

The backbone preserves the released UNITE mechanism: Gaussian register values
are the only predicted sequence, clean content tokens are appended only during
tokenization, and every Transformer block plus the final projection is
conditioned with adaptive layer normalization (AdaLN).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sincos_positions(length: int, width: int) -> torch.Tensor:
    """Return a fixed one-dimensional sine/cosine position table."""

    if length <= 0 or width <= 0:
        raise ValueError("position-table dimensions must be positive")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / float(width))
    )
    table = torch.zeros(length, width, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * frequency)
    odd_width = table[:, 1::2].shape[1]
    table[:, 1::2] = torch.cos(position * frequency[:odd_width])
    return table.unsqueeze(0)


class _RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1.0e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(width)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight.to(value.dtype)


class _GaussianFourierEmbedding(nn.Module):
    """Released-style continuous-time Fourier embedding followed by an MLP."""

    def __init__(self, hidden_dim: int, fourier_dim: int):
        super().__init__()
        if fourier_dim <= 0:
            raise ValueError("time_fourier_dim must be positive")
        frequencies = torch.randn(int(fourier_dim), dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * int(fourier_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 0:
            time = time.reshape(1)
        if time.ndim != 1:
            raise ValueError(f"time must have shape (B,), got {tuple(time.shape)}")
        angle = (
            time.float().unsqueeze(-1)
            * self.frequencies.float().unsqueeze(0)
            * (2.0 * math.pi)
        )
        features = torch.cat((angle.sin(), angle.cos()), dim=-1)
        return self.mlp(features)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1)
    return rotated.flatten(-2)


class _SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        qk_norm: bool,
        use_rope: bool,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        if use_rope and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head width")
        self.qk_norm = bool(qk_norm)
        self.use_rope = bool(use_rope)
        self.dropout = float(dropout)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=True)
        self.q_norm = _RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _apply_rope(self, value: torch.Tensor) -> torch.Tensor:
        sequence_length = int(value.shape[-2])
        frequency = 1.0 / (
            10_000.0
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    device=value.device,
                    dtype=torch.float32,
                )
                / float(self.head_dim)
            )
        )
        position = torch.arange(
            sequence_length, device=value.device, dtype=torch.float32
        )
        angle = torch.einsum("i,j->ij", position, frequency)
        angle = torch.repeat_interleave(angle, 2, dim=-1)
        cosine = angle.cos().to(value.dtype).reshape(1, 1, sequence_length, -1)
        sine = angle.sin().to(value.dtype).reshape(1, 1, sequence_length, -1)
        return value * cosine + _rotate_half(value) * sine

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        qkv = self.qkv(value).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, content = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if self.use_rope:
            query = self._apply_rope(query)
            key = self._apply_rope(key)
        attended = F.scaled_dot_product_attention(
            query.to(content.dtype),
            key.to(content.dtype),
            content,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_dim
        )
        return self.projection(attended)


class _SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, expansion_dim: int):
        super().__init__()
        self.gate_value = nn.Linear(hidden_dim, 2 * expansion_dim)
        self.output = nn.Linear(expansion_dim, hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.gate_value(value).chunk(2, dim=-1)
        return self.output(F.silu(gate) * content)


def _modulate(
    value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _AdaLNBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        qk_norm: bool,
        use_rope: bool,
        use_rmsnorm: bool,
        use_swiglu: bool,
        block_norm: bool,
        dropout: float,
    ):
        super().__init__()
        norm = (
            _RMSNorm
            if use_rmsnorm
            else lambda width: nn.LayerNorm(width, elementwise_affine=False, eps=1.0e-6)
        )
        self.norm1 = norm(hidden_dim)
        self.norm2 = norm(hidden_dim)
        self.norm3 = norm(hidden_dim) if block_norm else nn.Identity()
        self.block_norm = bool(block_norm)
        self.attention = _SelfAttention(
            hidden_dim,
            num_heads,
            qk_norm=qk_norm,
            use_rope=use_rope,
            dropout=dropout,
        )
        expanded = int(hidden_dim * float(mlp_ratio))
        if expanded <= 0:
            raise ValueError("mlp_ratio produces an empty feed-forward layer")
        self.mlp = (
            _SwiGLU(hidden_dim, int((2.0 / 3.0) * expanded))
            if use_swiglu
            else nn.Sequential(
                nn.Linear(hidden_dim, expanded),
                nn.GELU(approximate="tanh"),
                nn.Linear(expanded, hidden_dim),
            )
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )

    def forward(self, value: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_modulation(conditioning).chunk(6, dim=-1)
        value = value + gate_attention.unsqueeze(1) * self.attention(
            _modulate(self.norm1(value), shift_attention, scale_attention)
        )
        value = value + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(value), shift_mlp, scale_mlp)
        )
        return self.norm3(value) if self.block_norm else value


class _AdaLNFinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, use_rmsnorm: bool):
        super().__init__()
        self.norm = (
            _RMSNorm(hidden_dim)
            if use_rmsnorm
            else nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1.0e-6)
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, value: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_modulation(conditioning).chunk(2, dim=-1)
        return _modulate(self.norm(value), shift, scale)


class UniteDiTBackbone(nn.Module):
    """AdaLN Generative Encoder over latent registers.

    ``value`` always contains exactly ``horizon`` register slots. During the
    tokenizer pass, projected clean-action content is appended to those slots
    with ``content_tokens``. During denoising, observations are supplied via
    ``condition`` and summarized into AdaLN conditioning; they are never used
    as tokenizer content. Only the register-prefix outputs are returned.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        condition_dim: int,
        max_condition_tokens: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        block_norm: bool = True,
        time_fourier_dim: int = 256,
        context_adaln_summary: bool = False,
        max_content_tokens: int | None = None,
        trainable_position_embeddings: bool = False,
        trainable_register_position_embeddings: bool = True,
        trainable_content_position_embeddings: bool = False,
        trainable_condition_position_embeddings: bool = False,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.condition_dim = int(condition_dim)
        self.max_condition_tokens = int(max_condition_tokens)
        self.max_content_tokens = (
            self.horizon if max_content_tokens is None else int(max_content_tokens)
        )
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.context_adaln_summary = bool(context_adaln_summary)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.time_conditioning = "additive"
        if (
            min(
                self.input_dim,
                self.output_dim,
                self.horizon,
                self.condition_dim,
                self.max_condition_tokens,
                self.max_content_tokens,
                self.hidden_dim,
                self.depth,
                self.num_heads,
            )
            <= 0
        ):
            raise ValueError("UNITE backbone dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        # Keep the released attribute names used by existing contract audits.
        self.proj_u = nn.Linear(self.input_dim, self.hidden_dim)
        self.proj_d = nn.Linear(self.hidden_dim, self.output_dim)
        self.content_projection = nn.Linear(self.condition_dim, self.hidden_dim)
        self.condition_projection = nn.Linear(self.condition_dim, self.hidden_dim)
        self.time_embedder = _GaussianFourierEmbedding(
            self.hidden_dim, int(time_fourier_dim)
        )
        register_positions = _sincos_positions(self.horizon, self.hidden_dim)
        content_positions = _sincos_positions(self.max_content_tokens, self.hidden_dim)
        condition_positions = _sincos_positions(
            self.max_condition_tokens, self.hidden_dim
        )
        register_trainable = bool(
            trainable_position_embeddings or trainable_register_position_embeddings
        )
        condition_trainable = bool(
            trainable_position_embeddings or trainable_condition_position_embeddings
        )
        content_trainable = bool(
            trainable_position_embeddings or trainable_content_position_embeddings
        )
        if register_trainable:
            self.pos_emb = nn.Parameter(register_positions)
        else:
            self.register_buffer("pos_emb", register_positions, persistent=True)
        if content_trainable:
            self.content_pos_emb = nn.Parameter(content_positions)
        else:
            self.register_buffer("content_pos_emb", content_positions, persistent=True)
        if condition_trainable:
            self.condition_pos_emb = nn.Parameter(condition_positions)
        else:
            self.register_buffer(
                "condition_pos_emb", condition_positions, persistent=True
            )
        if self.context_adaln_summary:
            self.context_summary_gate = nn.Linear(self.hidden_dim, 1)
        else:
            self.context_summary_gate = None
        self.blocks = nn.ModuleList(
            _AdaLNBlock(
                self.hidden_dim,
                self.num_heads,
                float(mlp_ratio),
                qk_norm=bool(qk_norm),
                use_rope=bool(use_rope),
                use_rmsnorm=bool(use_rmsnorm),
                use_swiglu=bool(use_swiglu),
                block_norm=bool(block_norm),
                dropout=float(dropout),
            )
            for _ in range(self.depth)
        )
        self.final_layer = _AdaLNFinalLayer(self.hidden_dim, bool(use_rmsnorm))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.proj_d.weight)
        nn.init.zeros_(self.proj_d.bias)

    def _validate_registers(self, value: torch.Tensor) -> None:
        expected = (self.horizon, self.input_dim)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "UNITE register input must have shape "
                f"(B, {self.horizon}, {self.input_dim}), got {tuple(value.shape)}"
            )

    def _project_context(
        self,
        tokens: torch.Tensor,
        projection: nn.Module,
        position_embedding: torch.Tensor,
        max_tokens: int,
        context_name: str,
    ) -> torch.Tensor:
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.ndim != 3 or int(tokens.shape[-1]) != self.condition_dim:
            raise ValueError(
                f"UNITE {context_name} must have shape "
                f"(B, C, {self.condition_dim}), got {tuple(tokens.shape)}"
            )
        token_count = int(tokens.shape[1])
        if not 0 < token_count <= max_tokens:
            raise ValueError(
                f"UNITE {context_name} token count {token_count} is outside "
                f"[1, {max_tokens}]"
            )
        projected = projection(tokens)
        positions = position_embedding[:, :token_count].to(projected)
        return projected + positions

    def _condition_summary(
        self, time: torch.Tensor, condition: torch.Tensor | None
    ) -> torch.Tensor:
        summary = self.time_embedder(time)
        if condition is None:
            return summary
        projected = self._project_context(
            condition,
            self.condition_projection,
            self.condition_pos_emb,
            self.max_condition_tokens,
            "observation condition",
        )
        if self.context_summary_gate is None:
            context_summary = projected.mean(dim=1)
        else:
            weights = self.context_summary_gate(projected).softmax(dim=1)
            context_summary = (weights * projected).sum(dim=1)
        return summary + context_summary

    def shared_core_named_parameters(self):
        """Parameters traversed by both tokenization and denoising paths."""

        excluded = (
            "content_projection.",
            "condition_projection.",
            "content_pos_emb",
            "condition_pos_emb",
            "context_summary_gate.",
        )
        return tuple(
            (name, parameter)
            for name, parameter in self.named_parameters()
            if not name.startswith(excluded)
        )

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        content_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_registers(value)
        batch_size = int(value.shape[0])
        if time.ndim == 0:
            time = time.expand(batch_size)
        if time.ndim != 1 or int(time.shape[0]) != batch_size:
            raise ValueError(
                f"time must have shape ({batch_size},), got {tuple(time.shape)}"
            )
        if condition is not None and content_tokens is not None:
            raise ValueError(
                "observation conditioning and clean content tokens are exclusive"
            )
        hidden = self.proj_u(value) + self.pos_emb.to(value)
        if content_tokens is not None:
            content = self._project_context(
                content_tokens,
                self.content_projection,
                self.content_pos_emb,
                self.max_content_tokens,
                "clean-action content",
            )
            if int(content.shape[0]) != batch_size:
                raise ValueError("content-token batch does not match registers")
            hidden = torch.cat((hidden, content), dim=1)
        if condition is not None and int(condition.shape[0]) != batch_size:
            raise ValueError("condition batch does not match registers")
        conditioning = self._condition_summary(time, condition).to(hidden.dtype)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(block, hidden, conditioning, use_reentrant=False)
            else:
                hidden = block(hidden, conditioning)
        registers = hidden[:, : self.horizon]
        # proj_d is deliberately kept as the public released-style output layer.
        return self.proj_d(self.final_layer(registers, conditioning))
