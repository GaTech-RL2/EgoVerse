from __future__ import annotations

import torch
import torch.nn as nn


class SmallTemporalEncoder(nn.Module):
    """
    Fix temporal encoder for 100 seq of actiona
    """

    def __init__(
        self,
        *,
        action_dim: int,
        activation: str = "gelu",
        hidden_dim: int = 64,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = [
            nn.Conv1d(action_dim, action_dim * 2, kernel_size=8, stride=2, padding=3),
            self.act,
            nn.Conv1d(
                action_dim * 2, action_dim * 2, kernel_size=8, stride=2, padding=2
            ),
            self.act,
            nn.Conv1d(
                action_dim * 2, action_dim * 2, kernel_size=8, stride=2, padding=3
            ),
            self.act,
        ]

        self.down = nn.Sequential(*layers)
        self.proj = nn.Linear(action_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, T, D) or (T, D)
        Output: (B, K, H) or (K, H)
        """
        squeeze_B = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_B = True
        elif x.dim() != 3:
            raise ValueError(f"Expected (T,D) or (B,T,D), got {tuple(x.shape)}")

        x = x.transpose(1, 2)  # (B, D, T)
        x = self.down(x)  # (B, D, K)
        x = x.transpose(1, 2)  # (B, K, D)
        x = self.proj(x)  # (B, K, H)

        return x.squeeze(0) if squeeze_B else x


class SmallTemporalDecoder(nn.Module):
    """
    Decoder that mirrors SmallTemporalEncoder:
        Enc convs (over time, channels-first):
            (D -> 2D) k=8 s=2 p=3
            (2D -> 2D) k=8 s=2 p=2
            (2D -> 2D) k=8 s=2 p=3
        For T=100 this encoder produces K=12.

    This decoder maps:
        Input:  (B, K=12, H=64) or (K, H)
        Output: (B, T=100, D)   or (T, D)
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 64,
        activation: str = "gelu",
        use_layernorm: bool = True,
        K: int = 12,
        T: int = 100,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.T = T

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        C2 = action_dim * 2

        self.proj = nn.Linear(hidden_dim, C2)
        self.norm = nn.LayerNorm(C2) if use_layernorm else nn.Identity()

        self.up = nn.Sequential(
            nn.ConvTranspose1d(
                C2, C2, kernel_size=8, stride=2, padding=3, output_padding=0
            ),
            self.act,
            nn.ConvTranspose1d(
                C2, C2, kernel_size=8, stride=2, padding=2, output_padding=0
            ),
            self.act,
            nn.ConvTranspose1d(
                C2, action_dim, kernel_size=8, stride=2, padding=3, output_padding=0
            ),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_B = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze_B = True
        elif z.dim() != 3:
            raise ValueError(f"Expected (K,H) or (B,K,H), got {tuple(z.shape)}")

        B, K, H = z.shape
        if H != self.hidden_dim:
            raise ValueError(f"Expected H={self.hidden_dim}, got {H}")

        x = self.norm(self.proj(z))  # (B, K, 2D)
        x = x.transpose(1, 2)  # (B, 2D, K)
        x = self.up(x)  # (B, D, T)
        if x.shape[-1] != self.T:
            raise ValueError(f"Got T_out={x.shape[-1]}, expected T={self.T}")
        x = x.transpose(1, 2)  # (B, T, D)

        return x.squeeze(0) if squeeze_B else x


class LargeTemporalEncoder(nn.Module):
    """
    Encoder for (B, T=100, D) that halves channels: D -> D/2,
    and downsamples time: 100 -> 12.
    Output: (B, K=12, H)
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 64,
        activation: str = "gelu",
        use_layernorm: bool = True,
        expect_T: int | None = 100,
    ):
        super().__init__()
        if action_dim % 2 != 0:
            raise ValueError(f"action_dim must be even to halve. Got {action_dim}")

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.expect_T = expect_T

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        D = action_dim

        self.down = nn.Sequential(
            nn.Conv1d(D, action_dim, kernel_size=8, stride=2, padding=3),  # 100 -> 50
            self.act,
            nn.Conv1d(
                action_dim, action_dim, kernel_size=8, stride=2, padding=2
            ),  # 50 -> 24
            self.act,
            nn.Conv1d(
                action_dim, action_dim, kernel_size=8, stride=2, padding=3
            ),  # 24 -> 12
            self.act,
        )

        self.proj = nn.Linear(action_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_B = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_B = True
        elif x.dim() != 3:
            raise ValueError(f"Expected (T,D) or (B,T,D), got {tuple(x.shape)}")

        B, T, D = x.shape
        if D != self.action_dim:
            raise ValueError(f"Expected D={self.action_dim}, got {D}")
        if self.expect_T is not None and T != self.expect_T:
            raise ValueError(f"Expected T={self.expect_T}, got {T}")

        x = x.transpose(1, 2)  # (B, D, T)
        x = self.down(x)  # (B, D/2, K=12)
        x = x.transpose(1, 2)  # (B, K, D/2)
        x = self.proj(x)  # (B, K, H)
        return x.squeeze(0) if squeeze_B else x


class LargeTemporalDecoder(nn.Module):
    """
    Decoder that mirrors LargeTemporalEncoder:
        time: 12 -> 24 -> 50 -> 100
        channels: H -> D/2 -> D
    Input:  (B, K=12, H) or (K, H)
    Output: (B, T=100, D) or (T, D)
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 64,
        activation: str = "gelu",
        use_layernorm: bool = True,
        T: int = 100,
    ):
        super().__init__()
        if action_dim % 2 != 0:
            raise ValueError(f"action_dim must be even to halve. Got {action_dim}")

        self.action_dim = action_dim
        self.half_dim = action_dim // 2
        self.hidden_dim = hidden_dim
        self.T = T

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.proj = nn.Linear(hidden_dim, action_dim)
        self.norm = nn.LayerNorm(action_dim) if use_layernorm else nn.Identity()

        # Mirrors paddings/strides/kernels in reverse.
        # Lengths: 12 -> 24 -> 50 -> 100 with output_padding=0 for these params.
        self.up = nn.Sequential(
            nn.ConvTranspose1d(
                action_dim,
                action_dim,
                kernel_size=8,
                stride=2,
                padding=3,
                output_padding=0,
            ),
            self.act,
            nn.ConvTranspose1d(
                action_dim,
                action_dim,
                kernel_size=8,
                stride=2,
                padding=2,
                output_padding=0,
            ),
            self.act,
            nn.ConvTranspose1d(
                action_dim,
                action_dim,
                kernel_size=8,
                stride=2,
                padding=3,
                output_padding=0,
            ),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_B = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze_B = True
        elif z.dim() != 3:
            raise ValueError(f"Expected (K,H) or (B,K,H), got {tuple(z.shape)}")

        B, K, H = z.shape
        if H != self.hidden_dim:
            raise ValueError(f"Expected H={self.hidden_dim}, got {H}")

        x = self.norm(self.proj(z))  # (B, K, D/2)
        x = x.transpose(1, 2)  # (B, D/2, K)
        x = self.up(x)  # (B, D, T)
        if x.shape[-1] != self.T:
            raise ValueError(f"Got T_out={x.shape[-1]}, expected T={self.T}")
        x = x.transpose(1, 2)  # (B, T, D)
        return x.squeeze(0) if squeeze_B else x


class SmallTemporalEncoder_32_256(SmallTemporalEncoder):
    """
    Fix temporal encoder for 100 seq of actiona
    """

    def __init__(
        self,
        *,
        action_dim: int,
        activation: str = "gelu",
        hidden_dim: int = 256,
        use_layernorm: bool = True,
        n_layers: int = 4,
    ):
        super().__init__(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            use_layernorm=use_layernorm,
        )

        layers = [
            nn.Conv1d(action_dim, 512, kernel_size=9, stride=3, padding=2),
            self.act,
        ]

        self.down = nn.Sequential(*layers)
        self.proj = nn.Linear(512, hidden_dim)
        proj_layers = [
            nn.Linear(512, hidden_dim),
            self.act,
        ]
        for _ in range(n_layers - 1):
            proj_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    self.act,
                ]
            )
        proj_layers.extend(
            [nn.Linear(hidden_dim, hidden_dim)]
        )  # TODO check if I need activation here later
        self.proj = nn.Sequential(*proj_layers)


class SmallTemporalDecoder_32_256(SmallTemporalDecoder):
    """
    Decoder that mirrors SmallTemporalEncoder_32_128:
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 256,
        activation: str = "gelu",
        use_layernorm: bool = True,
        n_layers: int = 4,
    ):
        super().__init__(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            use_layernorm=use_layernorm,
        )

        layers = [
            nn.ConvTranspose1d(512, action_dim, kernel_size=9, stride=3, padding=1),
        ]

        self.up = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(512) if use_layernorm else nn.Identity()
        proj_layers = []
        for _ in range(n_layers - 1):
            proj_layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act])
        proj_layers.extend([nn.Linear(hidden_dim, 512)])
        self.proj = nn.Sequential(*proj_layers)


class LargeTemporalEncoder_32_256(LargeTemporalEncoder):
    """
    Encoder for (B, T=100, D) that halves channels: D -> D/2,
    and downsamples time: 100 -> 12.
    Output: (B, K=12, H)
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 256,
        activation: str = "gelu",
        use_layernorm: bool = True,
        expect_T: int | None = 100,
        n_layers: int = 4,
    ):
        super().__init__(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            expect_T=expect_T,
        )
        layers = [
            nn.Conv1d(action_dim, 2048, kernel_size=9, stride=3, padding=2),
            self.act,
        ]

        self.down = nn.Sequential(*layers)
        proj_layers = [
            nn.Linear(2048, 1024),
            self.act,
            nn.Linear(1024, hidden_dim),
            self.act,
        ]
        for _ in range(n_layers - 1):
            proj_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    self.act,
                ]
            )
        proj_layers.extend(
            [nn.Linear(hidden_dim, hidden_dim)]
        )  # TODO check if I need activation here later
        self.proj = nn.Sequential(*proj_layers)


class LargeTemporalDecoder_32_256(LargeTemporalDecoder):
    """
    Decoder that mirrors LargeTemporalEncoder_32_128:
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 256,
        activation: str = "gelu",
        use_layernorm: bool = True,
        T: int = 100,
        n_layers: int = 4,
    ):
        super().__init__(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            T=T,
        )

        layers = [
            nn.ConvTranspose1d(2048, action_dim, kernel_size=9, stride=3, padding=1),
        ]

        self.up = nn.Sequential(*layers)
        proj_layers = []
        for _ in range(n_layers - 1):
            proj_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    self.act,
                ]
            )
        proj_layers.extend([nn.Linear(hidden_dim, 2048)])
        self.proj = nn.Sequential(*proj_layers)
        self.norm = nn.LayerNorm(2048) if use_layernorm else nn.Identity()


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def print_param_breakdown(module: nn.Module, trainable_only: bool = False) -> None:
    total = 0
    for name, p in module.named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        n = p.numel()
        total += n
        print(f"{name:60s} {tuple(p.shape)!s:20s} {n}")
    print(f"\nTOTAL params: {total}")


if __name__ == "__main__":
    B, T, D = 8, 100, 140

    enc = LargeTemporalEncoder_32_256(action_dim=D, n_layers=5)
    dec = LargeTemporalDecoder_32_256(action_dim=D, use_layernorm=True, n_layers=5)

    x = torch.randn(B, T, D)
    z = enc(x)
    x_hat = dec(z)
    # z shape is 8, 12, 64
    print("LargeTemporalEncoder_32_256")
    print(count_params(enc))
    print(count_params(enc, trainable_only=True))
    print_param_breakdown(enc)

    B, T, D = 8, 100, 14
    enc = SmallTemporalEncoder_32_256(action_dim=D, n_layers=5)
    dec = SmallTemporalDecoder_32_256(action_dim=D, use_layernorm=True, n_layers=5)

    x = torch.randn(B, T, D)
    z = enc(x)
    x_hat = dec(z)
    # z shape is 8, 12, 64

    print("SmallTemporalEncoder_32_256")
    print(count_params(enc))
    print(count_params(enc, trainable_only=True))
    print_param_breakdown(enc)
