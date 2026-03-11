import torch
import torch.nn as nn


class MLPProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, output_dim: int):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in (B, T, D) -> (B, T, H)
        return self.net(x)
