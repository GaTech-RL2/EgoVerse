"""Stand-ins for ABC-DiT's two pretrained encoders, so tests load neither the
DINOv3 tower nor CLIP.

Both are deterministic functions of their input (a crc32-seeded generator, as
``stub_text_encoder`` does -- ``hash()`` is salted per process) and expose only
what ``ABCDiTPolicy`` reads: ``hidden_size`` / ``encode_image_tokens`` /
``backbone_parameters`` on the vision side, ``output_dim`` / ``__call__`` on
the text side. Both swallow the real modules' kwargs so a config can swap them
in by ``_target_`` alone.
"""

from __future__ import annotations

import zlib
from typing import List, Sequence

import torch
import torch.nn as nn


class StubVisionTower(nn.Module):
    """A trainable patch embedder with the DINOv3 tower's call convention."""

    def __init__(
        self,
        hidden_size: int = 32,
        num_patches: int = 4,
        image_size: int = 32,
        **kwargs,  # model_name / freeze / bf16_autocast are accepted and ignored
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_patches = int(num_patches)
        self.image_size = int(image_size)
        self.freeze = bool(kwargs.get("freeze", False))
        self.tower = nn.Linear(3, self.hidden_size)

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) in [0, 1] -> (N, num_patches, hidden_size)."""
        pooled = images.mean(dim=(-2, -1))  # (N, 3)
        tokens = self.tower(pooled)[:, None, :]
        return tokens.expand(-1, self.num_patches, -1).contiguous()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image_tokens(images)

    def backbone_parameters(self) -> List[nn.Parameter]:
        return list(self.tower.parameters())


class StubTaskEncoder(nn.Module):
    """One deterministic L2-normalised vector per prompt string."""

    def __init__(self, output_dim: int = 16, **kwargs) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.calls = 0
        # A parameter so ``.to(device)`` / ``.float()`` reach the module.
        self.register_buffer("_anchor", torch.zeros(1), persistent=False)

    def forward(self, prompts: Sequence[str]) -> torch.Tensor:
        self.calls += 1
        rows = []
        for prompt in prompts:
            generator = torch.Generator().manual_seed(zlib.crc32(str(prompt).encode()))
            row = torch.randn(self.output_dim, generator=generator)
            rows.append(row / row.norm().clamp(min=1e-6))
        return torch.stack(rows, dim=0).to(self._anchor.device)
