"""Swappable residual mixer for the H-Net dechunker.

Replaces the additive ``out = up + residual`` (trunk reconstruction + linear
skip) with a configurable combiner, so we can ablate HOW the trunk output and
the residual are fused. This is Fix 2 for the trunk-bypass loophole: a
non-linear mixer removes the clean LINEAR path that lets one embodiment route
its per-token detail around the shared trunk via the residual.

ALL variants start ``~= up + residual`` at init (the extra path is zero-gated)
so the change is training-stable from step 0. Add a new variant by registering
it in ``_MIXERS``.  See vault: 30_Projects/RL2/Transformer HNet/Findings.md.
"""

import torch
import torch.nn as nn


class AdditiveMixer(nn.Module):
    """out = up + residual (upstream behaviour; no new params)."""

    def forward(self, up, residual):
        return up + residual


class MLPMixer(nn.Module):
    """out = up + residual + MLP([up, residual]).  Last layer zero-init -> the
    MLP contributes nothing at step 0, so training starts exactly additive.

    With the duplicate upsampler the ``up`` (high-level plan) is CONSTANT within
    a chunk, so this MLP is the ONLY source of within-chunk variation -- it must
    be expressive. Default = 2 hidden layers, 4x width."""

    def __init__(self, dim, hidden_mult: float = 4.0, n_layers: int = 2):
        super().__init__()
        h = int(dim * hidden_mult)
        layers = [nn.Linear(2 * dim, h), nn.GELU()]
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(h, h), nn.GELU()]
        layers += [nn.Linear(h, dim)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, up, residual):
        return up + residual + self.net(torch.cat([up, residual], dim=-1))


class AttnMixer(nn.Module):
    """Treat (up, residual) as a length-2 sequence per token, self-attend, and
    read out the ``up`` slot.  Zero-init gate -> starts additive."""

    def __init__(self, dim, n_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, up, residual):
        base = up + residual
        seq = torch.stack([up, residual], dim=1)  # (T, 2, dim)
        h = self.norm(seq)
        a, _ = self.attn(h, h, h, need_weights=False)  # (T, 2, dim)
        return base + self.gate * a[:, 0]


_MIXERS = {
    "additive": AdditiveMixer,
    "add": AdditiveMixer,
    "none": AdditiveMixer,
    "mlp": MLPMixer,
    "attn": AttnMixer,
}


def build_residual_mixer(kind, dim, **kwargs):
    """kind: 'additive' (default) | 'mlp' | 'attn'.  dim = fine-token dim."""
    key = (kind or "additive").lower()
    if key not in _MIXERS:
        raise ValueError(
            f"unknown residual_mixer {kind!r}; choices: {sorted(set(_MIXERS))}"
        )
    cls = _MIXERS[key]
    return cls() if cls is AdditiveMixer else cls(dim, **kwargs)
