"""Token assembly and rollout machinery shared by every AR variant.

All four variants are the same causal transformer over a token sequence; they
differ only in which token TYPES appear, which positions carry a loss, and
whether a prediction is fed back into the next step's input. Keeping that
machinery here means a variant is a small declarative class rather than
another copy of the unroll loop.

CONVENTION: one timestep contributes tokens in a fixed order, and every
variant uses the same order so a checkpoint's positional embeddings mean the
same thing across variants:

    [ image_t , state_t , action_t ]

A variant simply omits the token types it does not use.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TokenSpec:
    """Which token types a variant emits per timestep, in canonical order."""

    image: bool = True
    state: bool = True
    action: bool = True

    @property
    def per_step(self) -> int:
        return int(self.image) + int(self.state) + int(self.action)

    def index_of(self, kind: str) -> int:
        """Offset of `kind` within one timestep's token group."""
        order = [k for k, on in
                 (("image", self.image), ("state", self.state),
                  ("action", self.action)) if on]
        if kind not in order:
            raise KeyError(f"variant emits no {kind!r} token: {order}")
        return order.index(kind)


class Tokenizer(nn.Module):
    """Project heterogeneous inputs into one shared token width.

    Separate projections per modality (rather than one shared linear on a
    concatenation) so a variant that drops a modality does not change the
    meaning of the others' weights -- that keeps variants checkpoint-
    comparable, which is the point of running them as an ablation.
    """

    def __init__(self, image_dim: int, state_dim: int, action_dim: int,
                 d_model: int):
        super().__init__()
        self.image = nn.Linear(image_dim, d_model)
        self.state = nn.Linear(state_dim, d_model)
        self.action = nn.Linear(action_dim, d_model)
        # Learned per-type embedding: without it the backbone cannot tell an
        # action token from a state token at the same position.
        self.type_emb = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.type_emb, std=0.02)

    def forward(self, spec: TokenSpec, image, state, action):
        """(B, T, *) inputs -> (B, T * spec.per_step, d_model) tokens."""
        b, t = state.shape[:2]
        toks = []
        if spec.image:
            toks.append(self.image(image) + self.type_emb[0])
        if spec.state:
            toks.append(self.state(state) + self.type_emb[1])
        if spec.action:
            toks.append(self.action(action) + self.type_emb[2])
        # Interleave so timestep order is preserved: t0 tokens, then t1, ...
        stacked = torch.stack(toks, dim=2)              # (B, T, K, D)
        return stacked.reshape(b, t * len(toks), -1)


def gather_positions(feats: torch.Tensor, spec: TokenSpec, kind: str,
                     n_steps: int) -> torch.Tensor:
    """Pull the per-timestep features sitting at `kind`'s token position."""
    off = spec.index_of(kind)
    idx = torch.arange(n_steps, device=feats.device) * spec.per_step + off
    return feats.index_select(1, idx)


class InverseDynamics(nn.Module):
    """(s_t, s_{t+1}) -> a_t.

    Deliberately small and separate from the backbone: the IDM variant's claim
    is that predicting STATE is the hard part and action recovery is nearly
    mechanical, so giving the IDM a large capacity would blur exactly the
    comparison the variant exists to make.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s_t, s_next):
        return self.net(torch.cat([s_t, s_next], dim=-1))
