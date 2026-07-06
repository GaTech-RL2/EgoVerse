"""Diagonal Gaussian-mixture (GMM) action head for BC-RNN.

Faithful port of robomimic's ``RNNGMMActorNetwork`` GMM semantics
(``robomimic/models/policy_nets.py::RNNGMMActorNetwork.forward_train`` +
``robomimic/algo/bc.py::BC_RNN_GMM``). The LSTM per-step hidden state is
decoded into an explicit ``num_modes`` (M)-component diagonal-Gaussian mixture
over the ``action_dim`` (D) action. Training minimises the negative
log-likelihood (NLL) of the demo action under that mixture.

robomimic semantics matched verbatim (policy_nets.py forward_train):
    # apply tanh squashing to mean if not using tanh-GMM to ensure [-1,1]
    if not self.use_tanh:
        means = torch.tanh(means)
    if self.low_noise_eval and (not self.training):
        scales = torch.ones_like(means) * 1e-4
    else:
        scales = self.activations[self.std_activation](scales) + self.min_std
    component_distribution = D.Normal(loc=means, scale=scales)
    component_distribution = D.Independent(component_distribution, 1)
    mixture_distribution = D.Categorical(logits=logits)
    dists = D.MixtureSameFamily(...)
And at rollout, ``forward_step`` returns ``dists.sample()`` — i.e. eval SAMPLES
from the (low-noise) mixture (NOT argmax-mode-mean). We replicate that exactly
via ``torch.distributions`` so the numerics are identical.

NOTE (normalization range — corrected 2026-06-04):
robomimic's GMM assumes actions pre-normalized to [-1,1] (hence the tanh on the
mean, which bounds the mode means to (-1,1)). The earlier claim that our
pipeline maps actions to "~[0,1]" was WRONG. With norm_mode="quantile" the
mapping is 2*((x - q1)/(q99 - q1)) - 1, i.e. q1 -> -1 and q99 -> +1, so the
~2% of tail actions beyond [q1,q99] (e.g. cursor at screen edges, very common
in this data) land OUTSIDE [-1,1] and the tanh'd means can NEVER reach them.
FIX for the BC-RNN replica: use norm_mode="minmax"
(x_norm = 2*(x - min)/(max - min) - 1), which maps the full [min,max] action
range EXACTLY into [-1,+1] so robomimic's tanh assumption holds for every
target. The unnormalization back to pixels happens in the algo, unchanged.
"""

from typing import Optional

import torch
import torch.distributions as D
import torch.nn as nn


class GMMActionHead(nn.Module):
    """Decode a hidden state ``(..., d_model)`` into a diagonal-GMM over actions.

    The projection emits ``chunk_len * M * (2D + 1)`` numbers per step: for each
    of the ``chunk_len`` action positions, M means (D each), M raw scales (D
    each), and M mixture logits (1 each). With ``chunk_len == 1`` (default) this
    is exactly ``M * (2D + 1)`` and the head is BYTE-IDENTICAL (state_dict +
    forward) to the pre-existing single-action head — the chunk axis is only
    materialised when ``chunk_len > 1`` (the ``chunk_len == 1`` branch takes the
    exact original code path: same reshape, same dist batch shape, same sample
    RNG draw).

    ACTION CHUNKING (chunk_len > 1): each of the ``chunk_len`` positions gets its
    OWN independent M-mode diagonal GMM over the D-dim action. The NLL sums the
    per-position log-probs (so each obs-step contributes one scalar = the sum of
    its chunk-position log-probs) and then applies the SAME masked/unmasked
    obs-step reduction as the single-action head. Decoding samples each position
    independently (low_noise_eval applies per position exactly as today).

    std parameterization (robomimic): ``std = softplus(raw) + min_std`` with
    ``min_std=1e-4`` default and ``std_activation='softplus'``.

    means are tanh-squashed to [-1, 1] (robomimic ``use_tanh=False`` branch).

    Args:
        d_model:    input hidden width.
        action_dim: action dimensionality D.
        chunk_len:  number of action positions predicted per obs-step (>= 1).
            Default 1 == the single-action head (byte-identical). With
            ``chunk_len = C``, the head predicts the next C actions per obs-step
            (action chunking for the obs-strided BC-RNN-Transformer-Chunk8).
        num_modes:  number of mixture components M (>= 1). robomimic study=5.
        min_std:    minimum std added after softplus. robomimic default 1e-4.
        std_activation: only 'softplus' supported (robomimic study default).
        low_noise_eval: robomimic eval trick — at eval (``not self.training``),
            build the mixture with a hard std of 1e-4 and SAMPLE from it
            (categorical over modes, then that Gaussian). True == study default.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        num_modes: int = 5,
        min_std: float = 1e-4,
        std_activation: str = "softplus",
        low_noise_eval: bool = True,
        chunk_len: int = 1,
        head_hidden_dim: int | None = None,
        head_n_layers: int = 0,
    ):
        super().__init__()
        if num_modes < 1:
            raise ValueError(f"num_modes must be >= 1, got {num_modes}")
        if int(chunk_len) < 1:
            raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
        if std_activation != "softplus":
            raise ValueError(
                "robomimic study uses std_activation='softplus'; "
                f"got {std_activation!r}"
            )
        self.action_dim = int(action_dim)
        self.num_modes = int(num_modes)
        self.chunk_len = int(chunk_len)
        self.min_std = float(min_std)
        self.std_activation = std_activation
        self.low_noise_eval = bool(low_noise_eval)
        # per_step = params for ONE GMM (M means + M scales + M logits over D).
        self.per_step = self.num_modes * (2 * self.action_dim + 1)
        # Optional MLP trunk BEFORE the projection (capacity knob, 2026-06-11).
        # Default (head_hidden_dim None / head_n_layers 0) -> nn.Identity, so the
        # proj keeps the EXACT original nn.Linear shape + init draw (byte-
        # identical to pre-existing heads). When set, build head_n_layers
        # Linear+GELU blocks of width head_hidden_dim feeding the projection.
        if head_hidden_dim and head_n_layers > 0:
            layers = []
            in_dim = d_model
            for _ in range(int(head_n_layers)):
                layers += [nn.Linear(in_dim, int(head_hidden_dim)), nn.GELU()]
                in_dim = int(head_hidden_dim)
            self.trunk = nn.Sequential(*layers)
            proj_in = int(head_hidden_dim)
        else:
            self.trunk = nn.Identity()
            proj_in = d_model
        # proj emits chunk_len independent GMMs. chunk_len==1 + Identity trunk =>
        # out_features == per_step (identical to the original head).
        self.proj = nn.Linear(proj_in, self.chunk_len * self.per_step)

    # ---- params ----------------------------------------------------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Hidden ``(..., d_model)`` -> flat GMM params ``(..., M*(2D+1))``."""
        return self.proj(self.trunk(h))

    def _make_dist(self, raw: torch.Tensor) -> D.MixtureSameFamily:
        """Flat params -> a MixtureSameFamily distribution.

        Replicates robomimic ``RNNGMMActorNetwork.forward_train`` exactly:
        tanh-squashed means; std = softplus(raw)+min_std in train, or a hard
        1e-4 scale at eval when ``low_noise_eval`` (and not training).

        chunk_len == 1 (default): ``raw`` is ``(..., M*(2D+1))`` and the dist has
        batch shape ``(...)`` — the EXACT original code path (byte-identical
        reshape, dist batch shape, and sample RNG draw).

        chunk_len > 1: ``raw`` is ``(..., chunk_len*M*(2D+1))``; it is first
        reshaped to expose a chunk axis ``(..., chunk_len, M*(2D+1))`` so the
        dist has batch shape ``(..., chunk_len)`` — i.e. ``chunk_len``
        INDEPENDENT M-mode GMMs, one per action position.
        """
        M, D_ = self.num_modes, self.action_dim
        if self.chunk_len > 1:
            # expose the chunk axis: (..., C*P) -> (..., C, P) with P=M*(2D+1).
            # everything below then operates per-position (extra batch dim C).
            lead = raw.shape[:-1]
            raw = raw.reshape(*lead, self.chunk_len, self.per_step)
        lead = raw.shape[:-1]
        means = raw[..., : M * D_].reshape(*lead, M, D_)
        scales = raw[..., M * D_ : 2 * M * D_].reshape(*lead, M, D_)
        logits = raw[..., 2 * M * D_ :].reshape(*lead, M)

        # robomimic: means = torch.tanh(means)  (use_tanh=False branch)
        means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            # robomimic: scales = torch.ones_like(means) * 1e-4
            scales = torch.ones_like(means) * 1e-4
        else:
            # robomimic: scales = softplus(scales) + min_std
            scales = torch.nn.functional.softplus(scales) + self.min_std

        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)
        mixture_distribution = D.Categorical(logits=logits)
        return D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

    # ---- loss ------------------------------------------------------------

    def log_prob(self, raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """GMM log-likelihood of ``target`` under the mixture.

        chunk_len == 1: ``target`` ``(..., D)`` -> ``(...)`` (original path).
        chunk_len > 1:  ``target`` ``(..., chunk_len, D)`` -> ``(..., chunk_len)``
            (one log-prob per action position; the dist batch shape carries the
            chunk axis, so ``MixtureSameFamily.log_prob`` broadcasts naturally).

        robomimic: ``log_probs = dists.log_prob(batch["actions"])``.
        """
        dist = self._make_dist(raw)
        return dist.log_prob(target)

    def nll(
        self,
        raw: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean NLL of ``target`` under the mixture. ``mask`` ``(...)`` (1=valid,
        per OBS-STEP) is applied when given (mean over valid entries), else a
        plain mean.

        chunk_len == 1: ``lp`` is ``(...)`` and the reduction is EXACTLY the
        original (byte-identical).

        chunk_len > 1: ``lp`` is ``(..., chunk_len)``; SUM the per-position
        log-probs over the chunk axis -> ``(...)`` (each obs-step contributes the
        joint log-likelihood of its whole predicted action chunk), then apply the
        SAME obs-step ``mask`` reduction. The mask is per obs-step (one entry per
        window step), not per chunk position — in repeat_unmasked every step
        (incl. repeat-padded tails) is counted, matching paper-exact.

        robomimic: ``action_loss = -predictions["log_probs"].mean()`` (no mask;
        our mask handles the packed->padded unpack of variable-length windows).
        """
        lp = self.log_prob(raw, target)  # (...) or (..., chunk_len)
        if self.chunk_len > 1:
            lp = lp.sum(dim=-1)  # sum joint LL over the chunk positions -> (...)
        if mask is None:
            return -lp.mean()
        mask = mask.to(lp.dtype)
        return -(lp * mask).sum() / (mask.sum() + 1e-8)

    # ---- decode ----------------------------------------------------------

    @torch.no_grad()
    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """Flat params -> action(s).

        chunk_len == 1: ``raw`` ``(..., M*(2D+1))`` -> action ``(..., D)``
            (EXACT original path — same dist batch shape and sample RNG draw).
        chunk_len > 1:  ``raw`` ``(..., chunk_len*M*(2D+1))`` -> the chunk of
            actions ``(..., chunk_len, D)`` (each position sampled from its own
            GMM; low_noise_eval applies per position exactly as today).

        Faithful to robomimic ``forward_step`` which returns ``dists.sample()``.
        With ``low_noise_eval`` (eval mode), ``_make_dist`` uses a hard 1e-4
        std, so the sample is effectively the chosen mode's mean plus negligible
        noise — exactly robomimic's deterministic-ish eval behavior. We call
        ``.sample()`` so the numerics match (categorical over modes -> that
        Gaussian). Caller is responsible for being in eval mode (``.eval()``)
        for the low-noise path to engage.
        """
        dist = self._make_dist(raw)
        return dist.sample()
