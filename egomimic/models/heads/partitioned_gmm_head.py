"""Partitioned-mode GMM action head for the dual-stream H-Net.

Subclass of :class:`GMMActionHead` whose mixture modes are split into two
groups parameterised by DIFFERENT input streams:

  * ``k_agnostic`` modes come from the *agnostic* top-of-trunk token ``a_top``
    (the weight-shared stream ``A``);
  * ``k_specific`` modes come from the per-embodiment *specific* token ``s``
    (the stream ``S``).
  * the joint mixture logits over all ``M = k_agnostic + k_specific`` modes
    come from a gate on ``cat([a_top, s], -1)``.

The trick that keeps this minimal: we call ``super().__init__`` with
``num_modes = k_agnostic + k_specific`` so EVERY inherited attribute
(``per_step``, ``num_modes``, ``action_dim``, ``chunk_len`` …) and EVERY
inherited method (``_make_dist`` / ``log_prob`` / ``nll`` / ``decode``) is valid
unchanged. ``forward`` then assembles a ``raw`` tensor in the EXACT flat layout
``GMMActionHead._make_dist`` consumes, so the loss / decode path is the parent's,
byte-for-byte.

Flat layout (verified against ``GMMActionHead._make_dist`` for ``chunk_len>=1``):
per chunk position the ``per_step = M*(2D+1)`` numbers are ordered
``[ all_means (M*D), all_scales (M*D), all_logits (M) ]`` and ``_make_dist``
reshapes the ``M*D`` blocks row-major to ``(M, D)`` — i.e. mode 0's D values,
then mode 1's D values, …. So concatenating the agnostic modes BEFORE the
specific modes along the mode axis is simply ``cat([A_*, S_*], dim=mode_axis)``
flattened, which is what we build here. The first ``k_agnostic`` modes are the
A-modes, the last ``k_specific`` are the S-modes — a stable, documented mapping.

The parent's single ``self.proj`` Linear is removed (we never use it) so it does
not become a dead, never-trained parameter (which would trip a DDP unused-param
check on the real multi-GPU run). Likewise the optional capacity trunk is built
PER-STREAM here (``trunk_A`` / ``trunk_S``); we do NOT forward ``head_hidden_dim``
to the parent, so the parent's ``self.trunk`` stays ``nn.Identity`` (no dead
params).
"""

from typing import Optional

import torch
import torch.nn as nn

from egomimic.models.heads.gmm_head import GMMActionHead


class PartitionedGMMActionHead(GMMActionHead):
    """GMM head with modes partitioned across the agnostic / specific streams.

    Args:
        d_model:    trunk hidden width (the per-stream token width). Both
                    ``a_top`` and ``s`` arrive at this width.
        action_dim: action dimensionality D.
        k_agnostic: number of mixture modes parameterised from ``a_top``.
        k_specific: number of mixture modes parameterised from ``s``.
        chunk_len:  number of action positions predicted per obs-step (>= 1).
        min_std / low_noise_eval / std_activation: passed through to the parent
                    (robomimic GMM semantics, unchanged).
        head_hidden_dim / head_n_layers: optional PER-STREAM MLP trunk applied
                    to ``a_top`` (feeding ``proj_A``) and ``s`` (feeding
                    ``proj_S``) before the projections. The gate always sees the
                    RAW ``cat([a_top, s])`` at ``2*d_model``. Default (None / 0)
                    => no trunk (Identity), proj_in == d_model.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        k_agnostic: int,
        k_specific: int,
        chunk_len: int = 1,
        min_std: float = 1e-4,
        std_activation: str = "softplus",
        low_noise_eval: bool = True,
        head_hidden_dim: Optional[int] = None,
        head_n_layers: int = 0,
    ):
        if int(k_agnostic) < 0 or int(k_specific) < 0:
            raise ValueError(
                f"k_agnostic / k_specific must be >= 0 (got {k_agnostic}, {k_specific})"
            )
        if int(k_agnostic) + int(k_specific) < 1:
            raise ValueError("k_agnostic + k_specific must be >= 1")
        # Build the parent with the TOTAL mode count so all inherited
        # attributes / methods (per_step, _make_dist, nll, decode) are valid.
        # Do NOT pass head_hidden_dim to the parent — we want its self.trunk to
        # stay Identity (no dead params); our trunks are per-stream below.
        super().__init__(
            d_model=d_model,
            action_dim=action_dim,
            num_modes=int(k_agnostic) + int(k_specific),
            min_std=min_std,
            std_activation=std_activation,
            low_noise_eval=low_noise_eval,
            chunk_len=chunk_len,
            head_hidden_dim=None,
            head_n_layers=0,
        )
        self.k_agnostic = int(k_agnostic)
        self.k_specific = int(k_specific)

        # The parent created a single self.proj we never use. Remove it so it is
        # not a never-trained parameter (would trip DDP unused-param detection).
        if hasattr(self, "proj"):
            del self.proj
        # Parent's self.trunk is Identity (we passed head_hidden_dim=None); drop
        # it too for clarity (we route through per-stream trunks instead).
        if hasattr(self, "trunk"):
            del self.trunk

        D = self.action_dim
        C = self.chunk_len

        # Optional per-stream capacity trunk (mirrors GMMActionHead's trunk
        # recipe: head_n_layers blocks of Linear+GELU at head_hidden_dim).
        def _make_trunk():
            if head_hidden_dim and head_n_layers > 0:
                layers = []
                in_dim = d_model
                for _ in range(int(head_n_layers)):
                    layers += [nn.Linear(in_dim, int(head_hidden_dim)), nn.GELU()]
                    in_dim = int(head_hidden_dim)
                return nn.Sequential(*layers), int(head_hidden_dim)
            return nn.Identity(), d_model

        self.trunk_A, proj_in_A = _make_trunk()
        self.trunk_S, proj_in_S = _make_trunk()

        # Projections: A-stream emits (means+scales) for the k_agnostic modes,
        # S-stream for the k_specific modes. Each mode is a D-dim diagonal
        # Gaussian -> 2*D numbers (mean + raw scale) per mode, per chunk pos.
        self.proj_A = nn.Linear(proj_in_A, C * self.k_agnostic * 2 * D)
        self.proj_S = nn.Linear(proj_in_S, C * self.k_specific * 2 * D)
        # Gate: joint logits over ALL modes from the concatenated streams.
        self.gate = nn.Linear(2 * d_model, C * (self.k_agnostic + self.k_specific))

    # ------------------------------------------------------------------
    # forward: (a_top, s) -> flat GMM params in _make_dist's layout.
    # ------------------------------------------------------------------

    def forward(self, a_top: torch.Tensor, s: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Assemble flat GMM params from the two streams.

        ``a_top`` / ``s`` are ``(..., d_model)`` (packed: ``(T_total, d_model)``).
        Returns ``raw`` of shape ``(..., chunk_len * per_step)`` laid out exactly
        as ``GMMActionHead._make_dist`` expects, so ``nll`` / ``log_prob`` /
        ``decode`` (all inherited, unchanged) operate correctly.
        """
        lead = a_top.shape[:-1]
        D = self.action_dim
        C = self.chunk_len
        Ka, Ks = self.k_agnostic, self.k_specific
        M = Ka + Ks

        # Per-stream means+scales. Reshape to expose (chunk, mode, 2, D) so we
        # can split mean vs scale and concat the two streams' modes.
        a_raw = self.proj_A(self.trunk_A(a_top)).reshape(*lead, C, Ka, 2, D)
        s_raw = self.proj_S(self.trunk_S(s)).reshape(*lead, C, Ks, 2, D)

        means_A, scales_A = a_raw[..., 0, :], a_raw[..., 1, :]  # (..., C, Ka, D)
        means_S, scales_S = s_raw[..., 0, :], s_raw[..., 1, :]  # (..., C, Ks, D)

        # Concat modes: A-modes first, then S-modes -> (..., C, M, D).
        means = torch.cat([means_A, means_S], dim=-2)
        scales = torch.cat([scales_A, scales_S], dim=-2)

        # Joint logits over all M modes from cat([a_top, s]) -> (..., C, M).
        logits = self.gate(torch.cat([a_top, s], dim=-1)).reshape(*lead, C, M)

        # --- mode-weighting probe (read + logged by DualStreamProbeCallback) ---
        # Mixture mass on the agnostic modes (first k_agnostic) vs specific modes
        # (last k_specific), averaged over all positions. Detached; does NOT
        # affect the returned raw / the loss.
        with torch.no_grad():
            probs = torch.softmax(logits.detach().float(), dim=-1)  # (..., C, M)
            self._last_w_A = float(probs[..., : self.k_agnostic].sum(-1).mean())
            self._last_w_S = float(probs[..., self.k_agnostic :].sum(-1).mean())
            self._fired = True

        # Flatten to _make_dist's per-position layout:
        #   [ means (M*D), scales (M*D), logits (M) ]  per chunk position,
        # then flatten the chunk axis -> (..., C * per_step).
        means_flat = means.reshape(*lead, C, M * D)
        scales_flat = scales.reshape(*lead, C, M * D)
        raw = torch.cat([means_flat, scales_flat, logits], dim=-1)  # (..., C, per_step)
        raw = raw.reshape(*lead, C * self.per_step)
        return raw
