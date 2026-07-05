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
from egomimic.models.hnet.per_emb import per_emb, pick


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
        d_model_a: Optional[int] = None,
        d_model_s: Optional[int] = None,
    ):
        # Per-stream input widths for the modular multi-stream trunk: a_top
        # arrives at d_a, s at d_s. Default to the shared d_model (back-compat
        # with the symmetric two-trunk configs).
        d_a = int(d_model_a) if d_model_a is not None else int(d_model)
        d_s = int(d_model_s) if d_model_s is not None else int(d_model)
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
        def _make_trunk(d_in):
            if head_hidden_dim and head_n_layers > 0:
                layers = []
                in_dim = d_in
                for _ in range(int(head_n_layers)):
                    layers += [nn.Linear(in_dim, int(head_hidden_dim)), nn.GELU()]
                    in_dim = int(head_hidden_dim)
                return nn.Sequential(*layers), int(head_hidden_dim)
            return nn.Identity(), d_in

        self.trunk_A, proj_in_A = _make_trunk(d_a)
        self.trunk_S, proj_in_S = _make_trunk(d_s)

        # Projections: A-stream emits (means+scales) for the k_agnostic modes,
        # S-stream for the k_specific modes. Each mode is a D-dim diagonal
        # Gaussian -> 2*D numbers (mean + raw scale) per mode, per chunk pos.
        self.proj_A = nn.Linear(proj_in_A, C * self.k_agnostic * 2 * D)
        self.proj_S = nn.Linear(proj_in_S, C * self.k_specific * 2 * D)
        # Gate: joint logits over ALL modes from the concatenated streams
        # (per-stream widths d_a + d_s).
        self.gate = nn.Linear(d_a + d_s, C * (self.k_agnostic + self.k_specific))

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


class _ModeHead(nn.Module):
    """Per-stream mode parameterizer: optional MLP trunk + Linear proj.
    ``x (d_in) -> (C * K * 2 * D)`` raw mean/scale numbers for K diagonal modes."""

    def __init__(self, d_in, C, K, D, hidden=None, n_layers=0):
        super().__init__()
        if hidden and n_layers > 0:
            layers, inn = [], d_in
            for _ in range(int(n_layers)):
                layers += [nn.Linear(inn, int(hidden)), nn.GELU()]
                inn = int(hidden)
            self.trunk = nn.Sequential(*layers)
            proj_in = int(hidden)
        else:
            self.trunk = nn.Identity()
            proj_in = d_in
        self.proj = nn.Linear(proj_in, C * K * 2 * D)

    def forward(self, x):
        return self.proj(self.trunk(x))


class _GateHead(nn.Module):
    """Mixture-weight gate: an ``n_layers``-deep MLP over ``cat[a_top, s]`` ->
    ``C*M`` logits (replaces the bare ``Linear`` gate -> as much flexibility as the
    mean/scale trunks)."""

    def __init__(self, d_in, C, M, hidden=None, n_layers=3):
        super().__init__()
        h = int(hidden) if hidden else d_in
        layers, inn = [], d_in
        for _ in range(max(1, int(n_layers))):
            layers += [nn.Linear(inn, h), nn.GELU()]
            inn = h
        layers += [nn.Linear(inn, C * M)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DualPartitionedGMMHead(GMMActionHead):
    """Partitioned GMM head built as a PARALLEL CIRCUIT (matches the dual-stream
    "agnostic shared throughout / specific per-emb" principle), in ONE module
    dispatched on ``emb_id`` via the shared ``PerEmb`` abstraction:

      * agnostic modes (``k_agnostic``, parameterised from ``a_top``) -> a SHARED
        ``_ModeHead`` (``share_agnostic=True``) so the agnostic decode is truly
        embodiment-invariant END-TO-END (trunk + this head). ``share_agnostic=False``
        falls back to a per-emb agnostic head (the legacy fully-per-emb behavior)
        -> the A/B knob.
      * specific modes (``k_specific``, from ``s``) -> PER-EMB ``_ModeHead``.
      * gate (joint logits over all ``M`` modes, from ``cat[a_top, s]``) -> PER-EMB
        ``_GateHead`` MLP (``gate_n_layers`` deep; the old ``PartitionedGMMActionHead``
        used a bare ``Linear``).

    Reuses ``GMMActionHead._make_dist`` / ``nll`` / ``decode`` unchanged (same flat
    layout: ``[means(M*D), scales(M*D), logits(M)]`` per chunk pos, A-modes first).
    ``is_emb_partitioned=True`` tells the outer stage to pass ``emb_id`` to forward.
    """

    is_emb_partitioned = True

    def __init__(
        self,
        d_model,
        action_dim,
        k_agnostic,
        k_specific,
        embodiments,
        chunk_len=1,
        min_std=1e-4,
        std_activation="softplus",
        low_noise_eval=True,
        head_hidden_dim=None,
        head_n_layers=0,
        gate_n_layers=3,
        gate_hidden_dim=None,
        d_model_a=None,
        d_model_s=None,
        share_agnostic=True,
    ):
        d_a = int(d_model_a) if d_model_a is not None else int(d_model)
        d_s = int(d_model_s) if d_model_s is not None else int(d_model)
        if (
            int(k_agnostic) < 0
            or int(k_specific) < 0
            or int(k_agnostic) + int(k_specific) < 1
        ):
            raise ValueError(
                f"need k_agnostic,k_specific >= 0 and sum >= 1 "
                f"(got {k_agnostic}, {k_specific})"
            )
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
        # parent's single proj/trunk are unused here -> drop (no dead DDP params).
        if hasattr(self, "proj"):
            del self.proj
        if hasattr(self, "trunk"):
            del self.trunk
        self.k_agnostic, self.k_specific = int(k_agnostic), int(k_specific)
        self.embodiments = list(embodiments)
        self.share_agnostic = bool(share_agnostic)
        D, C = self.action_dim, self.chunk_len
        Ka, Ks, M = self.k_agnostic, self.k_specific, self.num_modes
        ghd = gate_hidden_dim if gate_hidden_dim is not None else head_hidden_dim
        # agnostic-mode head: SHARED (share_agnostic) or per-emb; specific + gate per-emb.
        agn_embs = None if self.share_agnostic else self.embodiments
        self.A_head = per_emb(
            lambda: _ModeHead(d_a, C, Ka, D, head_hidden_dim, head_n_layers), agn_embs
        )
        self.S_head = per_emb(
            lambda: _ModeHead(d_s, C, Ks, D, head_hidden_dim, head_n_layers),
            self.embodiments,
        )
        self.gate = per_emb(
            lambda: _GateHead(d_a + d_s, C, M, ghd, gate_n_layers), self.embodiments
        )

    def forward(self, a_top, s, emb_id=None):  # type: ignore[override]
        lead = a_top.shape[:-1]
        D, C = self.action_dim, self.chunk_len
        Ka, Ks, M = self.k_agnostic, self.k_specific, self.num_modes
        a_raw = pick(self.A_head, emb_id)(a_top).reshape(*lead, C, Ka, 2, D)
        s_raw = pick(self.S_head, emb_id)(s).reshape(*lead, C, Ks, 2, D)
        means = torch.cat(
            [a_raw[..., 0, :], s_raw[..., 0, :]], dim=-2
        )  # (..., C, M, D)
        scales = torch.cat([a_raw[..., 1, :], s_raw[..., 1, :]], dim=-2)
        logits = pick(self.gate, emb_id)(torch.cat([a_top, s], dim=-1)).reshape(
            *lead, C, M
        )
        with torch.no_grad():
            probs = torch.softmax(logits.detach().float(), dim=-1)
            wA = float(probs[..., :Ka].sum(-1).mean())
            wS = float(probs[..., Ka:].sum(-1).mean())
            self._last_w_A, self._last_w_S, self._fired = wA, wS, True
            # single shared head dispatched over BOTH embodiments per step -> stash
            # PER-EMB so the probe logs wA/wS for each (else only the last emb shows).
            if not hasattr(self, "_w_by_emb"):
                self._w_by_emb = {}
            self._w_by_emb[emb_id if emb_id is not None else "x"] = (wA, wS)
        means_flat = means.reshape(*lead, C, M * D)
        scales_flat = scales.reshape(*lead, C, M * D)
        raw = torch.cat([means_flat, scales_flat, logits], dim=-1).reshape(
            *lead, C * self.per_step
        )
        return raw
