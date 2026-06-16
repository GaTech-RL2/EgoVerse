"""
H-Net policy for EgoVerse — stage-based architecture.

The policy treats the action sequence as the input modality (autoregressive,
causal). Observations are encoded by a ``CondEncoderModule`` into a
``cond_dict`` carried by an ``HNetContext`` that is threaded through the
stage tree. Each stage reads whichever cond key it wants (or ignores cond
entirely).

Loss = action MSE (next-action prediction) +
       sum_over_chunkers( weight * ratio_loss(boundary_predictions) ).

The per-chunker ratio-loss weights live inside the chunker stages themselves;
this algo just calls ``ratio_loss_from_aux(ctx.aux)`` after forward.
"""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.hnet.action_heads import (
    action_head_decode,
    action_head_decode_chunk,
    action_head_loss,
    action_head_raw,
    configure_action_head,
)
from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.hnet import HNet as HNetCore
from egomimic.models.hnet.hnet import chunk_stats_from_aux, ratio_loss_from_aux
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


def _apply_token_dropout(x, bos, p, training):
    """Anti-copy-cheat: replace prev-action tokens with BOS at rate ``p`` during
    training so the model cannot copy the previous action and must predict from
    obs (like HPT). Handles (B, T, D) and packed (T_total, D); ``bos`` is (D,).
    p=1.0 drops every action token (pure obs-conditioned)."""
    if (not training) or p <= 0:
        return x
    lead = tuple(x.shape[:-1])
    D = x.shape[-1]
    drop = torch.rand(*(lead + (1,)), device=x.device) < p
    bos_b = bos.reshape((1,) * len(lead) + (D,)).to(x.dtype).expand_as(x)
    return torch.where(drop, bos_b, x)


def _action_head_cfg(kwargs):
    """Pull the swappable-action-head knobs out of the algo ``**kwargs`` into
    a cfg dict consumed by ``configure_action_head``. Defaults reproduce the
    continuous MSE head (no behavior change for existing runs)."""
    cfg = {
        "mode": str(kwargs.get("action_head", "continuous")),
        "n_bins": int(kwargs.get("n_bins", 256) or 256),
        "bin_min": float(kwargs.get("bin_min", -4.0)),
        "bin_max": float(kwargs.get("bin_max", 4.0)),
        "sample_temp": float(kwargs.get("sample_temp", 1.0)),
        # One-shot action chunk: K=1 (default) keeps the single next-action head;
        # K>1 makes each timestep predict the next K actions in one shot.
        "chunk_k": int(kwargs.get("chunk_k", 1) or 1),
        # GMM head knobs (only consulted when action_head=gmm). None/absent ->
        # default 5; an explicit 0 must survive so configure_action_head's
        # ``M < 1`` guard can raise (the old ``or 5`` swallowed a falsy 0).
        "gmm_num_modes": (
            int(kwargs["gmm_num_modes"])
            if kwargs.get("gmm_num_modes") is not None
            else 5
        ),
        "gmm_sample": bool(kwargs.get("gmm_sample", False)),
    }
    # Std clamp bounds are optional; pass them through only if set so the head
    # falls back to its [-5, 2] log-std defaults otherwise.
    if kwargs.get("gmm_min_std", None) is not None:
        cfg["gmm_min_std"] = float(kwargs["gmm_min_std"])
    if kwargs.get("gmm_max_std", None) is not None:
        cfg["gmm_max_std"] = float(kwargs["gmm_max_std"])
    return cfg


def resolve_embodiment_keys(algo, norm_stats):
    """Populate ``algo.{embodiment_ids, proprio_keys, lang_keys, camera_keys,
    resolved_ac_keys}`` from the ``norm_stats`` (MultiDataset) key topology.

    Factored out so additional H-Net-family algos (e.g. the chunk-token
    baseline) reuse the exact resolution logic instead of re-copying it.
    ``HNet`` / ``HNetFused`` keep their own inline copies for now to avoid
    re-touching already-validated ``__init__`` paths; unifying them onto this
    helper is a safe future cleanup.
    """
    algo.embodiment_ids = {}
    algo.proprio_keys = {}
    algo.lang_keys = {}
    algo.camera_keys = {}
    algo.resolved_ac_keys = {}
    for emb in algo.domains:
        emb_id = get_embodiment_id(emb)
        algo.embodiment_ids[emb] = emb_id
        algo.proprio_keys[emb_id] = []
        algo.lang_keys[emb_id] = []
        algo.camera_keys[emb_id] = []
        for key in norm_stats.keys_of_type("action_keys", emb_id):
            if (
                norm_stats.is_key_with_embodiment(key, emb_id)
                and key == algo.ac_keys[emb]
            ):
                algo.resolved_ac_keys[emb_id] = key
        for key in norm_stats.keys_of_type("proprio_keys", emb_id):
            if norm_stats.is_key_with_embodiment(key, emb_id):
                algo.proprio_keys[emb_id].append(key)
        for key in norm_stats.keys_of_type("lang_keys", emb_id):
            if norm_stats.is_key_with_embodiment(key, emb_id):
                algo.lang_keys[emb_id].append(key)
        for key in norm_stats.keys_of_type("camera_keys", emb_id):
            if norm_stats.is_key_with_embodiment(key, emb_id):
                algo.camera_keys[emb_id].append(key)


class HNetPolicy(nn.Module):
    """
    action-tokenizer → stage-based H-Net → action-detokenizer.

    Owns action_in / action_out projections, BOS token, positional embedding,
    the ``CondEncoderModule``, and the ``HNetCore`` (stage tree).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        cond_encoder: CondEncoderModule,
        hnet: HNetCore,
        action_head_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model

        self.action_in = nn.Linear(action_dim, d_model)
        configure_action_head(self, d_model, action_dim, action_head_cfg)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)
        # F6: positional information is now carried by RoPE inside each
        # MultiHeadAttention (q/k rotation), matching upstream
        # ``goombalab/hnet/hnet/modules/mha.py``. The previous learned
        # absolute ``pos_emb = nn.Parameter(zeros(1, action_horizon,
        # d_model))`` capped episodes at ``action_horizon`` and added a
        # second positional signal on top of RoPE; both are gone.

        self.cond_encoder = cond_encoder
        self.hnet = hnet

        # Sanity-check that the stage tree's outer hidden dim matches d_model.
        if self.hnet.input_hidden_dim != d_model:
            raise ValueError(
                f"hnet.input_hidden_dim ({self.hnet.input_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )
        if self.hnet.output_hidden_dim != d_model:
            raise ValueError(
                f"hnet.output_hidden_dim ({self.hnet.output_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )

    def _build_ctx(self, obs: dict) -> HNetContext:
        cond_dict = self.cond_encoder.encode(obs, self.action_horizon)
        return HNetContext(cond_dict=cond_dict, aux=[], inference_params=None)

    def forward(self, actions: torch.Tensor, obs: dict):
        """
        actions: (B, T, action_dim) ground-truth actions for teacher-forcing.
        obs:     dict of (B, ...) obs tensors.

        Returns: (pred_actions (B, T, action_dim), aux list).
        """
        B, T, _ = actions.shape
        x = self.action_in(actions)
        x = torch.cat([self.bos.expand(B, -1, -1), x[:, :-1]], dim=1)
        x = _apply_token_dropout(
            x, self.bos.reshape(-1), getattr(self, "token_dropout_p", 0.0), self.training
        )
        # F6: no outer pos_emb — RoPE inside the attention modules handles
        # positional information.

        ctx = self._build_ctx(obs)
        h = self.hnet(x, ctx)
        return action_head_raw(self, h), ctx.aux

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """Packed-mode teacher-forced forward.

        Mirrors :meth:`forward` for variable-length episodes packed into a
        single flat stream (FlashAttention-style varlen). The BOS shift and
        ``pos_emb`` indexing happen per sub-sequence — for each subseq
        ``[s, e)``, position s gets BOS and positions s+1..e-1 get
        ``action_in(actions[s..e-2])``; ``pos_emb`` is indexed by ``t - s``
        within each subseq so every episode starts at position 0.

        Args:
            actions_packed: (T_total, action_dim) packed ground-truth actions.
            obs_packed:     dict of (T_total, ...) per-frame obs tensors.
            cu_seqlens:     (B+1,) long, cumulative subseq lengths (starts 0).
            max_seqlen:     int, longest subseq length.

        Returns: (pred_packed (T_total, action_dim), aux).
        """
        device = actions_packed.device
        T_total = actions_packed.shape[0]
        if not torch.is_tensor(cu_seqlens):
            cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.long)
        else:
            cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        # F6: with RoPE handling positions inside each MHA, there is no
        # hard ``action_horizon`` ceiling anymore — the old check on
        # ``max_seqlen > self.action_horizon`` was tied to the size of
        # ``self.pos_emb`` and is removed.

        # 1. Tokenize, then global shift-right by 1, then overwrite each
        #    subseq's first slot with BOS. Under autocast (e.g. bf16) the
        #    activations are downcast but ``self.bos`` stays in fp32; we
        #    match the activation dtype to keep the index-put happy.
        a_emb = self.action_in(actions_packed)  # (T_total, D)
        x_shifted = torch.cat(
            [
                torch.zeros(1, self.d_model, device=device, dtype=a_emb.dtype),
                a_emb[:-1],
            ],
            dim=0,
        )  # (T_total, D)
        bos = self.bos.squeeze(0).squeeze(0).to(a_emb.dtype)  # (D,)
        starts = cu_seqlens[:-1]  # (B,)
        x_shifted = x_shifted.clone()
        x_shifted[starts] = bos

        # F6: no outer per-subseq pos_emb add — the MHA RoPE applies per-subseq
        # rotary positions inside attention. ``cu_seqlens`` is threaded into
        # ``HNetContext`` below and re-derived in ``_forward_packed`` of
        # ``MultiHeadAttention``.
        x_packed = x_shifted
        x_packed = _apply_token_dropout(
            x_packed, bos, getattr(self, "token_dropout_p", 0.0), self.training
        )

        # 3. Packed cond. ``cond_encoder.encode`` expects (B, T, ...); for a
        #    packed stream the simplest path is to feed (1, T_total, ...) and
        #    squeeze the leading dim back out. The encoder's per-frame branch
        #    (state x.dim()==3 / images x.dim()==5) fires correctly because
        #    obs_packed already carries the per-frame dim.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_padded = self.cond_encoder.encode(obs_for_encode, T_action=T_total)
        cond_packed = {k: v.squeeze(0) for k, v in cond_padded.items()}

        # 4. Build packed ctx and run.
        ctx = HNetContext(
            cond_dict=cond_packed,
            aux=[],
            inference_params=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
        )
        h = self.hnet(x_packed, ctx)  # (T_total, D)
        return action_head_raw(self, h), ctx.aux

    @torch.no_grad()
    def generate(
        self,
        obs: dict,
        batch_size: int,
        device,
        T: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout from BOS for ``T`` steps (default
        ``action_horizon``). ``T`` may be < ``action_horizon`` when rolling
        out an individual episode whose length is known to be shorter — the
        pos_emb is sized at action_horizon so any T <= action_horizon
        works."""
        if T is None:
            T = self.action_horizon
        # F6: no upper bound on T from a learned pos_emb anymore — RoPE
        # handles arbitrary positions inside the MHA. The previous check
        # `T > self.action_horizon` is removed. ``action_horizon`` is still
        # consulted as a *default rollout length* when ``T`` is None.
        cond_dict = self.cond_encoder.encode(obs, T)
        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        dtype = next(self.parameters()).dtype

        inference_params = self.hnet.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=T,
            device=device,
            dtype=dtype,
        )

        # Per-step cond_dict slice (B, d_cond) — AdaLN broadcasts over the
        # single-token sequence dim inside the encoder.
        def slice_cond(t: int) -> dict:
            return {k: v[:, t] if v.dim() == 3 else v for k, v in cond_dict.items()}

        # F6: no outer pos_emb add; RoPE inside MHA reads positions from each
        # KVCache's ``offsets`` (set by ``MultiHeadAttention.step``).
        cur = self.bos.expand(batch_size, -1, -1)
        for t in range(T):
            ctx = HNetContext(
                cond_dict=slice_cond(t),
                aux=[],
                inference_params=inference_params,
            )
            h = self.hnet.step(cur, ctx)
            a_t = action_head_decode(self, action_head_raw(self, h))
            actions[:, t : t + 1] = a_t
            if t < T - 1:
                cur = self.action_in(a_t)
        return actions

    # ----- Single-step inference API for closed-loop rollout -----

    @torch.no_grad()
    def init_step_state(self, batch_size: int, T_max: int, device, dtype=None) -> dict:
        """Allocate the AR inference state for ``batch_size`` parallel
        rollouts of at most ``T_max`` steps. The state is opaque to the
        caller; pass it back to :meth:`step` along with the current obs.
        """
        # F6: T_max is no longer clamped by self.action_horizon — RoPE
        # handles arbitrary positions. We still need ``T_max`` to size the
        # KV cache allocations.
        T_max = int(T_max)
        dtype = dtype or next(self.parameters()).dtype
        params = self.hnet.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=T_max,
            device=device,
            dtype=dtype,
        )
        # F6: BOS only — no outer pos_emb add. RoPE applies inside the MHA.
        cur = self.bos.expand(batch_size, 1, self.d_model).to(dtype)
        return {
            "params": params,
            "cur": cur,
            "dtype": dtype,
            "T_max": T_max,
        }

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int) -> torch.Tensor:
        """Single AR step. Returns normalized action ``(B, 1, action_dim)``.

        ``obs_norm`` is a dict of single-frame, normalized obs tensors
        (``state_agent_obj`` ``(B, D)``, ``front_img_1`` ``(B, C, H, W)``,
        …). Mutates ``state`` in place: replaces ``cur`` with the next
        step's input token.
        """
        cond_dict_seq = self.cond_encoder.encode(obs_norm, T_action=1)
        # Block step paths accept the (B, d_cond) view (AdaLN) or auto-
        # unsqueeze the (B, 1, d_cond) view (cross-attn).
        cond_2d = {k: v.squeeze(1) for k, v in cond_dict_seq.items()}
        ctx = HNetContext(
            cond_dict=cond_2d,
            aux=[],
            inference_params=state["params"],
        )
        h = self.hnet.step(state["cur"], ctx)
        a_t_norm = action_head_decode(self, action_head_raw(self, h))

        # F6: no per-step pos_emb add; the next-token input is just the
        # projected predicted action. RoPE is applied inside each MHA.step
        # using cache.offsets as the position.
        state["cur"] = self.action_in(a_t_norm).to(state["dtype"])
        return a_t_norm


class FlatFusedPolicy(nn.Module):
    """Flat transformer with interleaved [c_t, a_t] tokens.

    Drop-in replacement for ``HNetPolicy`` that bypasses the H-Net stage
    hierarchy (no chunker, no ratio loss). Each timestep contributes TWO
    tokens to the input sequence: a cond token and a (shifted) action token.
    Causal masking means the model predicting ``a_t`` (at sequence position
    2t+1) has seen ``c_0, BOS, c_1, a_0, c_2, a_1, ..., c_t, a_{t-1}``.

    Input layout (length 2T):
      x[:, 0]  = cond_in(c_0)         x[:, 1]  = BOS
      x[:, 2]  = cond_in(c_1)         x[:, 3]  = action_in(a_0)
      ...
      x[:, 2t]   = cond_in(c_t)       x[:, 2t+1] = action_in(a_{t-1})

    Output extraction:
      pred[:, t] = action_out(out[:, 2t+1])

    AR rollout walks the sequence one token at a time (2T model steps for T
    actions). Each "outer" step emits one action prediction and adds two
    tokens (the new cond + the new predicted action) to the cache.

    Same ``forward(actions, obs)`` / ``forward_packed(...)`` / ``generate(...)``
    contracts as ``HNetPolicy`` so the same ``HNet`` algo wrapper consumes
    it. Aux is always ``[]`` (no chunker contributions).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        arch_layout: str = "T8",
        num_heads: int = 4,
        d_intermediate: int = 512,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
        action_head_cfg: Optional[dict] = None,
        cond_mode: str = "adaln",
        spatial_cond_key: str = "spatial_cond_tokens",
        hist_spatial_mode: str = "current",
    ):
        super().__init__()
        from egomimic.models.hnet.isotropic_builder import build_isotropic

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.d_cond = d_cond
        self.cond_encoder = cond_encoder
        # cond_mode (run D): "adaln" (default) = input-layer fused cond token,
        # backbone unconditioned (byte-identical to before this knob). When
        # "cross_attn", the backbone blocks gain an extra cross-attention layer
        # whose K/V are the per-frame SPATIAL cond tokens (HPT-style image
        # localization). The pooled fused_cond input token is STILL used (so the
        # [c_0, BOS] layout / pos_emb are unchanged); cross-attn ADDS spatial
        # conditioning on top. Only the chunk paths (chunk_forward_history +
        # generate chunk_k>1) thread the spatial cond -- the AR/interleaved
        # paths are not used by run D and stay adaln-only.
        self.cond_mode = str(cond_mode)
        self.spatial_cond_key = str(spatial_cond_key)
        if self.cond_mode not in ("adaln", "cross_attn"):
            raise ValueError(f"FlatFusedPolicy: unknown cond_mode {self.cond_mode!r}")
        # hist_spatial_mode (D-fullhist-v2): how the obs-history chunk paths feed
        # SPATIAL cross-attn cond when n_obs_history N>1 (cross_attn only).
        #   "current" (default, byte-identical to run D / D-fullhist): ALL N+1
        #     tokens (incl. the N history cond tokens) cross-attend over the
        #     CURRENT frame's (frame N-1) M spatial tokens -- past frames get
        #     only their pooled fused_cond input token (a "gist"), no spatial
        #     detail. Built as a shared rank-3 (B, M, d_cond) set.
        #   "per_frame": token i (cond c_i) cross-attends over FRAME i's own M
        #     spatial tokens for i in 0..N-1; the trailing BOS (token N) reads
        #     the CURRENT frame (N-1). Built as a rank-4 (B, N+1, M, d_cond) set
        #     -> CrossAttention._forward_per_frame (each token attends ONLY its
        #     own frame). Makes every frame's processing IDENTICAL (same encoders,
        #     same spatial cross-attn), with BOS reading the present. At N=1 this
        #     degenerates to "current" (single frame is the current frame).
        self.hist_spatial_mode = str(hist_spatial_mode)
        if self.hist_spatial_mode not in ("current", "per_frame"):
            raise ValueError(
                f"FlatFusedPolicy: unknown hist_spatial_mode {self.hist_spatial_mode!r}"
            )

        self.action_in = nn.Linear(action_dim, d_model)
        configure_action_head(self, d_model, action_dim, action_head_cfg)
        self.cond_in = nn.Linear(d_cond, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)
        # pos_emb covers the 2T-token interleaved sequence.
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * action_horizon, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Single Isotropic stack, causal. adaln: no per-block cond (fusion at the
        # input layer). cross_attn: per-block cross-attention over the spatial
        # cond tokens (cond=True, cond_mode=cross_attn). Cross-attn causal is
        # auto-disabled inside the block when T_q != T_k (action/BOS tokens vs
        # M spatial tokens), giving HPT-style full attention to localize.
        _cross = self.cond_mode == "cross_attn"
        self.backbone = build_isotropic(
            {
                "arch_layout": arch_layout,
                "d_model": d_model,
                "d_intermediate": d_intermediate,
                "num_heads": num_heads,
                "cond": _cross,
                "cond_mode": self.cond_mode,
                "dropout": dropout,
                "resid_dropout": resid_dropout,
            },
            d_cond=(d_cond if _cross else 0),
            causal=True,
        )

    def _encode_cond(self, obs: dict, T: int) -> torch.Tensor:
        cond_dict = self.cond_encoder.encode(obs, T)
        c = cond_dict.get("fused_cond")
        if c is None:
            raise KeyError(
                "FlatFusedPolicy requires 'fused_cond' in cond_encoder output."
            )
        return c  # (B, T, d_cond)

    def _encode_spatial_cond(self, obs: dict, T: int):
        """Return the per-frame SPATIAL cond tokens (B, T, n_spatial, d_cond)
        for cross_attn mode, or None for adaln. Requires the cond_encoder to
        emit ``spatial_cond_key`` (a SpatialCondEncoderModule). Reuses the same
        cond_dict the pooled fused_cond came from -- one encode call upstream
        would suffice, but the chunk paths call _encode_cond separately, so we
        re-encode here (cheap: same image forward). Kept separate + gated so the
        adaln path never touches it."""
        if self.cond_mode != "cross_attn":
            return None
        cond_dict = self.cond_encoder.encode(obs, T)
        sp = cond_dict.get(self.spatial_cond_key)
        if sp is None:
            raise KeyError(
                f"cond_mode=cross_attn requires '{self.spatial_cond_key}' from the "
                "cond_encoder (use SpatialCondEncoderModule with spatial_img_encoders)."
            )
        return sp  # (B, T, n_spatial, d_cond)

    def _hist_spatial_cond(self, sp, N: int, dtype):
        """Build the SPATIAL cross-attn cond for the obs-history chunk paths
        (``chunk_forward_history`` + ``generate`` chunk_k>1), shared by both so
        train==eval. ``sp``: the encoded per-frame spatial tokens
        ``(B, N, M, d_cond)`` (from ``_encode_spatial_cond``) or ``None`` (adaln).

        Returns the ``cond`` arg for ``self.backbone(x, cond=...)`` over the
        length-``N+1`` token stream ``[c_0..c_{N-1}, BOS]``:
          - ``None`` if ``sp is None`` (adaln -> backbone byte-identical).
          - ``hist_spatial_mode="current"``: a shared rank-3 ``(B, M, d_cond)``
            == ``sp[:, N-1]`` (the current frame). EVERY token cross-attends over
            this single set. BYTE-IDENTICAL to the pre-v2 path.
          - ``hist_spatial_mode="per_frame"``: a rank-4 ``(B, N+1, M, d_cond)``
            where token i gets frame i's M tokens for i in 0..N-1 and the trailing
            BOS (token N) gets the CURRENT frame (N-1). CrossAttention's rank-4
            ``_forward_per_frame`` makes each token attend ONLY its own frame, so
            every frame is processed identically. At N=1 the two stacked frames
            are both frame 0 -> mathematically the same single set as "current".
        """
        if sp is None:
            return None
        if self.hist_spatial_mode == "current":
            return sp[:, N - 1].to(dtype)  # (B, M, d_cond) shared over all tokens
        # per_frame: stack frame i for token i (i<N) + current frame for the BOS.
        cur = sp[:, N - 1 : N]                       # (B, 1, M, d_cond)
        cond = torch.cat([sp, cur], dim=1)           # (B, N+1, M, d_cond)
        return cond.to(dtype)

    def chunk_forward_history(self, obs_hist: dict) -> torch.Tensor:
        """OBS-HISTORY chunk forward (teacher-forced, batched, WITH grad).

        ``obs_hist``: dict of per-key tensors carrying an N-frame history axis
        for the CURRENT chunk anchor. Each value is either ``(B, N, ...)`` (the
        last N obs frames, oldest->newest, frame N-1 == the current frame) or a
        single-frame ``(B, ...)`` (encoded as N=1). Returns the RAW chunk-head
        output ``(B, K, A[/params])`` read at a TRAILING BOS that has attended
        (causally) over all N cond tokens.

        Token layout (length N+1, NO interleaved action tokens -- OBS ONLY, so
        no past-action copying): ``[c_0, c_1, ..., c_{N-1}, BOS]`` with pos_emb
        indices ``0..N``. For N=1 this is ``[c_0, BOS]`` at pos_emb ``[0,1]`` --
        BYTE-IDENTICAL to the single-frame ``forward(dummy, obs)`` T=1 path
        (cond@0, BOS@1, read@BOS). The chunk loss consumes the raw params.
        """
        # _encode_cond returns (B, N, d_cond): a per-key (B, N, ...) obs is
        # encoded frame-wise; a single-frame (B, ...) obs broadcasts to T=1.
        any_v = next(iter(obs_hist.values()))
        N = any_v.shape[1] if any_v.dim() >= 3 else 1
        c = self._encode_cond(obs_hist, N)          # (B, N, d_cond)
        B = c.shape[0]
        c_tok = self.cond_in(c)                     # (B, N, d_model)
        bos = self.bos.expand(B, 1, -1)             # (B, 1, d_model)
        x = torch.cat([c_tok, bos], dim=1)          # (B, N+1, d_model)
        x = x + self.pos_emb[:, : N + 1].to(x.dtype)
        # cross_attn: spatial cross-attn cond over the [c_0..c_{N-1}, BOS] stream.
        #   "current" (default): every token reads the CURRENT frame's M tokens
        #     (shared (B, M, d_cond)) -- past frames contribute only their pooled
        #     fused_cond input token. adaln: cond=None -> backbone unchanged.
        #   "per_frame" (D-fullhist-v2): token i reads FRAME i's M tokens, BOS
        #     reads the current frame ((B, N+1, M, d_cond)) -> uniform per-frame
        #     spatial processing. See _hist_spatial_cond.
        _sp = self._encode_spatial_cond(obs_hist, N)
        _cond = self._hist_spatial_cond(_sp, N, x.dtype)
        x = self.backbone(x, cond=_cond)            # causal: BOS attends over c_0..c_{N-1}
        raw = action_head_raw(self, x[:, N : N + 1])  # (B, 1, K, A[/params])
        return raw[:, 0]                            # (B, K, A[/params])

    def forward(self, actions: torch.Tensor, obs: dict):
        """Padded teacher-forced forward.

        actions: (B, T, action_dim)
        obs:     dict of per-frame (B, T, ...) obs tensors.
        Returns: (pred (B, T, action_dim), aux=[]).
        """
        B, T, _ = actions.shape
        c = self._encode_cond(obs, T)  # (B, T, d_cond)
        c_tok = self.cond_in(c)  # (B, T, d_model)
        if getattr(self, "reactive", False):
            # REACTIVE: no obs/action history -- predict a_t from c_t alone,
            # each timestep an independent [cond, BOS] pair.
            _bos = self.bos.expand(B, T, -1)
            _x = torch.stack([c_tok, _bos], dim=2).reshape(B * T, 2, self.d_model)
            _x = _x + self.pos_emb[:, :2].to(_x.dtype)
            _x = self.backbone(_x)
            _pred = action_head_raw(self, _x[:, 1])
            return _pred.reshape(B, T, *_pred.shape[1:]), []
        a_tok = self.action_in(actions)  # (B, T, d_model)
        # Shift: BOS at position 0, a_0..a_{T-2} after.
        a_shifted = torch.cat([self.bos.expand(B, -1, -1), a_tok[:, :-1]], dim=1)
        a_shifted = _apply_token_dropout(
            a_shifted, self.bos.reshape(-1), getattr(self, "token_dropout_p", 0.0), self.training
        )

        # Interleave: x[:, 0::2] = c_tok, x[:, 1::2] = a_shifted.
        x = torch.empty(
            B, 2 * T, self.d_model, device=actions.device, dtype=a_tok.dtype
        )
        x[:, 0::2] = c_tok
        x[:, 1::2] = a_shifted
        x = x + self.pos_emb[:, : 2 * T].to(x.dtype)

        # cross_attn (run E): each of the 2T interleaved tokens cross-attends
        # over ITS OWN frame's M spatial tokens. Token 2t (cond) and 2t+1
        # (action->predicts a_t) both belong to frame t, so both get frame t's
        # spatial set. Build (B, 2T, M, d_cond) by repeating each frame's M
        # tokens twice along the sequence axis. adaln: cond stays None ->
        # self.backbone(x) byte-identical.
        _cond = self._per_frame_cond_interleaved(obs, T, x.dtype)
        x = self.backbone(x, cond=_cond)  # (B, 2T, d_model)
        pred = action_head_raw(self, x[:, 1::2])  # (B, T, action_dim[, n_bins])
        return pred, []

    def _per_frame_cond_interleaved(self, obs, T, dtype):
        """Build the per-token spatial cond for the 2T interleaved stream.

        Returns ``(B, 2T, M, d_cond)`` (each frame's M tokens repeated for its
        [cond, action] token pair) for cond_mode=cross_attn, else ``None`` so
        the adaln backbone call is byte-identical."""
        sp = self._encode_spatial_cond(obs, T)  # (B, T, M, d_cond) or None
        if sp is None:
            return None
        B, _T, M, dc = sp.shape
        # Repeat each frame twice along the seq axis: frame t -> positions 2t, 2t+1.
        cond = sp.unsqueeze(2).expand(B, T, 2, M, dc).reshape(B, 2 * T, M, dc)
        return cond.to(dtype)

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """Packed-mode teacher-forced forward.

        Builds a fused-token packed stream where each sub-sequence ``[s, e)``
        becomes a length-``2*(e-s)`` interleaved chunk. Returns predictions
        in packed format ``(T_total, action_dim)`` matching the input
        ``actions_packed`` layout.
        """
        device = actions_packed.device
        T_total = actions_packed.shape[0]
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        if max_seqlen > self.action_horizon:
            raise ValueError(
                f"max_seqlen={max_seqlen} exceeds action_horizon={self.action_horizon}"
            )

        # Encode cond. Feed obs as (1, T_total, ...) via cond_encoder.encode.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_seq = self._encode_cond(obs_for_encode, T_total).squeeze(
            0
        )  # (T_total, d_cond)
        c_tok = self.cond_in(cond_seq)  # (T_total, d_model)
        if getattr(self, "reactive", False):
            # REACTIVE (packed): no obs/action history. Each packed frame t
            # is an independent [c_t, BOS] 2-token sequence -> predict a_t
            # (or its K-chunk) from c_t alone. Matches generate()/step()'s
            # current-frame [c_t,BOS] @ pos 0/1 so train == eval. cu_seqlens
            # is irrelevant here (no cross-frame attention); the chunk loss
            # still uses cu_seqlens to window targets within each episode.
            _bos = self.bos.expand(T_total, 1, -1)  # (T_total, 1, d_model)
            _x = torch.cat([c_tok.unsqueeze(1), _bos], dim=1)  # (T_total, 2, d_model)
            _x = _x + self.pos_emb[:, :2].to(_x.dtype)
            _x = self.backbone(_x)  # plain batched, no cu_seqlens
            _pred = action_head_raw(self, _x[:, 1])  # (T_total, [K,] A)
            return _pred, []
        a_tok = self.action_in(actions_packed)  # (T_total, d_model)

        # Build BOS-shifted actions per sub-sequence.
        a_shifted = torch.empty_like(a_tok)
        bos = self.bos.squeeze(0).squeeze(0).to(a_tok.dtype)  # (d_model,)
        a_shifted[cu_seqlens[:-1]] = bos
        # For positions that aren't sub-seq starts, copy a_tok[t-1].
        non_start = torch.ones(T_total, dtype=torch.bool, device=device)
        non_start[cu_seqlens[:-1]] = False
        # Indices of non-start positions:
        idx_non_start = torch.nonzero(non_start, as_tuple=False).squeeze(-1)
        a_shifted[idx_non_start] = a_tok[idx_non_start - 1]
        a_shifted = _apply_token_dropout(
            a_shifted, bos, getattr(self, "token_dropout_p", 0.0), self.training
        )

        # Build per-sub-seq position indices: 0, 1, ..., (e-s)-1 within each
        # sub-seq. Then the 2-token interleave doubles to 2*(e-s).
        pos = torch.arange(T_total, device=device)
        seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)
        local_pos = pos - cu_seqlens[seq_idx]  # (T_total,)

        # The interleaved stream has 2T_total tokens. cond_t at 2*local_pos
        # within each sub-seq; action_t at 2*local_pos+1.
        # Compute new cu_seqlens for the doubled stream.
        sub_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        new_lens = 2 * sub_lens
        new_cu = torch.zeros(len(cu_seqlens), dtype=torch.long, device=device)
        new_cu[1:] = torch.cumsum(new_lens, dim=0)
        new_T_total = int(new_cu[-1].item())

        # Build the packed interleaved stream and apply pos_emb based on
        # 2*local_pos / 2*local_pos+1.
        x = torch.empty(new_T_total, self.d_model, device=device, dtype=a_tok.dtype)
        # For each t, write c_tok[t] at new_cu[seq_idx[t]] + 2*local_pos[t]
        # and a_shifted[t] at the next position.
        target_c = new_cu[seq_idx] + 2 * local_pos
        target_a = target_c + 1
        x[target_c] = c_tok
        x[target_a] = a_shifted
        # pos_emb indexing: doubled local positions within each sub-seq.
        # pos_emb is (1, 2*action_horizon, d_model). Apply pos_emb[2*local_pos]
        # to cond positions and pos_emb[2*local_pos+1] to action positions.
        pos_c = (2 * local_pos).clamp(max=2 * self.action_horizon - 1)
        pos_a = (2 * local_pos + 1).clamp(max=2 * self.action_horizon - 1)
        x[target_c] = x[target_c] + self.pos_emb[0, pos_c].to(x.dtype)
        x[target_a] = x[target_a] + self.pos_emb[0, pos_a].to(x.dtype)

        # cross_attn (run E): per-token spatial cond aligned to the doubled
        # packed stream. Each source frame t's M spatial tokens go to BOTH its
        # cond slot (target_c) and action slot (target_a). adaln -> None.
        _cond_packed = None
        if self.cond_mode == "cross_attn":
            sp = self._encode_spatial_cond(obs_for_encode, T_total)  # (1, T_total, M, dc)
            sp = sp.squeeze(0)  # (T_total, M, dc)
            M = sp.shape[1]
            dc = sp.shape[2]
            _cond_packed = torch.zeros(
                new_T_total, M, dc, device=device, dtype=a_tok.dtype
            )
            _cond_packed[target_c] = sp.to(a_tok.dtype)
            _cond_packed[target_a] = sp.to(a_tok.dtype)

        # Run the backbone on the doubled packed stream.
        out = self.backbone(
            x,
            cond=_cond_packed,
            cu_seqlens=new_cu,
            max_seqlen=2 * int(max_seqlen),
        )

        # Predictions: out at action positions (target_a). action_out projects
        # to (T_total, action_dim). Order matches actions_packed.
        pred = action_head_raw(self, out[target_a])
        return pred, []

    @torch.no_grad()
    def generate(
        self,
        obs: dict,
        batch_size: int,
        device,
        T: Optional[int] = None,
    ) -> torch.Tensor:
        """AR rollout for T action steps over a 2T-token interleaved stream.

        Each AR outer step does TWO inner step() calls: one for the cond
        token, one for the (predicted) action token. The output of the
        action-step is the next action prediction.
        """
        if T is None:
            T = self.action_horizon
        if T > self.action_horizon:
            raise ValueError(f"generate T={T} exceeds action_horizon")

        # chunk_k>1: return the trained chunk head's ONE-SHOT K-action plan from the
        # current obs, so chunk_te + temporal-ensemble uses ALL K actions (not just a_0).
        _K = getattr(self, "_act_chunk_k", 1)
        if _K > 1:
            # OBS-HISTORY: obs may carry an N-frame axis ((B, N, ...)). Encode N
            # cond tokens, step them at pos_emb 0..N-1, then BOS at pos_emb N and
            # read the chunk there. N=1 (single-frame obs) reproduces the original
            # [cond@0, BOS@1] two-token path BYTE-IDENTICALLY (train==eval).
            _any = next(iter(obs.values()))
            _N = _any.shape[1] if _any.dim() >= 3 else 1
            _c = self.cond_in(self._encode_cond(obs, _N))   # (B, N, d_model)
            _dt = _c.dtype
            if self.cond_mode == "cross_attn":
                # cross_attn: one-shot batched [c_0..c_{N-1}, BOS] forward with
                # the spatial cond tokens as cross-attn K/V -- IDENTICAL math to
                # chunk_forward_history (via the SHARED _hist_spatial_cond) so
                # eval == train, including hist_spatial_mode={current,per_frame}.
                # (The AR-step KV-cache cross-attn accumulates one cond/step,
                # which is for per-frame cond, NOT a fixed M-token spatial set;
                # the chunk head is a single one-shot read, so a plain forward is
                # correct.)
                _bos = self.bos.expand(batch_size, 1, -1).to(_dt)
                _x = torch.cat([_c, _bos], dim=1)            # (B, N+1, d_model)
                _x = _x + self.pos_emb[:, : _N + 1].to(_dt)
                _sp = self._encode_spatial_cond(obs, _N)
                _cond = self._hist_spatial_cond(_sp, _N, _dt)
                _x = self.backbone(_x, cond=_cond)
                _h = _x[:, _N : _N + 1]                      # (B, 1, d_model) at BOS
            else:
                _p = self.backbone.allocate_inference_cache(batch_size, _N + 1, device, _dt)
                for _i in range(_N):
                    self.backbone.step(
                        _c[:, _i : _i + 1] + self.pos_emb[:, _i : _i + 1].to(_dt), _p
                    )
                _h = self.backbone.step(
                    self.bos.expand(batch_size, -1, -1).to(_dt)
                    + self.pos_emb[:, _N : _N + 1].to(_dt),
                    _p,
                )
            # Decode the full K-step chunk (mode-committing for discrete/gmm;
            # identity for continuous, which reproduces the old explicit
            # reshape to (B, K, action_dim)). _h is (B, 1, d_model) so raw
            # carries a singleton T axis -> squeeze it before decode.
            _raw = action_head_raw(self, _h)  # (B, 1, K, action_dim[/n_bins/params])
            _chunk = action_head_decode_chunk(self, _raw.squeeze(1))  # (B, K, action_dim)
            return _chunk[:, :T]

        if getattr(self, "reactive", False):
            _rc = self._encode_cond(obs, 1)
            _rct = self.cond_in(_rc.squeeze(1)).unsqueeze(1)
            _rbos = self.bos.expand(batch_size, 1, -1)
            _rx = (torch.cat([_rct, _rbos], dim=1) + self.pos_emb[:, :2]).to(_rct.dtype)
            _rx = self.backbone(_rx)
            _ra = action_head_decode(self, action_head_raw(self, _rx[:, 1:2]))
            return _ra.expand(batch_size, T, self.action_dim)

        cond_seq = self._encode_cond(obs, T)  # (B, T, d_cond)
        c_tok = self.cond_in(cond_seq)  # (B, T, d_model)
        dtype = c_tok.dtype

        # cross_attn (run E): per-frame spatial cond (B, T, M, d_cond). Each AR
        # step feeds frame t's M tokens to BOTH the cond-step and action-step
        # (matches forward's per-frame cross-attn). adaln -> None.
        sp_seq = self._encode_spatial_cond(obs, T)  # (B, T, M, dc) or None

        # Allocate the backbone's inference cache sized for the doubled stream.
        params = self.backbone.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=2 * T,
            device=device,
            dtype=dtype,
        )

        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        a_prev = self.bos.expand(batch_size, -1, -1).to(dtype)  # (B, 1, d_model)
        for t in range(T):
            _cond_t = sp_seq[:, t].to(dtype) if sp_seq is not None else None  # (B,M,dc)
            # Cond step.
            x_c = c_tok[:, t : t + 1] + self.pos_emb[:, 2 * t : 2 * t + 1].to(dtype)
            _ = self.backbone.step(x_c, params, cond=_cond_t)
            # Action step.
            x_a = a_prev + self.pos_emb[:, 2 * t + 1 : 2 * t + 2].to(dtype)
            h = self.backbone.step(x_a, params, cond=_cond_t)
            a_t = action_head_decode(self, action_head_raw(self, h))  # (B, 1, action_dim)
            actions[:, t : t + 1] = a_t
            # Prepare next-step's a_prev (a_t becomes a_{t-1} for the next outer step).
            a_prev = self.action_in(a_t)

        return actions

    # ----- Single-step inference API for closed-loop rollout -----

    @torch.no_grad()
    def init_step_state(self, batch_size: int, T_max: int, device, dtype=None) -> dict:
        """Allocate the AR inference state. The flat fused policy runs a
        2-token-per-step interleaved stream, so the backbone cache is
        sized at ``2 * T_max``. ``a_prev_emb`` is the previous step's
        action embedding (BOS at t=0).
        """
        T_max = int(min(T_max, self.action_horizon))
        dtype = dtype or next(self.parameters()).dtype
        params = self.backbone.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=2 * T_max,
            device=device,
            dtype=dtype,
        )
        a_prev_emb = self.bos.expand(batch_size, 1, self.d_model).to(dtype)
        return {
            "params": params,
            "a_prev_emb": a_prev_emb,
            "dtype": dtype,
            "T_max": T_max,
        }

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int) -> torch.Tensor:
        """Single outer step. Internally runs two backbone steps: one
        for the cond token at position ``2t``, one for the action token
        at position ``2t+1``. Returns normalized action ``(B, 1,
        action_dim)``. Mutates ``state`` in place.
        """
        dtype = state["dtype"]
        if getattr(self, "reactive", False):
            _cs = self._encode_cond(obs_norm, T=1)
            _ct = self.cond_in(_cs.squeeze(1)).unsqueeze(1)
            _bos = self.bos.expand(_ct.shape[0], 1, -1)
            _x = (torch.cat([_ct, _bos], dim=1) + self.pos_emb[:, :2]).to(dtype)
            _x = self.backbone(_x)
            return action_head_decode(self, action_head_raw(self, _x[:, 1:2]))
        cond_seq = self._encode_cond(obs_norm, T=1)  # (B, 1, d_cond)
        fused = cond_seq.squeeze(1)  # (B, d_cond)
        # cross_attn (run E): current frame's M spatial tokens (B, M, d_cond),
        # fed to both the cond-step and action-step. adaln -> None.
        _sp = self._encode_spatial_cond(obs_norm, T=1)  # (B, 1, M, dc) or None
        _cond_t = _sp[:, 0].to(dtype) if _sp is not None else None
        c_tok = (
            self.cond_in(fused).unsqueeze(1) + self.pos_emb[:, 2 * t : 2 * t + 1]
        ).to(dtype)
        _ = self.backbone.step(c_tok, state["params"], cond=_cond_t)

        a_tok = (state["a_prev_emb"] + self.pos_emb[:, 2 * t + 1 : 2 * t + 2]).to(dtype)
        h = self.backbone.step(a_tok, state["params"], cond=_cond_t)
        a_t_norm = action_head_decode(self, action_head_raw(self, h))
        state["a_prev_emb"] = self.action_in(a_t_norm).to(dtype)
        return a_t_norm


class HNet(Algo):
    """
    H-Net policy Algo. Single-domain action-sequence model with per-frame
    obs conditioning -- each action token sees the obs at its own timestep.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        hnet: HNetCore,
        norm_stats,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        init_weights_range: Optional[float] = None,
        lr_multipliers: Optional[list] = None,
        use_parameter_groups: bool = False,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Training recipe knobs (all OFF by default — opt-in):

        - ``init_weights_range``: if set (e.g. ``0.02``), call
          ``hnet.init_weights(init_weights_range)`` after policy construction
          so ``out_proj`` / ``fc2`` weights get ``1/sqrt(n_residuals)``
          scaling.
        - ``lr_multipliers``: list of per-stage LR scales (outer→inner). If
          set, call ``hnet.apply_lr_multiplier(...)`` which stamps every
          parameter's ``_optim`` dict with ``lr_multiplier``.
        - ``use_parameter_groups``: if True, expose
          ``self.parameter_groups()`` so ``pl_model.configure_optimizers``
          builds AdamW param groups (bias / norm weights get
          ``weight_decay=0``; per-group ``lr = base_lr * lr_multiplier``).
        - ``weight_decay``: the base WD used when building parameter groups
          (only consulted when ``use_parameter_groups=True``). Outside of
          that, the optimizer config in the model YAML drives WD.

        Leaving all of these at their defaults reproduces "standard
        training": PyTorch default init, single LR for all params, single
        WD for all params from the optimizer config.
        """
        super().__init__()
        # ``norm_stats`` is a ``MultiDataset`` instance that owns the
        # per-embodiment per-feature normalization stats AND the key-topology
        # helpers (``keys_of_type``, ``is_key_with_embodiment``,
        # ``zarr_key_to_keyname``). pl_model._instantiate_model passes it in
        # automatically; the previous ``data_schematic`` parameter was legacy
        # and is gone.
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = action_horizon
        self.d_cond = d_cond
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Cache training-recipe knobs for configure_optimizers.
        self.use_parameter_groups = bool(use_parameter_groups)
        self.lr_multipliers = list(lr_multipliers) if lr_multipliers else None
        self.weight_decay = float(weight_decay)

        policy = HNetPolicy(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_model=d_model,
            cond_encoder=cond_encoder,
            hnet=hnet,
            action_head_cfg=_action_head_cfg(kwargs),
        )
        policy.token_dropout_p = float(kwargs.get("token_dropout_p", 0.0) or 0.0)
        self.train_mode = str(kwargs.get("train_mode", "tf"))
        self.ss_prob = float(kwargs.get("ss_prob", 0.5) or 0.5)
        _win = int(kwargs.get("window_size", 0) or 0)
        if _win > 0:
            for _m in policy.modules():
                if _m.__class__.__name__ == "MultiHeadAttention":
                    _m.window = _win
        # Apply opt-in training recipe BEFORE moving to device so the init
        # writes hit cpu params (matches upstream's pattern of init pre-move).
        if init_weights_range is not None:
            hnet.init_weights(initializer_range=float(init_weights_range))
        if self.lr_multipliers is not None:
            hnet.apply_lr_multiplier(self.lr_multipliers)
        # Stash the inner HNetCore so parameter_groups can reach it without
        # going through nets["policy"].hnet.
        self._hnet_core = hnet

        self.nets = nn.ModuleDict({"policy": policy})
        self.nets = self.nets.float().to(self.device)

        # Resolve per-embodiment keys via norm_stats (which owns the
        # MultiDataset key topology — same surface HPT uses).
        self.embodiment_ids = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        self.camera_keys = {}
        self.resolved_ac_keys = {}
        for emb in self.domains:
            emb_id = get_embodiment_id(emb)
            self.embodiment_ids[emb] = emb_id
            self.proprio_keys[emb_id] = []
            self.lang_keys[emb_id] = []
            self.camera_keys[emb_id] = []
            for key in norm_stats.keys_of_type("action_keys", emb_id):
                if (
                    norm_stats.is_key_with_embodiment(key, emb_id)
                    and key == self.ac_keys[emb]
                ):
                    self.resolved_ac_keys[emb_id] = key
            for key in norm_stats.keys_of_type("proprio_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.proprio_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.lang_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("camera_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.camera_keys[emb_id].append(key)

    # ---- Algo API --------------------------------------------------------

    # Keys emitted by ``pack_collate`` that aren't zarr-mapped data tensors
    # and must be passed through ``process_batch_for_training`` unchanged.
    _PACKED_META_KEYS = (
        "cu_seqlens",
        "max_seq_len",
        "seq_lens",
        "batch_size",
        "embodiment",
        "episode_idx",
        "chunk_offset",
    )

    @override
    def process_batch_for_training(self, batch):
        processed = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            processed[emb_id] = {}
            # Detect packed batches by the presence of cu_seqlens. Packed and
            # padded batches have a different key topology; treat them
            # separately so we don't try to keyname-resolve the meta keys.
            is_packed = "cu_seqlens" in _batch

            for key, value in _batch.items():
                if is_packed and key in self._PACKED_META_KEYS:
                    processed[emb_id][key] = value
                    continue
                key_name = self.norm_stats.zarr_key_to_keyname(key, emb_id)
                # Pre-existing typo: tested ``key is not None`` instead of
                # ``key_name``, which caused unrelated batch keys (e.g.
                # ``metadata.robot_name`` from _read_span) to be stored under
                # the None key. Skip silently when keyname can't be resolved.
                if key_name is not None:
                    processed[emb_id][key_name] = value

            ac_key = self.resolved_ac_keys[emb_id]
            # F5: H-Net does NOT consume ``pad_mask`` — variable-length is
            # carried by ``cu_seqlens`` in packed mode and by the full-length
            # convention in padded mode. The previous code populated an
            # all-ones ``pad_mask`` (or None) here as a vestige from the
            # ACT/HPT algos; no downstream H-Net code reads it. Don't
            # re-introduce it: if you need a true valid-token mask, plumb
            # it through ``HNetContext`` instead.
            processed[emb_id]["_packed"] = is_packed
            # Per-feature normalization via MultiDataset stats: each tensor
            # gets ``(x - mean) / std`` (or quantile equivalent) broadcast
            # against (action_dim,) / (proprio_dim,) stats. Works for both
            # padded ``(B, T, D)`` and packed ``(T_total, D)`` shapes.
            processed[emb_id] = self.norm_stats.normalize(processed[emb_id], emb_id)
            # Preserve goal_pose (goal_keys type) for the sim evaluator:
            # norm_stats.zarr_key_to_keyname returns None for it, so the
            # keyname loop above drops it. The closed-loop env init needs it.
            if "goal_pose" in _batch:
                processed[emb_id]["goal_pose"] = _batch["goal_pose"]
            processed[emb_id]["embodiment"] = torch.tensor(
                [emb_id], device=self.device, dtype=torch.int64
            )
            for key, value in processed[emb_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed[emb_id][key] = value
        return processed

    def _build_obs(self, _batch, emb_id):
        obs = {}
        for key in (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        ):
            if key in _batch:
                obs[key] = _batch[key]
        return obs

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            obs = self._build_obs(_batch, emb_id)

            is_packed = _batch.get("_packed", False)
            cu_seqlens = _batch["cu_seqlens"] if is_packed else None
            max_seqlen = int(_batch["max_seq_len"]) if is_packed else None

            def _fwd(acts):
                if is_packed:
                    return policy.forward_packed(acts, obs, cu_seqlens, max_seqlen)
                return policy(acts, obs)

            if getattr(self, "train_mode", "tf") == "ar":
                # Scheduled-sampling AR training: pass 1 teacher-forced (no grad)
                # gives the model's own action predictions; pass 2 substitutes
                # them for the GT prev-action inputs at rate ss_prob so the model
                # learns to recover from its own errors. Loss target stays GT.
                with torch.no_grad():
                    pred1, _ = _fwd(actions)
                    # Decode to continuous before feeding back as prev-action
                    # input (pred1 is logits when the head is discrete).
                    pred1 = action_head_decode(policy, pred1)
                _ss = float(getattr(self, "ss_prob", 0.5))
                _mask = torch.rand(actions.shape[:-1] + (1,), device=actions.device) < _ss
                mixed = torch.where(_mask, pred1.detach(), actions)
                pred, aux = _fwd(mixed)
            else:
                pred, aux = _fwd(actions)

            aloss = action_head_loss(policy, pred, actions, cu_seqlens=cu_seqlens)
            rloss = ratio_loss_from_aux(aux, device=aloss.device)
            predictions[f"{emb_id}_pred"] = pred
            predictions[f"{emb_id}_action_loss"] = aloss
            predictions[f"{emb_id}_ratio_loss"] = rloss

            # Per-chunker stats for logging. avg_chunk_len = T / max(#boundaries, 1)
            # so it == 1/F when F>0 and falls back to T when F==0 (no compression).
            stats = chunk_stats_from_aux(aux)
            for k, v in stats.items():
                predictions[f"{emb_id}_{k}"] = torch.tensor(v, device=aloss.device)
        return predictions

    @override
    def forward_eval(self, batch):
        """Per-frame teacher-forced eval.

        For each batch entry we run a SINGLE forward pass with the GT
        action stream (same as training) and compare per-frame predictions
        against GT in raw / unnormalized action space. AR rollout used to
        live here but is pointless when the obs sequence is fixed: the
        predicted action doesn't change what the model sees next, so AR
        just compounds exposure-bias error without testing anything
        useful. To test closed-loop behavior, run a separate
        envrollout evaluator that steps a simulator with predicted
        actions.

        Returns a dict with keys ``emb{id}_{ac_key}`` carrying
        ``(B, T_max, action_dim)`` unnormalized predictions (zero-padded
        past each episode's length) and ``emb{id}_seq_lens`` ``(B,)`` so
        downstream metric code can mask the padded positions.
        """
        unnorm = {}
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            if _batch.get("_packed", False):
                preds_padded, seq_lens = self._teacher_forced_packed(_batch, emb_id)
                preds = OrderedDict()
                preds[ac_key] = preds_padded
                unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
                for key, val in unnorm_actions.items():
                    unnorm[f"emb{emb_id}_{key}"] = val
                unnorm[f"emb{emb_id}_seq_lens"] = seq_lens
                continue
            # Padded mode (legacy): no packed dataset wired but kept for
            # completeness.
            obs = self._build_obs(_batch, emb_id)
            actions = _batch[ac_key]
            pred, _ = policy(actions, obs)
            pred = action_head_decode(policy, pred)
            preds = OrderedDict()
            preds[ac_key] = pred
            unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
            for key, val in unnorm_actions.items():
                unnorm[f"emb{emb_id}_{key}"] = val
        return unnorm

    @torch.no_grad()
    def _teacher_forced_packed(self, _batch: dict, emb_id: int):
        """Single-pass teacher-forced eval for a packed validation batch.

        Runs ``policy.forward_packed`` (the same path used in training)
        and unpacks the resulting ``(T_total, action_dim)`` predictions
        into ``(B, T_max, action_dim)`` zero-padded per-episode for
        downstream metric / viz code.
        """
        policy = self.nets["policy"]
        ac_key = self.resolved_ac_keys[emb_id]
        actions = _batch[ac_key]
        obs = self._build_obs(_batch, emb_id)
        cu = _batch["cu_seqlens"]
        max_seqlen = int(_batch["max_seq_len"])
        seq_lens = _batch["seq_lens"].clone()
        pred_packed, _ = policy.forward_packed(actions, obs, cu, max_seqlen)
        pred_packed = action_head_decode(policy, pred_packed)

        B = int(seq_lens.shape[0])
        T_max = int(seq_lens.max().item())
        action_dim = policy.action_dim
        preds_padded = torch.zeros(
            B,
            T_max,
            action_dim,
            device=pred_packed.device,
            dtype=pred_packed.dtype,
        )
        for b in range(B):
            s = int(cu[b].item())
            e = int(cu[b + 1].item())
            preds_padded[b, : e - s] = pred_packed[s:e]
        return preds_padded, seq_lens

    @torch.no_grad()
    def _ar_rollout_packed(self, _batch: dict, emb_id: int):
        """Per-episode AR rollout for a packed validation batch.

        For each sub-sequence ``[s, e)`` in ``cu_seqlens``:
          1. Slice that episode's obs into ``(1, T_ep, ...)``.
          2. Call ``policy.generate(obs_ep, batch_size=1, T=T_ep)`` to AR
             rollout exactly ``T_ep`` steps from BOS.
          3. Stash the prediction.

        Returns:
            preds_padded: ``(B, T_max, action_dim)`` (zero-padded past each
                episode's length).
            seq_lens:     ``(B,)`` long, the per-episode rollout lengths
                (matches ``_batch['seq_lens']`` and used for masking the
                padding in downstream MSE).
        """
        policy = self.nets["policy"]
        cu = _batch["cu_seqlens"]
        seq_lens = _batch["seq_lens"].clone()
        B = int(seq_lens.shape[0])
        T_max = int(seq_lens.max().item())
        action_dim = policy.action_dim
        device = self.device

        # Gather the obs keys we need for the cond encoder.
        obs_keys = (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        )
        obs_keys = [k for k in obs_keys if k in _batch]

        preds_padded = torch.zeros(B, T_max, action_dim, device=device)

        for b in range(B):
            s = int(cu[b].item())
            e = int(cu[b + 1].item())
            T_ep = e - s
            # Slice each obs key to the episode's range and add a leading
            # batch dim. The packed tensor is (T_total, ...) so slicing along
            # dim 0 gives (T_ep, ...) → unsqueeze → (1, T_ep, ...).
            obs_ep = {k: _batch[k][s:e].unsqueeze(0) for k in obs_keys}
            a_ep = policy.generate(
                obs_ep,
                batch_size=1,
                device=device,
                T=T_ep,
            )  # (1, T_ep, action_dim)
            preds_padded[b, :T_ep] = a_ep.squeeze(0)

        return preds_padded, seq_lens

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for emb_id in batch.keys():
            a = predictions[f"{emb_id}_action_loss"]
            r = predictions[f"{emb_id}_ratio_loss"]
            loss_dict[f"emb{emb_id}_action_loss"] = a
            loss_dict[f"emb{emb_id}_ratio_loss"] = r
            # Ratio-loss weights are baked into each chunker stage, so r
            # is already a properly-weighted sum.
            total = total + a + r

            # Pass non-loss stats through to logging (boundary_rate /
            # avg_chunk_len, per-chunker and aggregate). They are 0-dim
            # tensors so log_info.item() still works.
            for key, value in predictions.items():
                prefix = f"{emb_id}_"
                if not key.startswith(prefix):
                    continue
                tail = key[len(prefix) :]
                if tail in ("pred", "action_loss", "ratio_loss"):
                    continue
                loss_dict[f"emb{emb_id}_{tail}"] = value
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log

    # ----- Sim eval hook (SimRolloutEval.inference_step contract) ----- #
    # Single entry point. t=0 is the universal reset signal (allocates a
    # fresh AR state); t>0 steps against the cached state. The eval class
    # never touches model state — it lives entirely in self._sim_state.

    @torch.no_grad()
    def inference_step(
        self, obs_zarr: dict, t: int, emb_id: int
    ) -> "np.ndarray":
        """One closed-loop sim step.

        Args:
            obs_zarr: env obs in canonical zarr-key dict (already on device).
            t: timestep within the rollout. t=0 resets state.
            emb_id: embodiment id.

        Returns:
            absolute-frame action as np.float32 of shape (action_dim,).
        """
        import numpy as np
        policy = self.nets["policy"]
        if t == 0:
            device = next(self.nets["policy"].parameters()).device
            T_max = int(getattr(policy, "action_horizon", 1024))
            self._sim_state = policy.init_step_state(
                batch_size=1, T_max=T_max, device=device
            )
        embodiment_name = get_embodiment(emb_id).lower()
        ac_key = self.ac_keys[embodiment_name] if embodiment_name in self.ac_keys else self.ac_keys[emb_id]
        obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
        action_norm = policy.step(self._sim_state, obs_norm, t)
        action_unnorm = self.norm_stats.unnormalize(
            {ac_key: action_norm.squeeze(0).squeeze(0)}, emb_id,
        )[ac_key]
        return action_unnorm.detach().cpu().numpy().reshape(-1).astype(np.float32)

    # ----- Optional training recipe hook for pl_model.configure_optimizers ----- #

    def parameter_groups(self, base_lr: float):
        """Return AdamW-ready ``list[dict]`` if ``use_parameter_groups``,
        else ``None`` (caller falls back to ``self.nets.parameters()``).

        Groups are built by the inner HNet stage tree via
        ``HNetCore.parameter_groups(weight_decay=self.weight_decay)``, then
        each group's ``lr`` is set to ``base_lr * lr_multiplier``. Params
        that aren't part of the HNet stage tree (e.g. ``action_in``,
        ``action_out``, ``cond_encoder``, ``bos``, ``pos_emb``) are added in
        a single extra group with ``lr_multiplier=1.0`` so optimizer
        instantiation still sees every learnable parameter exactly once.
        """
        if not self.use_parameter_groups:
            return None

        # Groups for params inside the HNet stage tree.
        groups = self._hnet_core.parameter_groups(weight_decay=self.weight_decay)
        for g in groups:
            g["lr"] = float(base_lr) * float(g.get("lr_multiplier", 1.0))

        # Extra group for everything *not* inside the HNet stage tree.
        hnet_param_ids = {id(p) for g in groups for p in g["params"]}
        extra_params, extra_bias_norm = [], []
        for name, p in self.nets.named_parameters():
            if id(p) in hnet_param_ids or not p.requires_grad:
                continue
            # Bias / norm-weight detection (same rule as parameter_groups).
            if name.endswith(".bias") or ".norm" in name or "rmsnorm" in name.lower():
                extra_bias_norm.append(p)
            else:
                extra_params.append(p)
        if extra_params:
            groups.append(
                {
                    "params": extra_params,
                    "lr": float(base_lr),
                    "lr_multiplier": 1.0,
                    "weight_decay": self.weight_decay,
                }
            )
        if extra_bias_norm:
            groups.append(
                {
                    "params": extra_bias_norm,
                    "lr": float(base_lr),
                    "lr_multiplier": 1.0,
                    "weight_decay": 0.0,
                }
            )
        return groups


class HNetFused(HNet):
    """Flat-transformer (no chunker) variant: interleaved [c_t, a_t] tokens.

    Same Algo contract as :class:`HNet`. Replaces the H-Net stage hierarchy
    with a single :class:`FlatFusedPolicy` (one Isotropic stack over a 2T
    interleaved input). ``aux`` is always ``[]`` so ratio_loss is 0 and
    chunk_stats is empty; logging surfaces only action_loss.

    Reuses HNet's ``process_batch_for_training`` / ``forward_training`` /
    ``forward_eval`` / ``compute_losses`` / ``log_info`` /
    ``_ar_rollout_packed`` unchanged because the policy interface matches
    HNetPolicy verbatim.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        norm_stats,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        arch_layout: str = "T8",
        num_heads: int = 4,
        d_intermediate: int = 512,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
        **kwargs,
    ):
        # Skip HNet.__init__ — it requires a HNetCore. We re-implement the
        # tiny init here for the flat path.
        Algo.__init__(self)
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = action_horizon
        self.d_cond = d_cond
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_parameter_groups = False
        self.lr_multipliers = None
        self.weight_decay = 0.0
        self._hnet_core = None
        self.action_dim = action_dim

        policy = FlatFusedPolicy(
            action_dim=action_dim,
            action_horizon=action_horizon,
            d_model=d_model,
            d_cond=d_cond,
            cond_encoder=cond_encoder,
            arch_layout=arch_layout,
            num_heads=num_heads,
            d_intermediate=d_intermediate,
            dropout=dropout,
            resid_dropout=resid_dropout,
            action_head_cfg=_action_head_cfg(kwargs),
            cond_mode=str(kwargs.get("cond_mode", "adaln")),
            spatial_cond_key=str(
                kwargs.get("spatial_cond_key", "spatial_cond_tokens")
            ),
            hist_spatial_mode=str(kwargs.get("hist_spatial_mode", "current")),
        )
        policy.token_dropout_p = float(kwargs.get("token_dropout_p", 0.0) or 0.0)
        policy.reactive = bool(kwargs.get("reactive", False))
        self.train_mode = str(kwargs.get("train_mode", "tf"))
        self.ss_prob = float(kwargs.get("ss_prob", 0.5) or 0.5)
        _win = int(kwargs.get("window_size", 0) or 0)
        if _win > 0:
            for _m in policy.modules():
                if _m.__class__.__name__ == "MultiHeadAttention":
                    _m.window = _win
        self.nets = nn.ModuleDict({"policy": policy})
        self.nets = self.nets.float().to(self.device)

        # Replicate HNet's key-resolution loop (norm_stats-based).
        self.embodiment_ids = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        self.camera_keys = {}
        self.resolved_ac_keys = {}
        for emb in self.domains:
            emb_id = get_embodiment_id(emb)
            self.embodiment_ids[emb] = emb_id
            self.proprio_keys[emb_id] = []
            self.lang_keys[emb_id] = []
            self.camera_keys[emb_id] = []
            for key in norm_stats.keys_of_type("action_keys", emb_id):
                if (
                    norm_stats.is_key_with_embodiment(key, emb_id)
                    and key == self.ac_keys[emb]
                ):
                    self.resolved_ac_keys[emb_id] = key
            for key in norm_stats.keys_of_type("proprio_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.proprio_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.lang_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("camera_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.camera_keys[emb_id].append(key)
